#!/usr/bin/env python3
"""Probe Qwen2.5-Omni audio tower features with a Whisper decoder.

This is an experimental script. It bypasses the Qwen LLM path, captures the
Qwen audio tower hidden states, and feeds them to a Whisper decoder as encoder
outputs. The goal is to test whether the AUT/audio_tower representation is
directly usable for ASR, before adding any server integration.

Example:
    python scripts/probe_asr_decoder.py \
        --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
        --whisper-dir openai/whisper-large-v3 \
        --audio data/eval/audio/window/window_001.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Iterable

import torch

if TYPE_CHECKING:
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen audio_tower hidden states with a Whisper decoder."
    )
    parser.add_argument("--model-dir", required=True, help="Qwen2.5-Omni model directory.")
    parser.add_argument(
        "--whisper-dir",
        default="openai/whisper-large-v3",
        help="Whisper model name or local path.",
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--audio", help="Single audio file path to transcribe.")
    input_group.add_argument("--eval-file", help="Eval JSON file with query/query_audio fields.")
    input_group.add_argument("--eval-dir", help="Directory containing *_test.json eval files.")
    parser.add_argument(
        "--audio-root",
        default="data/eval",
        help="Base directory for relative query_audio paths in eval JSON files.",
    )
    parser.add_argument("--output", help="Write batch probe rows as JSONL.")
    parser.add_argument("--limit", type=int, help="Limit number of eval rows to probe.")
    parser.add_argument("--language", default="zh", help="Whisper language hint.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--hook-layer",
        default="ln_post",
        choices=["ln_post", "avg_pooler", "full"],
        help=(
            "ln_post captures pre-pooling/pre-projection 1280-dim features; "
            "avg_pooler captures pooled 1280-dim features; full captures the "
            "full audio_tower output, which is usually 2048-dim and incompatible."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to from_pretrained for both models.",
    )
    parser.add_argument(
        "--print-structure",
        action="store_true",
        help="Print top-level model module structure and exit.",
    )
    parser.add_argument(
        "--print-audio-shapes",
        action="store_true",
        help=(
            "Run one audio sample through Qwen and print audio_tower module output "
            "tensor shapes. This does not load Whisper or decode ASR."
        ),
    )
    parser.add_argument(
        "--shape-min-rank",
        type=int,
        default=3,
        help="Minimum tensor rank to print in --print-audio-shapes mode.",
    )
    parser.add_argument(
        "--shape-name-filter",
        default="",
        help="Optional substring filter for module names in --print-audio-shapes mode.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def first_param_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move BatchEncoding tensors while preserving integer token tensors."""
    out = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def find_audio_tower(model: torch.nn.Module) -> torch.nn.Module:
    candidates = [
        "thinker.audio_tower",
        "thinker.model.audio_tower",
        "audio_tower",
    ]
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            print(f"[probe] found audio tower: model.{path}")
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find audio_tower. Try --print-structure.")


def print_structure(model: torch.nn.Module) -> None:
    print("\n=== model named_modules, depth <= 2 ===")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            indent = "  " * name.count(".")
            print(f"{indent}{name or '<root>'}: {type(module).__name__}")


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def print_audio_tower_shapes(
    qwen_model: Qwen2_5OmniForConditionalGeneration,
    qwen_processor: Qwen2_5OmniProcessor,
    audio_tower: torch.nn.Module,
    audio_path: str,
    dtype: torch.dtype,
    min_rank: int,
    name_filter: str,
) -> None:
    qwen_device = first_param_device(qwen_model)
    inputs = build_qwen_inputs(qwen_processor, audio_path, qwen_device, dtype)
    print(f"[shape] audio={audio_path}")
    if "input_features" in inputs:
        print(f"[shape] input_features={tuple(inputs['input_features'].shape)}")
    if "feature_attention_mask" in inputs:
        print(f"[shape] feature_attention_mask={tuple(inputs['feature_attention_mask'].shape)}")

    records: list[dict[str, Any]] = []
    handles = []

    def make_hook(module_name: str, module_type: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None or tensor.ndim < min_rank:
                return
            records.append(
                {
                    "name": module_name,
                    "type": module_type,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "device": str(tensor.device),
                }
            )

        return hook

    for name, module in audio_tower.named_modules():
        module_name = f"audio_tower.{name}" if name else "audio_tower"
        if name_filter and name_filter not in module_name:
            continue
        handles.append(module.register_forward_hook(make_hook(module_name, type(module).__name__)))

    try:
        with torch.inference_mode():
            qwen_model.thinker(
                **{key: value for key, value in inputs.items() if key != "labels"},
                output_hidden_states=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if not records:
        print("[shape] no matching tensor outputs captured")
        return

    print("\n[shape] captured audio_tower outputs")
    print(f"{'name':72} {'type':34} {'shape':22} {'dtype':10} device")
    print("-" * 150)
    for row in records:
        shape = "[" + ",".join(str(part) for part in row["shape"]) + "]"
        print(f"{row['name'][:72]:72} {row['type'][:34]:34} {shape:22} {row['dtype']:10} {row['device']}")


def select_hook_target(audio_tower: torch.nn.Module, hook_layer: str) -> torch.nn.Module:
    if hook_layer == "ln_post" and hasattr(audio_tower, "ln_post"):
        print("[probe] hook target: audio_tower.ln_post, expected dim=1280")
        return audio_tower.ln_post
    if hook_layer == "avg_pooler" and hasattr(audio_tower, "avg_pooler"):
        print("[probe] hook target: audio_tower.avg_pooler, expected dim=1280")
        return audio_tower.avg_pooler
    print("[probe] hook target: full audio_tower, often dim=2048")
    return audio_tower


def resolve_whisper_dir(whisper_dir: str) -> str:
    if Path(whisper_dir).exists():
        return whisper_dir

    modelscope_ids = {
        "openai/whisper-large-v3": "AI-ModelScope/whisper-large-v3",
        "openai/whisper-large-v2": "AI-ModelScope/whisper-large-v2",
        "openai/whisper-medium": "AI-ModelScope/whisper-medium",
    }
    ms_id = modelscope_ids.get(whisper_dir, whisper_dir)

    fallback_cache = Path(os.path.expanduser("~/.cache/modelscope/hub/models")) / ms_id
    if fallback_cache.is_dir() and (fallback_cache / "config.json").exists():
        print(f"[probe] using cached Whisper from modelscope: {fallback_cache}")
        return str(fallback_cache)

    try:
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"[probe] downloading Whisper via modelscope: {ms_id}")
        return snapshot_download(ms_id)
    except Exception as exc:
        if fallback_cache.is_dir() and (fallback_cache / "config.json").exists():
            print(f"[probe] download failed but cache exists: {fallback_cache}")
            return str(fallback_cache)
        print(f"[probe] modelscope unavailable ({exc}); falling back to {whisper_dir}")
        return whisper_dir


def build_qwen_inputs(
    processor: Qwen2_5OmniProcessor,
    audio_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from qwen_omni_utils import process_mm_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an assistant."}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return move_inputs(inputs, device=device, dtype=dtype)


def capture_audio_hidden(
    qwen_model: Qwen2_5OmniForConditionalGeneration,
    hook_target: torch.nn.Module,
    inputs: dict[str, Any],
) -> torch.Tensor:
    encoder_hidden: torch.Tensor | None = None

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal encoder_hidden
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        else:
            tensor = output
        if not torch.is_tensor(tensor):
            raise TypeError(f"Hook output is not a tensor: {type(tensor).__name__}")
        encoder_hidden = tensor.detach().float().cpu()

    handle = hook_target.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            qwen_model.thinker(
                **{key: value for key, value in inputs.items() if key != "labels"},
                output_hidden_states=False,
            )
    finally:
        handle.remove()

    if encoder_hidden is None:
        raise RuntimeError("Hook did not capture any output.")
    if encoder_hidden.ndim == 2:
        encoder_hidden = encoder_hidden.unsqueeze(0)
    if encoder_hidden.ndim != 3:
        raise RuntimeError(f"Expected hidden shape [B, T, C], got {tuple(encoder_hidden.shape)}")
    return encoder_hidden


def decode_with_whisper(
    encoder_hidden: torch.Tensor,
    whisper_model: WhisperForConditionalGeneration,
    whisper_processor: WhisperProcessor,
    language: str,
    max_new_tokens: int,
) -> str:
    from transformers.modeling_outputs import BaseModelOutput

    encoder_dim = encoder_hidden.shape[-1]
    whisper_dim = whisper_model.config.d_model
    if encoder_dim != whisper_dim:
        print("[warn] hidden dim mismatch; Whisper decoder output is likely invalid.")

    whisper_device = first_param_device(whisper_model)
    encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_hidden.to(
            whisper_device,
            dtype=first_param_dtype(whisper_model),
        )
    )
    attention_mask = torch.ones(
        encoder_hidden.shape[:2],
        dtype=torch.long,
        device=whisper_device,
    )
    forced_ids = whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.inference_mode():
        generated = whisper_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_ids,
            max_new_tokens=max_new_tokens,
        )
    return whisper_processor.batch_decode(generated, skip_special_tokens=True)[0]


def first_param_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def load_whisper(
    whisper_dir: str,
    dtype: torch.dtype,
    device_map: str,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    whisper_source = resolve_whisper_dir(whisper_dir)
    print(f"[4/4] loading Whisper from {whisper_source}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_source,
        torch_dtype=dtype,
        device_map=device_map,
    )
    whisper_model.eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_source)
    print(f"[probe] Whisper d_model={whisper_model.config.d_model}")
    return whisper_model, whisper_processor


def load_qwen(
    model_dir: str,
    dtype: torch.dtype,
    device_map: str,
) -> tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"[1/4] loading Qwen2.5-Omni from {model_dir}")
    qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
    )
    qwen_model.eval()
    qwen_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    return qwen_model, qwen_processor


def is_repetitive(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 8:
        return False
    for size in range(1, min(6, len(compact) // 2) + 1):
        unit = compact[:size]
        repeated = unit * (len(compact) // size)
        if compact.startswith(repeated) and len(repeated) >= len(compact) * 0.8:
            return True
    chars = list(compact)
    if not chars:
        return False
    most_common = max(chars.count(ch) for ch in set(chars))
    return most_common / len(chars) >= 0.65


def char_similarity(expected: str, actual: str) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(None, expected, actual).ratio()


def iter_eval_rows(eval_paths: Iterable[Path], audio_root: Path) -> Iterable[dict[str, Any]]:
    for eval_path in eval_paths:
        with eval_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Eval file must contain a list: {eval_path}")
        for row in rows:
            query_audio = row.get("query_audio")
            if not query_audio:
                continue
            audio_path = Path(query_audio)
            if not audio_path.is_absolute():
                audio_path = audio_root / audio_path
            yield {
                "id": row.get("id"),
                "eval_file": str(eval_path),
                "query": row.get("query", ""),
                "audio": str(audio_path),
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
                "intent": row.get("intent"),
            }


def collect_probe_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.audio:
        return [{"id": Path(args.audio).stem, "query": "", "audio": args.audio}]
    audio_root = Path(args.audio_root).expanduser()
    if args.eval_file:
        eval_paths = [Path(args.eval_file).expanduser()]
    elif args.eval_dir:
        eval_paths = sorted(Path(args.eval_dir).expanduser().glob("*_test.json"))
    else:
        raise ValueError("One of --audio, --eval-file, or --eval-dir is required.")

    items = list(iter_eval_rows(eval_paths, audio_root))
    if args.limit is not None:
        items = items[: args.limit]
    return items


def probe_one(
    item: dict[str, Any],
    qwen_model: Qwen2_5OmniForConditionalGeneration,
    qwen_processor: Qwen2_5OmniProcessor,
    hook_target: torch.nn.Module,
    whisper_model: WhisperForConditionalGeneration,
    whisper_processor: WhisperProcessor,
    dtype: torch.dtype,
    language: str,
    max_new_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    audio_path = str(Path(item["audio"]).expanduser().resolve())
    if verbose:
        print(f"[2/4] processing audio: {audio_path}")
    qwen_device = first_param_device(qwen_model)
    inputs = build_qwen_inputs(qwen_processor, audio_path, qwen_device, dtype)
    if verbose and "input_features" in inputs:
        print(f"[probe] input_features shape={tuple(inputs['input_features'].shape)}")
    if verbose and "feature_attention_mask" in inputs:
        print(f"[probe] feature_attention_mask shape={tuple(inputs['feature_attention_mask'].shape)}")

    if verbose:
        print("[3/4] running Qwen thinker and capturing audio hidden states")
    encoder_hidden = capture_audio_hidden(qwen_model, hook_target, inputs)
    stats = encoder_hidden.float()
    hidden_shape = tuple(encoder_hidden.shape)
    if verbose:
        print(
            "[probe] hidden "
            f"shape={hidden_shape} "
            f"dtype={encoder_hidden.dtype} "
            f"mean={stats.mean().item():.6f} "
            f"std={stats.std().item():.6f}"
        )

    text = decode_with_whisper(
        encoder_hidden=encoder_hidden,
        whisper_model=whisper_model,
        whisper_processor=whisper_processor,
        language=language,
        max_new_tokens=max_new_tokens,
    )
    query = item.get("query", "")
    result = {
        **item,
        "audio": audio_path,
        "asr": text,
        "empty": not bool(text.strip()),
        "repetitive": is_repetitive(text),
        "char_similarity": round(char_similarity(query, text), 4) if query else None,
        "hidden_shape": list(hidden_shape),
        "hidden_mean": round(stats.mean().item(), 6),
        "hidden_std": round(stats.std().item(), 6),
    }
    return result


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)

    qwen_model, qwen_processor = load_qwen(args.model_dir, dtype, args.device_map)

    if args.print_structure:
        print_structure(qwen_model)
        return

    items = collect_probe_items(args)
    if not items:
        raise ValueError("No audio items found.")

    audio_tower = find_audio_tower(qwen_model)
    if args.print_audio_shapes:
        if not args.audio:
            raise ValueError("--print-audio-shapes requires --audio")
        print_audio_tower_shapes(
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            audio_tower=audio_tower,
            audio_path=str(Path(args.audio).expanduser().resolve()),
            dtype=dtype,
            min_rank=args.shape_min_rank,
            name_filter=args.shape_name_filter,
        )
        return

    hook_target = select_hook_target(audio_tower, args.hook_layer)
    whisper_model, whisper_processor = load_whisper(args.whisper_dir, dtype, args.device_map)

    output_f = None
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_f = output_path.open("w", encoding="utf-8")

    total = 0
    empty = 0
    repetitive = 0
    sim_sum = 0.0
    sim_count = 0
    try:
        for idx, item in enumerate(items, start=1):
            verbose = len(items) == 1
            print(f"[{idx}/{len(items)}] {item.get('id') or Path(item['audio']).name}")
            result = probe_one(
                item=item,
                qwen_model=qwen_model,
                qwen_processor=qwen_processor,
                hook_target=hook_target,
                whisper_model=whisper_model,
                whisper_processor=whisper_processor,
                dtype=dtype,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
                verbose=verbose,
            )
            total += 1
            empty += int(result["empty"])
            repetitive += int(result["repetitive"])
            if result["char_similarity"] is not None:
                sim_sum += result["char_similarity"]
                sim_count += 1
            print(
                "  "
                f"gt={result.get('query')!r} asr={result['asr']!r} "
                f"sim={result['char_similarity']} empty={result['empty']} "
                f"repeat={result['repetitive']} hidden={result['hidden_shape']}"
            )
            if output_f:
                output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_f.flush()
    finally:
        if output_f:
            output_f.close()

    avg_sim = sim_sum / sim_count if sim_count else 0.0
    print(
        f"\n[summary] total={total} empty={empty} repetitive={repetitive} "
        f"avg_char_similarity={avg_sim:.4f}"
    )


if __name__ == "__main__":
    main()
