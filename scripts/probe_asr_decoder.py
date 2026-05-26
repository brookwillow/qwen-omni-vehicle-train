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
import os
from pathlib import Path
from typing import Any

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_outputs import BaseModelOutput


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
    parser.add_argument("--audio", required=True, help="Audio file path to transcribe.")
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
    whisper_dir: str,
    dtype: torch.dtype,
    device_map: str,
    language: str,
    max_new_tokens: int,
) -> str:
    whisper_source = resolve_whisper_dir(whisper_dir)
    print(f"[4/4] loading Whisper from {whisper_source}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_source,
        torch_dtype=dtype,
        device_map=device_map,
    )
    whisper_model.eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_source)

    encoder_dim = encoder_hidden.shape[-1]
    whisper_dim = whisper_model.config.d_model
    print(f"[probe] encoder_hidden dim={encoder_dim}, whisper d_model={whisper_dim}")
    if encoder_dim != whisper_dim:
        print("[warn] hidden dim mismatch; Whisper decoder output is likely invalid.")

    encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_hidden.to(first_param_device(whisper_model), dtype=dtype)
    )
    forced_ids = whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.inference_mode():
        generated = whisper_model.generate(
            encoder_outputs=encoder_outputs,
            forced_decoder_ids=forced_ids,
            max_new_tokens=max_new_tokens,
        )
    return whisper_processor.batch_decode(generated, skip_special_tokens=True)[0]


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    audio_path = str(Path(args.audio).expanduser().resolve())

    print(f"[1/4] loading Qwen2.5-Omni from {args.model_dir}")
    qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    qwen_model.eval()
    qwen_processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir)

    if args.print_structure:
        print_structure(qwen_model)
        return

    audio_tower = find_audio_tower(qwen_model)
    hook_target = select_hook_target(audio_tower, args.hook_layer)

    print(f"[2/4] processing audio: {audio_path}")
    qwen_device = first_param_device(qwen_model)
    inputs = build_qwen_inputs(qwen_processor, audio_path, qwen_device, dtype)
    if "input_features" in inputs:
        print(f"[probe] input_features shape={tuple(inputs['input_features'].shape)}")
    if "feature_attention_mask" in inputs:
        print(f"[probe] feature_attention_mask shape={tuple(inputs['feature_attention_mask'].shape)}")

    print("[3/4] running Qwen thinker and capturing audio hidden states")
    encoder_hidden = capture_audio_hidden(qwen_model, hook_target, inputs)
    stats = encoder_hidden.float()
    print(
        "[probe] hidden "
        f"shape={tuple(encoder_hidden.shape)} "
        f"dtype={encoder_hidden.dtype} "
        f"mean={stats.mean().item():.6f} "
        f"std={stats.std().item():.6f}"
    )

    text = decode_with_whisper(
        encoder_hidden=encoder_hidden,
        whisper_dir=args.whisper_dir,
        dtype=dtype,
        device_map=args.device_map,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"\n[ASR result] {text}")


if __name__ == "__main__":
    main()
