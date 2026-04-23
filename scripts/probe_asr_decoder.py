#!/usr/bin/env python3
"""Probe script: extract Qwen2.5-Omni audio encoder features and decode ASR
using a Whisper-Large decoder, bypassing the LLM path.

Usage:
    python scripts/probe_asr_decoder.py \
        --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
        --whisper-dir openai/whisper-large-v3 \
        --audio data/eval/audio/window/window_001.wav

This validates whether the Qwen audio encoder's hidden states are compatible
with a Whisper decoder for ASR transcription.

Architecture rationale:
  Qwen2.5-Omni audio encoder config matches Whisper-Large-v3:
    d_model=1280, encoder_layers=32, num_mel_bins=128
  The encoder produces raw 1280-dim hidden states before a linear projection
  to 2048-dim for injection into the LLM. We hook the pre-projection output
  and feed it to the Whisper decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from qwen_omni_utils import process_mm_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Qwen audio encoder → Whisper decoder ASR")
    p.add_argument("--model-dir", required=True, help="Qwen2.5-Omni model directory")
    p.add_argument("--whisper-dir", default="openai/whisper-large-v3",
                   help="Whisper model name or local path (default: openai/whisper-large-v3)")
    p.add_argument("--audio", required=True, help="Audio file path to transcribe")
    p.add_argument("--print-structure", action="store_true",
                   help="Print audio tower module names and exit")
    p.add_argument("--language", default="zh", help="ASR language hint for Whisper decoder")
    p.add_argument("--hook-layer", default="ln_post",
                   choices=["ln_post", "avg_pooler", "full"],
                   help="Which layer to hook for encoder features: "
                        "ln_post=pre-projection 1280-dim (recommended), "
                        "avg_pooler=after pooling but before proj, "
                        "full=entire audio_tower output (2048-dim, likely incompatible)")
    return p.parse_args()


def find_audio_tower(model) -> torch.nn.Module:
    """Locate the audio encoder inside Qwen2.5-Omni thinker."""
    # Try common attribute paths
    candidates = [
        "thinker.audio_tower",
        "thinker.model.audio_tower",
        "audio_tower",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            print(f"[probe] Found audio tower at: model.{path}")
            return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not find audio_tower. Run with --print-structure to inspect the model."
    )


def print_structure(model) -> None:
    print("\n=== Model named modules (top 3 levels) ===")
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth <= 2:
            print(f"  {'  ' * depth}{name}: {type(mod).__name__}")


def _resolve_whisper_dir(whisper_dir: str) -> str:
    """Return a local path for the Whisper model.

    If `whisper_dir` is already a local path that exists, return it as-is.
    Otherwise try to resolve via modelscope (avoids slow/blocked HuggingFace downloads).
    Falls back to the original value so transformers can attempt its own download.
    """
    if Path(whisper_dir).exists():
        return whisper_dir

    # Try modelscope cache first (may already be downloaded)
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        # Map HF model IDs to modelscope equivalents
        ms_id_map = {
            "openai/whisper-large-v3": "AI-ModelScope/whisper-large-v3",
            "openai/whisper-large-v2": "AI-ModelScope/whisper-large-v2",
            "openai/whisper-medium": "AI-ModelScope/whisper-medium",
        }
        ms_id = ms_id_map.get(whisper_dir, whisper_dir)
        print(f"[probe] Downloading Whisper via modelscope: {ms_id} ...")
        local_dir = snapshot_download(ms_id)
        print(f"[probe] Whisper cached at: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"[probe] modelscope download failed ({e}), falling back to: {whisper_dir}")
        return whisper_dir


def main() -> None:
    args = parse_args()

    print(f"[1/4] Loading Qwen2.5-Omni from {args.model_dir} ...")
    qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map="auto",
    )
    qwen_model.eval()
    qwen_proc = Qwen2_5OmniProcessor.from_pretrained(args.model_dir)

    if args.print_structure:
        print_structure(qwen_model)
        return

    audio_tower = find_audio_tower(qwen_model)
    print(f"[probe] Audio tower type: {type(audio_tower).__name__}")

    # Hook at the chosen layer to get pre-projection 1280-dim features.
    # audio_tower.ln_post  → before avg_pooler and proj (full sequence, 1280-dim) ← best for Whisper
    # audio_tower.avg_pooler → after pooling, before proj (shorter sequence, 1280-dim)
    # full audio_tower     → after proj (2048-dim, incompatible with Whisper decoder)
    hook_layer = args.hook_layer
    if hook_layer == "ln_post" and hasattr(audio_tower, "ln_post"):
        hook_target = audio_tower.ln_post
        print(f"[probe] Hooking at: audio_tower.ln_post (1280-dim, pre-pooling)")
    elif hook_layer == "avg_pooler" and hasattr(audio_tower, "avg_pooler"):
        hook_target = audio_tower.avg_pooler
        print(f"[probe] Hooking at: audio_tower.avg_pooler (1280-dim, post-pooling)")
    else:
        hook_target = audio_tower
        print(f"[probe] Hooking entire audio_tower (likely 2048-dim, may be incompatible)")

    # ── Step 1: get mel spectrogram via Qwen processor ──────────────────────
    print(f"\n[2/4] Processing audio: {args.audio}")
    audio_path = str(Path(args.audio).resolve())
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an assistant."}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]
    text_tmpl = qwen_proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = qwen_proc(
        text=text_tmpl, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(qwen_model.device).to(torch.float16)

    # The mel-spectrogram is in inputs["input_features"] (if present)
    # or can be extracted from the prepared inputs.
    if "input_features" not in inputs:
        # Fallback: reprocess just the audio with whisper processor to get input_features
        print("[probe] 'input_features' not found in Qwen inputs – will extract via audio_tower directly")

    # ── Step 2: run audio encoder, hook last_hidden_state ───────────────────
    print("\n[3/4] Running Qwen audio encoder ...")
    encoder_hidden = None

    def _hook(module, input, output):
        nonlocal encoder_hidden
        # output may be a tuple or a BaseModelOutput; grab the tensor
        if isinstance(output, (tuple, list)):
            encoder_hidden = output[0].detach().float()
        elif hasattr(output, "last_hidden_state"):
            encoder_hidden = output.last_hidden_state.detach().float()
        else:
            encoder_hidden = output.detach().float()

    handle = hook_target.register_forward_hook(_hook)

    with torch.inference_mode():
        # Always trigger via the full thinker forward pass so that
        # feature_lens (derived from feature_attention_mask internally)
        # is computed and passed to the audio tower correctly.
        _ = qwen_model.thinker(
            **{k: v for k, v in inputs.items() if k != "labels"},
            output_hidden_states=False,
        )

    handle.remove()

    if encoder_hidden is None:
        raise RuntimeError("Hook did not capture any output. Check the audio tower path.")

    # Ensure shape is [B, T, 1280] — hook may return [T, 1280] without batch dim
    if encoder_hidden.ndim == 2:
        encoder_hidden = encoder_hidden.unsqueeze(0)
    print(f"[probe] Encoder hidden state shape: {encoder_hidden.shape}")
    # Expected: [1, T, 1280]  where T ~ audio_len / 2 frames

    # ── Step 3: load Whisper decoder and decode ──────────────────────────────
    whisper_source = _resolve_whisper_dir(args.whisper_dir)
    print(f"\n[4/4] Loading Whisper decoder from {whisper_source} ...")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_source, torch_dtype=torch.float16, device_map="auto",
    )
    whisper_model.eval()
    whisper_proc = WhisperProcessor.from_pretrained(whisper_source)

    # Whisper encoder expects shape [B, T, d_model] where d_model=1280 for large
    encoder_dim = encoder_hidden.shape[-1]
    whisper_d_model = whisper_model.config.d_model
    print(f"[probe] encoder_hidden dim={encoder_dim}, Whisper d_model={whisper_d_model}")

    if encoder_dim != whisper_d_model:
        print(f"[WARN] Dimension mismatch: {encoder_dim} vs {whisper_d_model}. "
              f"Whisper decoder may not work correctly.")

    # Wrap encoder output as BaseModelOutput for Whisper
    from transformers.modeling_outputs import BaseModelOutput
    encoder_out = BaseModelOutput(
        last_hidden_state=encoder_hidden.to(whisper_model.device),
    )

    # Generate ASR tokens
    forced_ids = whisper_proc.get_decoder_prompt_ids(language=args.language, task="transcribe")
    with torch.inference_mode():
        generated = whisper_model.generate(
            encoder_outputs=encoder_out,
            forced_decoder_ids=forced_ids,
            max_new_tokens=256,
        )

    asr_text = whisper_proc.batch_decode(generated, skip_special_tokens=True)
    print(f"\n[ASR result] {asr_text[0]}")


if __name__ == "__main__":
    main()
