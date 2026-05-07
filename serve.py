#!/usr/bin/env python3
"""OpenAI-compatible inference server for Qwen2.5-Omni (+ optional LoRA).

Implements a subset of the OpenAI Chat Completions API:
  POST /v1/chat/completions   – standard text/audio chat
  GET  /v1/models             – list loaded model

Usage:
    python serve.py \
        --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
        --lora-dir lora_output \
        --host 0.0.0.0 \
        --port 8000

Client example:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen2.5-omni",
        "messages": [{"role": "user", "content": "帮我打开主驾车窗"}],
        "max_tokens": 128,
        "temperature": 0
      }'

Audio input (base64):
    {
      "messages": [{
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}]
      }]
    }
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, Field
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from tool_postprocess import postprocess_action_call


_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_SP_FILE = _PROJECT_DIR / "data" / "system-prompt.txt"
_SAVE_DIR = _PROJECT_DIR / "data" / "serve_logs"


class _PerfAverages:
    """Track running average latency per named stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._totals: dict[str, float] = {}

    def record(self, stage_ms: dict[str, float]) -> tuple[int, dict[str, float]]:
        with self._lock:
            self._count += 1
            for name, value in stage_ms.items():
                self._totals[name] = self._totals.get(name, 0.0) + value
            averages = {name: self._totals[name] / self._count for name in stage_ms}
            return self._count, averages


def _log_perf_average(name: str, count: int, averages: dict[str, float], ordered_keys: list[str]) -> None:
    fields = " ".join(f"{key}={averages[key]:.1f}ms" for key in ordered_keys if key in averages)
    logger.info("[PERF_AVG] %s n=%d %s", name, count, fields)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class _KvPromptCacheHit:
    suffix_text: str
    prefix_tokens: int
    past_key_values: Any


def _clone_past_key_values(value: Any) -> Any:
    """Clone past_key_values into raw tensor lists stored independently of any cache object.

    We intentionally avoid storing a DynamicCache instance because model.generate()
    mutates it in-place (cache.update(), _seen_tokens, etc.). Instead we extract
    the key/value tensors and store them as plain Python lists of cloned tensors.
    A fresh DynamicCache is reconstructed from these tensors on every request via
    _build_dynamic_cache(), guaranteeing zero cross-request contamination.
    """
    # DynamicCache-style: extract raw tensor lists
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return {
            "__dynamic_cache__": True,
            "key_cache": [t.detach().clone() for t in value.key_cache],
            "value_cache": [t.detach().clone() for t in value.value_cache],
        }
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_past_key_values(item) for item in value)
    if isinstance(value, list):
        return [_clone_past_key_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_past_key_values(item) for key, item in value.items()}
    if hasattr(value, "to_legacy_cache"):
        return _clone_past_key_values(value.to_legacy_cache())
    return value


def _build_dynamic_cache(stored: Any) -> Any:
    """Reconstruct a fresh DynamicCache from the stored raw tensor dict.

    Each call produces a brand-new cache object populated with cloned tensors,
    so generate() can mutate it freely without touching the stored prefix data.
    """
    if not (isinstance(stored, dict) and stored.get("__dynamic_cache__")):
        # Legacy tuple cache or plain tensors: clone recursively
        return _clone_past_key_values(stored)
    try:
        from transformers import DynamicCache
        cache = DynamicCache()
        cache.key_cache = [t.detach().clone() for t in stored["key_cache"]]
        cache.value_cache = [t.detach().clone() for t in stored["value_cache"]]
        cache._seen_tokens = sum(t.shape[-2] for t in cache.key_cache[:1]) if cache.key_cache else 0
        return cache
    except Exception as exc:
        logger.warning("[PROMPT_CACHE] _build_dynamic_cache failed (%s); falling back to tensor clone", exc)
        return (
            tuple(t.detach().clone() for t in stored["key_cache"]),
            tuple(t.detach().clone() for t in stored["value_cache"]),
        )


class _KvPromptCache:
    """Reusable KV cache for the fixed system prompt prefix."""

    def __init__(self) -> None:
        self.system_prompt = ""
        self.prefix_text = ""
        self.prefix_tokens = 0
        self.past_key_values: Any = None
        self.last_miss_reason = ""

    def prepare(self, model, processor, system_prompt: str) -> bool:
        start = time.perf_counter()
        prefix_messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        prefix_text = processor.apply_chat_template(prefix_messages, add_generation_prompt=False, tokenize=False)
        prefix_inputs = processor(
            text=prefix_text,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        prefix_inputs = _inputs_to_model_device(prefix_inputs, model)
        try:
            with torch.inference_mode():
                # Use the full model's forward pass (not the thinker sub-module directly)
                # so that the returned past_key_values are in exactly the same format
                # that model.generate() expects when we inject them later.
                thinker = _get_thinker_model(model)
                if thinker is None:
                    logger.warning("[PROMPT_CACHE] mode=kv disabled reason=thinker_not_found")
                    return False
                # Filter inputs to only keys thinker accepts
                thinker_keys = {"input_ids", "attention_mask", "position_ids",
                                "inputs_embeds", "past_key_values", "use_cache",
                                "output_attentions", "output_hidden_states", "return_dict"}
                thinker_inputs = {k: v for k, v in prefix_inputs.items() if k in thinker_keys}
                output = thinker(**thinker_inputs, use_cache=True, return_dict=True)
        except Exception as exc:
            logger.warning("[PROMPT_CACHE] mode=kv disabled reason=%s: %s", type(exc).__name__, exc)
            return False

        pkv = output.past_key_values
        logger.info(
            "[PROMPT_CACHE] mode=kv pkv_type=%s",
            type(pkv).__name__,
        )
        self.system_prompt = system_prompt
        self.prefix_text = prefix_text
        self.prefix_tokens = int(prefix_inputs["input_ids"].shape[-1])
        self.past_key_values = _clone_past_key_values(pkv)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[PROMPT_CACHE] mode=kv prepared prefix_tokens=%d elapsed=%.1fms",
            self.prefix_tokens,
            elapsed_ms,
        )
        return True

    def match(self, full_text: str, system_prompt: str) -> Optional[_KvPromptCacheHit]:
        self.last_miss_reason = ""
        if not self.past_key_values:
            self.last_miss_reason = "cache_empty"
            return None
        if system_prompt != self.system_prompt:
            self.last_miss_reason = (
                "system_prompt_mismatch "
                f"cached_chars={len(self.system_prompt)} request_chars={len(system_prompt)} "
                f"cached_hash={_short_hash(self.system_prompt)} request_hash={_short_hash(system_prompt)}"
            )
            return None
        if not full_text.startswith(self.prefix_text):
            full_prefix = full_text[:len(self.prefix_text)]
            self.last_miss_reason = (
                "prefix_text_mismatch "
                f"cached_prefix_chars={len(self.prefix_text)} full_chars={len(full_text)} "
                f"cached_prefix_hash={_short_hash(self.prefix_text)} "
                f"full_prefix_hash={_short_hash(full_prefix)}"
            )
            return None
        return _KvPromptCacheHit(
            suffix_text=full_text[len(self.prefix_text):],
            prefix_tokens=self.prefix_tokens,
            past_key_values=_build_dynamic_cache(self.past_key_values),
        )


def _inputs_to_model_device(inputs, model):
    inputs = inputs.to(model.device)
    if getattr(model, "dtype", None) is not None:
        inputs = inputs.to(model.dtype)
    return inputs


def _get_thinker_model(model):
    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        thinker = getattr(candidate, "thinker", None)
        if thinker is not None:
            return thinker
    return None


# ── Pydantic models (OpenAI schema subset) ───────────────────

class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str          # "text" | "input_audio"
    text: Optional[str] = None
    input_audio: Optional[Dict[str, Any]] = None  # {"data": "<b64>", "format": "wav", "sample_rate": int, "channels": int}


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any  # str | list[ContentPart]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = "qwen2.5-omni"
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False

    def resolved_max_tokens(self) -> int:
        """Return max_tokens, falling back to max_completion_tokens, then default 256."""
        return self.max_tokens or self.max_completion_tokens or 256


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    index: int = 0
    function: ToolFunction


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    reasoning_content: str = ""
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


# ── Model loading ─────────────────────────────────────────────

def load_model(model_dir: str, lora_dir: str, torch_dtype: str = "auto"):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"[model] attention implementation: {attn_impl}", flush=True)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    if lora_dir:
        if hasattr(model, "_no_split_modules"):
            raw = model._no_split_modules
            if isinstance(raw, (set, frozenset)):
                flat: list = []
                for item in raw:
                    if isinstance(item, (set, frozenset, list, tuple)):
                        flat.extend(item)
                    else:
                        flat.append(item)
                model._no_split_modules = flat
        model = PeftModel.from_pretrained(model, lora_dir)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    return model, processor


# ── Message conversion ────────────────────────────────────────

def _safe_b64decode(data: str) -> bytes:
    """Decode base64 (standard or URL-safe), tolerating whitespace, missing padding,
    and the 'mod-4 == 1' corruption case (drop one char to make it decodable).
    """
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    # Normalize URL-safe chars
    data = data.replace('-', '+').replace('_', '/')
    # Strip all non-base64 chars
    data = re.sub(r'[^A-Za-z0-9+/]', '', data)
    r = len(data) % 4
    if r == 1:
        # 1 mod 4 is always invalid (6 bits < 1 byte).
        # One extra char snuck in; drop it so we get 0 mod 4.
        data = data[:-1]
    elif r:
        # 2 or 3 mod 4: just add padding
        data += '=' * (4 - r)
    return base64.b64decode(data)


def _detect_audio_fmt(data: bytes, declared_fmt: str) -> tuple[str, str]:
    """Return (ffmpeg_input_flag, file_ext) based on magic bytes."""
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return ("wav", "wav")
    if len(data) >= 3 and (data[:3] == b'ID3' or data[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2')):
        return ("mp3", "mp3")
    if len(data) >= 4 and data[:4] == b'OggS':
        return ("ogg", "ogg")
    if len(data) >= 8 and data[4:8] == b'ftyp':
        return ("mp4", "m4a")
    if len(data) >= 4 and data[:4] == b'\x1aE\xdf\xa3':
        return ("webm", "webm")
    # Raw PCM hint from declared format or unknown bytes claiming to be wav
    dl = declared_fmt.lower()
    if any(k in dl for k in ("pcm", "raw", "s16", "f32")):
        return ("s16le", "pcm")
    # Declared wav but no RIFF header → likely raw PCM
    if dl in ("wav", "wave"):
        return ("s16le", "pcm")
    # Unknown: let ffmpeg probe, use .bin to avoid extension-based misdetection
    return ("", "bin")


def _is_wav_16k_mono_pcm16(data: bytes) -> bool:
    if not (len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE'):
        return False
    try:
        import io
        with wave.open(io.BytesIO(data), "rb") as wav:
            return (
                wav.getframerate() == 16000
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
            )
    except (EOFError, wave.Error):
        return False


def _run_ffmpeg(src_path: str, wav_path: str, fmt_flag: str = "", extra_input_args: list = None) -> bool:
    """Run ffmpeg to convert src_path -> wav_path (16kHz mono WAV). Returns True on success."""
    import subprocess
    cmd = ["ffmpeg", "-y"]
    if extra_input_args:
        cmd += extra_input_args
    if fmt_flag:
        cmd += ["-f", fmt_flag]
    cmd += ["-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── Debug ASR (lazy-loaded, CPU-only) ────────────────────────
_asr_debug_model: "Any" = None
_asr_debug_available: bool | None = None  # None = not tried yet


def _asr_transcribe_debug(wav_path: str) -> str | None:
    """Transcribe wav_path with a tiny Whisper model for debugging. Returns text or None."""
    import sys as _sys
    global _asr_debug_model, _asr_debug_available
    if _asr_debug_available is False:
        return None
    try:
        if _asr_debug_model is None:
            try:
                from faster_whisper import WhisperModel
                _asr_debug_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                _asr_debug_available = True
                print("[ASR] loaded faster-whisper tiny (cpu)", file=_sys.stderr, flush=True)
            except ImportError:
                try:
                    import whisper as _oai_whisper
                    _asr_debug_model = _oai_whisper.load_model("tiny", device="cpu")
                    _asr_debug_available = True
                    print("[ASR] loaded openai-whisper tiny (cpu)", file=_sys.stderr, flush=True)
                except ImportError:
                    _asr_debug_available = False
                    print("[ASR] disabled: neither faster-whisper nor openai-whisper installed. "
                          "Run: pip install faster-whisper", file=_sys.stderr, flush=True)
                    return None

        model = _asr_debug_model
        # Detect API flavour
        try:
            from faster_whisper import WhisperModel
            if isinstance(model, WhisperModel):
                segs, _ = model.transcribe(wav_path, language="zh", beam_size=1)
                return "".join(s.text for s in segs)
        except ImportError:
            pass
        # openai-whisper API
        result = model.transcribe(wav_path, language="zh")
        return result["text"]
    except Exception as _e:
        _asr_debug_available = False
        print(f"[ASR] disabled after error: {_e}", file=_sys.stderr, flush=True)
        return None


def _write_audio_tmp(raw_bytes: bytes, fmt: str, tmp_dir: str,
                      sample_rate: int | None = None, channels: int | None = None) -> str:
    """Write audio bytes to a temp WAV (16 kHz mono) that librosa can read."""
    import sys

    start = time.perf_counter()
    # Log first bytes for debugging unknown formats
    hex_head = raw_bytes[:16].hex() if raw_bytes else ""
    print(f"[AUDIO] declared_fmt={fmt!r} size={len(raw_bytes)} head={hex_head}", flush=True, file=sys.stderr)

    # Save raw bytes with declared extension for manual inspection
    raw_ext = fmt.lower().lstrip('.') or "bin"
    debug_raw = f"/tmp/qwen_audio_debug_latest.{raw_ext}"
    try:
        with open(debug_raw, "wb") as _df:
            _df.write(raw_bytes)
        print(f"[AUDIO] raw copy saved → {debug_raw}", flush=True, file=sys.stderr)
    except OSError as _e:
        print(f"[AUDIO] could not save raw copy: {_e}", flush=True, file=sys.stderr)

    ffmpeg_fmt, src_ext = _detect_audio_fmt(raw_bytes, fmt)
    src_path = os.path.join(tmp_dir, f"audio_in_{uuid.uuid4().hex}.{src_ext}")
    wav_path = os.path.join(tmp_dir, f"audio_{uuid.uuid4().hex}.wav")

    with open(src_path, "wb") as f:
        f.write(raw_bytes)

    try:
        if ffmpeg_fmt == "wav" and _is_wav_16k_mono_pcm16(raw_bytes):
            elapsed_ms = (time.perf_counter() - start) * 1000
            debug_wav = "/tmp/qwen_audio_converted_latest.wav"
            try:
                shutil.copy2(src_path, debug_wav)
                print(f"[AUDIO] converted WAV saved → {debug_wav}", flush=True, file=sys.stderr)
            except OSError as _e:
                print(f"[AUDIO] could not save converted WAV: {_e}", flush=True, file=sys.stderr)
            print(
                f"[AUDIO] fast path: using existing 16kHz mono PCM16 WAV ({elapsed_ms:.1f} ms)",
                flush=True,
                file=sys.stderr,
            )
            return src_path

        ok = False

        if ffmpeg_fmt == "s16le":
            # If client declared sample_rate and channels, use them directly
            if sample_rate and channels:
                print(f"[AUDIO] PCM direct decode sr={sample_rate} ch={channels}", flush=True, file=sys.stderr)
                ok = _run_ffmpeg(src_path, wav_path,
                                 extra_input_args=["-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels)])
                if ok and _debug_asr_enabled:
                    transcript = _asr_transcribe_debug(wav_path)
                    if transcript is not None:
                        print(f"[ASR] transcript: {transcript!r}", file=sys.stderr, flush=True)

            if not ok:
                # Raw PCM: try multiple configurations, pick best via ASR
                pcm_configs = [
                    ("s16le", "16000", "1"),
                    ("s16le", "48000", "1"),
                    ("s16le", "44100", "1"),
                    ("s16le", "48000", "2"),
                    ("s16le", "44100", "2"),
                    ("f32le", "16000", "1"),
                    ("f32le", "48000", "1"),
                ]
                best_wav = None
                best_transcript = ""
                for pcm_fmt, sr, ch in pcm_configs:
                    trial_wav = os.path.join(tmp_dir, f"audio_trial_{uuid.uuid4().hex}.wav")
                    trial_ok = _run_ffmpeg(src_path, trial_wav,
                                           extra_input_args=["-f", pcm_fmt, "-ar", sr, "-ac", ch])
                    if not trial_ok:
                        try:
                            os.remove(trial_wav)
                        except OSError:
                            pass
                        continue
                    transcript = _asr_transcribe_debug(trial_wav) if _debug_asr_enabled else ""
                    print(f"[AUDIO] trial {pcm_fmt}/{sr}Hz/{ch}ch → ASR: {transcript!r}",
                          flush=True, file=sys.stderr)
                    if transcript and len(transcript) > len(best_transcript):
                        # Prefer the config that yields longer (more meaningful) transcript
                        if best_wav:
                            try:
                                os.remove(best_wav)
                            except OSError:
                                pass
                        best_wav = trial_wav
                        best_transcript = transcript
                    else:
                        if not best_wav:
                            best_wav = trial_wav  # keep first success as fallback
                        else:
                            try:
                                os.remove(trial_wav)
                            except OSError:
                                pass
                if best_wav:
                    shutil.move(best_wav, wav_path)
                    ok = True
                    print(f"[AUDIO] best PCM decode → ASR: {best_transcript!r}",
                          flush=True, file=sys.stderr)
        elif ffmpeg_fmt:
            ok = _run_ffmpeg(src_path, wav_path, fmt_flag=ffmpeg_fmt)
        
        if not ok:
            # Try without any format hint (ffmpeg auto-probe)
            ok = _run_ffmpeg(src_path, wav_path)

        if not ok:
            # Last resort: try treating as raw s16le 16kHz
            ok = _run_ffmpeg(src_path, wav_path,
                             extra_input_args=["-f", "s16le", "-ar", "16000", "-ac", "1"])
            if ok:
                print("[AUDIO] fallback: treated as raw s16le 16kHz", flush=True, file=sys.stderr)

        if ok:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Save converted WAV for listening
            debug_wav = "/tmp/qwen_audio_converted_latest.wav"
            try:
                shutil.copy2(wav_path, debug_wav)
                print(f"[AUDIO] converted WAV saved → {debug_wav}", flush=True, file=sys.stderr)
            except OSError as _e:
                print(f"[AUDIO] could not save converted WAV: {_e}", flush=True, file=sys.stderr)
            # Final ASR check (skip if already done in PCM trial)
            if _debug_asr_enabled and ffmpeg_fmt != "s16le":
                transcript = _asr_transcribe_debug(wav_path)
                if transcript is not None:
                    print(f"[ASR] transcript: {transcript!r}", file=sys.stderr, flush=True)
            try:
                os.remove(src_path)
            except OSError:
                pass
            print(f"[AUDIO] prepare elapsed={elapsed_ms:.1f} ms", flush=True, file=sys.stderr)
            return wav_path
        else:
            print(f"[WARN] all ffmpeg attempts failed for fmt={fmt!r}", flush=True, file=sys.stderr)

    except FileNotFoundError:
        print("[WARN] ffmpeg not found", flush=True, file=sys.stderr)

    try:
        os.remove(wav_path)
    except OSError:
        pass
    return src_path


def _messages_to_qwen(
    messages: List[Message],
    system_prompt: str,
    tmp_dir: str,
) -> tuple[list, list]:
    """Convert OpenAI-format messages to Qwen format.
    Returns (qwen_messages, temp_files_to_cleanup).
    """
    qwen_msgs = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    tmp_files: list[str] = []

    for msg in messages:
        role = msg.role
        if role == "system":
            # Server-side system prompt is authoritative; ignore client overrides
            # so the pre-warmed KV prompt cache remains valid.
            continue

        # Convert content to Qwen format
        if isinstance(msg.content, str):
            qwen_content = [{"type": "text", "text": msg.content}]
        else:
            qwen_content = []
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype == "text":
                        qwen_content.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "input_audio":
                        audio_info = part.get("input_audio", {})
                        b64data = audio_info.get("data", "")
                        fmt = audio_info.get("format", "wav")
                        if b64data:
                            raw = _safe_b64decode(b64data)
                            sr = int(audio_info["sample_rate"]) if audio_info.get("sample_rate") else None
                            ch = int(audio_info.get("channels") or audio_info.get("channel") or 0) or None
                            tmp_path = _write_audio_tmp(raw, fmt, tmp_dir, sample_rate=sr, channels=ch)
                            tmp_files.append(tmp_path)
                            qwen_content.append({"type": "audio", "audio": tmp_path})
                else:
                    # Pydantic ContentPart
                    if part.type == "text" and part.text:
                        qwen_content.append({"type": "text", "text": part.text})
                    elif part.type == "input_audio" and part.input_audio:
                        b64data = part.input_audio.get("data", "")
                        fmt = part.input_audio.get("format", "wav")
                        if b64data:
                            raw = _safe_b64decode(b64data)
                            sr = int(part.input_audio["sample_rate"]) if part.input_audio.get("sample_rate") else None
                            ch = int(part.input_audio.get("channels") or part.input_audio.get("channel") or 0) or None
                            tmp_path = _write_audio_tmp(raw, fmt, tmp_dir, sample_rate=sr, channels=ch)
                            tmp_files.append(tmp_path)
                            qwen_content.append({"type": "audio", "audio": tmp_path})

        qwen_msgs.append({"role": role, "content": qwen_content})

    return qwen_msgs, tmp_files


# ── Inference ─────────────────────────────────────────────────

def run_inference(
    model,
    processor,
    qwen_messages: list,
    max_new_tokens: int,
    temperature: float,
    prompt_cache: Optional[_KvPromptCache] = None,
) -> tuple[str, int, int]:
    start = time.perf_counter()
    text = processor.apply_chat_template(qwen_messages, add_generation_prompt=True, tokenize=False)
    chat_template_ms = (time.perf_counter() - start) * 1000
    mm_start = time.perf_counter()
    audios, images, videos = process_mm_info(qwen_messages, use_audio_in_video=False)
    mm_ms = (time.perf_counter() - mm_start) * 1000
    # Sanity-check: count audio placeholders in the template vs what process_mm_info returned.
    # If they don't match, process_mm_info silently dropped audio data → model will misalign.
    audio_placeholder_count = text.count("<|AUDIO|>")
    logger.info(
        "[MM] audio_placeholders=%d process_mm_info_returned=%d mm_ms=%.1f",
        audio_placeholder_count,
        len(audios or []),
        mm_ms,
    )
    if audio_placeholder_count != len(audios or []):
        logger.warning(
            "[MM] audio mismatch: template has %d <|AUDIO|> placeholders but process_mm_info returned %d arrays; "
            "disabling KV cache for this request to avoid misalignment",
            audio_placeholder_count,
            len(audios or []),
        )
    system_prompt = ""
    if qwen_messages and qwen_messages[0].get("role") == "system":
        content = qwen_messages[0].get("content") or []
        if content and isinstance(content[0], dict):
            system_prompt = content[0].get("text", "")
    # SP tokens are a contiguous prefix; audio tokens land in the user turn after SP.
    # The cache covers only SP tokens so position alignment is preserved for all requests.
    # Only skip cache if process_mm_info dropped audio data (placeholder/array count mismatch).
    audio_ok = (audio_placeholder_count == len(audios or []))
    cache_hit = prompt_cache.match(text, system_prompt) if (prompt_cache and audio_ok) else None

    processor_start = time.perf_counter()
    processor_text = cache_hit.suffix_text if cache_hit else text
    inputs = processor(
        text=processor_text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    processor_ms = (time.perf_counter() - processor_start) * 1000
    move_start = time.perf_counter()
    inputs = _inputs_to_model_device(inputs, model)
    move_ms = (time.perf_counter() - move_start) * 1000
    suffix_prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    prompt_len = suffix_prompt_len
    if cache_hit:
        prompt_len += cache_hit.prefix_tokens
        if "attention_mask" in inputs:
            prefix_mask = torch.ones(
                (inputs["attention_mask"].shape[0], cache_hit.prefix_tokens),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            inputs["attention_mask"] = torch.cat([prefix_mask, inputs["attention_mask"]], dim=-1)
        inputs["past_key_values"] = cache_hit.past_key_values

    generate_start = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_audio=False,
        )
    generate_ms = (time.perf_counter() - generate_start) * 1000

    decode_start = time.perf_counter()
    decode_prompt_len = suffix_prompt_len if cache_hit else prompt_len
    gen_ids = out_ids[:, decode_prompt_len:]
    decoded = processor.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decode_ms = (time.perf_counter() - decode_start) * 1000
    total_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[PERF] inference chat_template=%.1fms mm=%.1fms processor=%.1fms "
        "to_device=%.1fms generate=%.1fms decode=%.1fms total=%.1fms "
        "prompt_cache=%s cache_prefix_tokens=%d cache_miss_reason=%s",
        chat_template_ms,
        mm_ms,
        processor_ms,
        move_ms,
        generate_ms,
        decode_ms,
        total_ms,
        "hit" if cache_hit else ("miss" if prompt_cache else "off"),
        cache_hit.prefix_tokens if cache_hit else 0,
        "-" if cache_hit or not prompt_cache else (prompt_cache.last_miss_reason or "unknown"),
    )
    inference_keys = ["chat_template", "mm", "processor", "to_device", "generate", "decode", "total"]
    count, averages = _inference_perf_averages.record(
        {
            "chat_template": chat_template_ms,
            "mm": mm_ms,
            "processor": processor_ms,
            "to_device": move_ms,
            "generate": generate_ms,
            "decode": decode_ms,
            "total": total_ms,
        }
    )
    _log_perf_average("inference", count, averages, inference_keys)
    return decoded.strip(), int(prompt_len), int(gen_ids.shape[-1])


# ── Output parsing ───────────────────────────────────────────

_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*\})",
    re.DOTALL,
)


def parse_model_output(text: str, query: str = "") -> tuple:
    """Parse model text output into structured form.

    Returns:
        ("tool_call", tool_name: str, args_json: str)
      | ("text", content: str)
    """
    m = _ACTION_RE.search(text)
    if m:
        tool_name = m.group(1).strip()
        args_str = m.group(2).strip()
        try:
            args = json.loads(args_str)
            tool_name, args = postprocess_action_call(query, tool_name, args)
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # still return as-is; caller receives raw string
        return ("tool_call", tool_name, args_str)

    # Strip "Final Answer: " prefix if present
    if text.startswith("Final Answer:"):
        content = text[len("Final Answer:"):].strip()
    else:
        content = text
    return ("text", content)


# ── Request/Response persistence ──────────────────────────────

def _save_request_artifacts(
    request_id: str,
    tmp_files: list[str],
    response: "ChatResponse",
    messages: List[Message],
) -> None:
    """Save audio files and response for each request, keyed by request_id."""
    try:
        req_dir = _SAVE_DIR / request_id
        req_dir.mkdir(parents=True, exist_ok=True)

        # Save audio wav files
        for i, fpath in enumerate(tmp_files):
            if os.path.exists(fpath):
                suffix = Path(fpath).suffix or ".wav"
                dst = req_dir / f"audio_{i}{suffix}"
                shutil.copy2(fpath, dst)

        # Save the user text query (if any)
        text_query = _last_text_query(messages)
        if text_query:
            (req_dir / "query.txt").write_text(text_query, encoding="utf-8")

        # Save the full response JSON
        (req_dir / "response.json").write_text(
            response.model_dump_json(indent=2), encoding="utf-8"
        )

        logger.info(f"[SAVE] artifacts saved → {req_dir}")
    except Exception as e:
        logger.warning(f"[SAVE] failed to save artifacts for {request_id}: {e}")


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(title="Qwen2.5-Omni Inference Server", version="1.0.0")


def _last_text_query(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            return msg.content
        for part in reversed(msg.content if isinstance(msg.content, list) else []):
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
            if not isinstance(part, dict) and getattr(part, "type", "") == "text":
                return getattr(part, "text", "") or ""
    return ""


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every POST body before validation, so 422 source is always visible."""
    import sys
    if request.method == "POST":
        try:
            raw = await request.body()
            body_str = raw.decode("utf-8", errors="replace")
            truncated = body_str[:1000] + f"...(total {len(raw)} bytes)" if len(body_str) > 1000 else body_str
            print(f"\n[REQUEST] POST {request.url.path}", flush=True, file=sys.stderr)
            print(f"[REQUEST] Content-Type: {request.headers.get('content-type', 'N/A')}", flush=True, file=sys.stderr)
            print(f"[REQUEST] Body: {truncated}\n", flush=True, file=sys.stderr)
            # Save full raw request to /tmp for inspection
            try:
                with open("/tmp/qwen_last_request.json", "wb") as _rf:
                    _rf.write(raw)
                print(f"[REQUEST] Full body saved → /tmp/qwen_last_request.json ({len(raw)} bytes)", flush=True, file=sys.stderr)
            except OSError:
                pass
        except Exception as e:
            print(f"[REQUEST] Could not read body: {e}", flush=True, file=sys.stderr)
    response = await call_next(request)
    if response.status_code == 422:
        print(f"[RESPONSE] 422 Unprocessable Entity for {request.url.path}", flush=True, file=sys.stderr)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import sys
    print(f"[422 DETAIL] Validation errors: {exc.errors()}", flush=True, file=sys.stderr)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Globals set at startup
_model = None
_processor = None
_system_prompt = ""
_model_name = "qwen2.5-omni"
_tmp_dir = tempfile.mkdtemp(prefix="qwen_serve_")
_debug_asr_enabled = False
_inference_perf_averages = _PerfAverages()
_request_perf_averages = _PerfAverages()
_prompt_cache_mode = "none"
_kv_prompt_cache: Optional[_KvPromptCache] = None


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    import sys, traceback
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    request_start = time.perf_counter()
    try:
        convert_start = time.perf_counter()
        qwen_msgs, tmp_files = _messages_to_qwen(req.messages, _system_prompt, _tmp_dir)
        convert_ms = (time.perf_counter() - convert_start) * 1000
    except Exception as e:
        print(f"\n[ERROR] _messages_to_qwen failed: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=422, detail=f"Message conversion error: {e}") from e

    try:
        inference_start = time.perf_counter()
        reply, prompt_tokens, gen_tokens = run_inference(
            _model, _processor, qwen_msgs,
            max_new_tokens=req.resolved_max_tokens(),
            temperature=req.temperature,
            prompt_cache=_kv_prompt_cache if _prompt_cache_mode == "kv" else None,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000
    except Exception as e:
        print(f"\n[ERROR] run_inference failed: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    print(f"[MODEL_RAW] {repr(reply)}", flush=True, file=sys.stderr)
    parse_start = time.perf_counter()
    parsed = parse_model_output(reply, _last_text_query(req.messages))
    parse_ms = (time.perf_counter() - parse_start) * 1000
    if parsed[0] == "tool_call":
        _, tool_name, args_str = parsed
        choice = Choice(
            message=AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:22]}",
                        function=ToolFunction(name=tool_name, arguments=args_str),
                    )
                ]
            ),
            finish_reason="tool_calls",
        )
    else:
        _, content = parsed
        choice = Choice(
            message=AssistantMessage(content=content),
            finish_reason="stop",
        )

    resp = ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model or _model_name,
        choices=[choice],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
        ),
    )
    print(f"[RESPONSE] {resp.model_dump_json()}", file=sys.stderr, flush=True)

    # Persist audio + response for training data collection
    save_start = time.perf_counter()
    _save_request_artifacts(resp.id, tmp_files, resp, req.messages)
    save_ms = (time.perf_counter() - save_start) * 1000

    # Cleanup temp audio files
    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    total_ms = (time.perf_counter() - request_start) * 1000
    logger.info(
        "[PERF] request convert=%.1fms inference=%.1fms parse=%.1fms save=%.1fms total=%.1fms",
        convert_ms,
        inference_ms,
        parse_ms,
        save_ms,
        total_ms,
    )
    request_keys = ["convert", "inference", "parse", "save", "total"]
    count, averages = _request_perf_averages.record(
        {
            "convert": convert_ms,
            "inference": inference_ms,
            "parse": parse_ms,
            "save": save_ms,
            "total": total_ms,
        }
    )
    _log_perf_average("request", count, averages, request_keys)

    return resp


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── CLI entrypoint ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible inference server for Qwen2.5-Omni")
    p.add_argument("--model-dir", required=True, help="Base model directory")
    p.add_argument("--lora-dir", default="", help="Optional LoRA adapter directory")
    p.add_argument("--system-prompt-file", default=str(_DEFAULT_SP_FILE))
    p.add_argument("--system-prompt", default="", help="Inline system prompt (overrides file)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="qwen2.5-omni", help="Model name returned in API responses")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--debug-asr", action="store_true", help="Run local Whisper ASR for audio debugging.")
    p.add_argument(
        "--prompt-cache",
        default="none",
        choices=["none", "kv"],
        help="Prompt cache mode. 'kv' is experimental and caches the fixed system-prompt prefix.",
    )
    return p.parse_args()


def main():
    global _model, _processor, _system_prompt, _model_name, _debug_asr_enabled, _prompt_cache_mode, _kv_prompt_cache

    args = parse_args()

    # Resolve system prompt
    if args.system_prompt:
        _system_prompt = args.system_prompt
    else:
        sp_path = Path(args.system_prompt_file)
        if sp_path.exists():
            _system_prompt = sp_path.read_text(encoding="utf-8").strip()
            print(f"[sp] loaded {sp_path} ({len(_system_prompt)} chars)")
        else:
            _system_prompt = "你是车载语音助手。"
            print(f"[sp] using fallback ({len(_system_prompt)} chars)")

    _model_name = args.model_name
    _debug_asr_enabled = args.debug_asr
    _prompt_cache_mode = args.prompt_cache
    print(f"[asr] debug_asr={_debug_asr_enabled}")
    print(f"[prompt_cache] mode={_prompt_cache_mode}")

    print(f"[model] loading {args.model_dir} ...")
    _model, _processor = load_model(args.model_dir, args.lora_dir, args.torch_dtype)
    if _prompt_cache_mode == "kv":
        _kv_prompt_cache = _KvPromptCache()
        if not _kv_prompt_cache.prepare(_model, _processor, _system_prompt):
            _kv_prompt_cache = None
            _prompt_cache_mode = "none"
            print("[prompt_cache] mode=kv disabled; falling back to none")
    print(f"[model] ready  lora={args.lora_dir or 'none'}")
    print(f"[server] starting on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
