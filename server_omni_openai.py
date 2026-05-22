#!/usr/bin/env python3
"""
Qwen2.5-Omni OpenAI-compatible inference server with block-level KV prompt cache.

Combines:
- Block-level LRU KV cache (KVCacheManager) from server_omni.py
- Full OpenAI-compatible API format from serve.py

Usage:
    python server_omni_openai.py \
        --model-dir ./models/Qwen2.5-Omni-3B \
        --host 0.0.0.0 \
        --port 8000 \
        --prompt-cache kv

    # With LoRA:
    python server_omni_openai.py \
        --model-dir ./models/Qwen2.5-Omni-3B \
        --lora-dir ./lora/my_adapter \
        --prompt-cache kv

    # Disable cache:
    python server_omni_openai.py --model-dir ./models/Qwen2.5-Omni-3B --prompt-cache none
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
import sys
import tempfile
import threading
import time
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

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

_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_SP_FILE = _PROJECT_DIR / "data" / "system-prompt.txt"
_SAVE_DIR = _PROJECT_DIR / "data" / "serve_logs"
_DEFAULT_LOG_FILE = Path("/tmp/qwen_omni_serve.log")


# ============================================================
# Block-level KV Cache Manager (from server_omni.py)
# ============================================================

@dataclass
class KVCacheBlock:
    token_ids: list[int]
    key_states: list[torch.Tensor]
    value_states: list[torch.Tensor]
    last_access: float = field(default_factory=time.time)


class KVCacheManager:
    """Block-level LRU KV cache. Splits token sequences into fixed-size blocks keyed by
    content hash. Supports O(1) prefix lookup and LRU eviction."""

    def __init__(self, max_blocks: int = 256, block_size: int = 16):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self._blocks: OrderedDict[str, KVCacheBlock] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "total_tokens_saved": 0}

    def _hash_tokens(self, token_ids: list[int]) -> str:
        return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]

    def split_into_blocks(self, token_ids: list[int]) -> list[list[int]]:
        return [
            token_ids[i:i + self.block_size]
            for i in range(0, len(token_ids) - self.block_size + 1, self.block_size)
        ]

    def lookup_prefix(self, token_ids: list[int]):
        """Return (kv_list, matched_length). kv_list is None on miss."""
        matched_blocks: list[KVCacheBlock] = []
        for block_tokens in self.split_into_blocks(token_ids):
            bh = self._hash_tokens(block_tokens)
            if bh not in self._blocks:
                break
            block = self._blocks[bh]
            block.last_access = time.time()
            self._blocks.move_to_end(bh)
            matched_blocks.append(block)

        if not matched_blocks:
            self._stats["misses"] += 1
            return None, 0

        num_layers = len(matched_blocks[0].key_states)
        kv_list = []
        for layer_idx in range(num_layers):
            keys = torch.cat([b.key_states[layer_idx] for b in matched_blocks], dim=2)
            values = torch.cat([b.value_states[layer_idx] for b in matched_blocks], dim=2)
            kv_list.append((keys, values))

        matched_length = len(matched_blocks) * self.block_size
        self._stats["hits"] += 1
        self._stats["total_tokens_saved"] += matched_length
        return kv_list, matched_length

    def store(self, token_ids: list[int], key_values: list[tuple[torch.Tensor, torch.Tensor]], offset: int = 0):
        remaining = token_ids[offset:]
        num_layers = len(key_values)
        for block_idx, block_tokens in enumerate(self.split_into_blocks(remaining)):
            bh = self._hash_tokens(block_tokens)
            if bh in self._blocks:
                continue
            while len(self._blocks) >= self.max_blocks:
                self._blocks.popitem(last=False)
            start = offset + block_idx * self.block_size
            end = start + self.block_size
            self._blocks[bh] = KVCacheBlock(
                token_ids=block_tokens,
                key_states=[key_values[i][0][:, :, start:end, :].clone() for i in range(num_layers)],
                value_states=[key_values[i][1][:, :, start:end, :].clone() for i in range(num_layers)],
            )

    @property
    def stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "cached_blocks": len(self._blocks),
            "cached_tokens": len(self._blocks) * self.block_size,
            "hit_rate": f"{self._stats['hits'] / total:.2%}" if total > 0 else "0.00%",
        }

    def clear(self):
        self._blocks.clear()
        self._stats = {"hits": 0, "misses": 0, "total_tokens_saved": 0}


# ============================================================
# Logging & perf helpers (from serve.py)
# ============================================================

class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def setup_file_logging(log_file: str, mode: str = "a") -> TextIO | None:
    if not log_file:
        return None
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = path.open(mode, encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, log_handle)  # type: ignore[assignment]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    root = logging.getLogger("")
    root.addHandler(fh)
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        tgt = logging.getLogger(name)
        if not tgt.propagate:
            tgt.addHandler(fh)
        tgt.setLevel(logging.INFO)
    print(f"[log] serve log file: {path}")
    return log_handle


class _PerfAverages:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._totals: dict[str, float] = {}

    def record(self, stage_ms: dict[str, float]) -> tuple[int, dict[str, float]]:
        with self._lock:
            self._count += 1
            for k, v in stage_ms.items():
                self._totals[k] = self._totals.get(k, 0.0) + v
            return self._count, {k: self._totals[k] / self._count for k in stage_ms}


def _log_perf_average(name: str, count: int, averages: dict[str, float], keys: list[str]) -> None:
    fields = " ".join(f"{k}={averages[k]:.1f}ms" for k in keys if k in averages)
    logger.info("[PERF_AVG] %s n=%d %s", name, count, fields)


# ============================================================
# Model utilities
# ============================================================

def _inputs_to_model_device(inputs, model):
    inputs = inputs.to(model.device)
    if getattr(model, "dtype", None) is not None:
        inputs = inputs.to(model.dtype)
    return inputs


def _get_thinker_model(model):
    candidates = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            candidates.append(get_base())
        except Exception:
            pass
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(base)
        nested = getattr(base, "model", None)
        if nested is not None:
            candidates.append(nested)
    for c in candidates:
        t = getattr(c, "thinker", None)
        if t is not None:
            return t
    return None


def load_model(model_dir: str, lora_dir: str, torch_dtype: str = "auto"):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"[model] attention: {attn_impl}", flush=True)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map="auto", attn_implementation=attn_impl,
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


# ============================================================
# KV cache reconstruction helpers
# ============================================================

def _extract_kv_list(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if past_key_values is None:
        return None
    try:
        return [(k, v) for k, v, *_ in past_key_values]
    except Exception:
        pass
    if hasattr(past_key_values, "key_cache"):
        return [(past_key_values.key_cache[i], past_key_values.value_cache[i])
                for i in range(len(past_key_values.key_cache))]
    if isinstance(past_key_values, (tuple, list)):
        try:
            return [(k, v) for k, v in past_key_values]
        except Exception:
            pass
    return None


def _build_dynamic_cache(kv_list: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    try:
        from transformers import DynamicCache
        cache = DynamicCache()
        cache.key_cache = [k.detach().clone() for k, _ in kv_list]
        cache.value_cache = [v.detach().clone() for _, v in kv_list]
        cache._seen_tokens = cache.key_cache[0].shape[-2] if cache.key_cache else 0
        return cache
    except Exception as exc:
        logger.warning("[KV_CACHE] DynamicCache build failed (%s); using tuple fallback", exc)
        return tuple((k.detach().clone(), v.detach().clone()) for k, v in kv_list)


# ============================================================
# Pydantic models — OpenAI schema subset (from serve.py)
# ============================================================

class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    text: Optional[str] = None
    input_audio: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any
    tool_calls: Optional[Any] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = "qwen-omni"
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False

    def resolved_max_tokens(self) -> int:
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


def build_chat_response(choice: Choice, prompt_tokens: int, gen_tokens: int,
                         model_name: str | None = None) -> ChatResponse:
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name or _model_name,
        choices=[choice],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=gen_tokens,
                    total_tokens=prompt_tokens + gen_tokens),
    )


# ============================================================
# Audio handling (from serve.py)
# ============================================================

def _safe_b64decode(data: str) -> bytes:
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    data = data.replace('-', '+').replace('_', '/')
    data = re.sub(r'[^A-Za-z0-9+/]', '', data)
    r = len(data) % 4
    if r == 1:
        data = data[:-1]
    elif r:
        data += '=' * (4 - r)
    return base64.b64decode(data)


def _detect_audio_fmt(data: bytes, declared_fmt: str) -> tuple[str, str]:
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
    dl = declared_fmt.lower()
    if any(k in dl for k in ("pcm", "raw", "s16", "f32")):
        return ("s16le", "pcm")
    if dl in ("wav", "wave"):
        return ("s16le", "pcm")
    return ("", "bin")


def _is_wav_16k_mono_pcm16(data: bytes) -> bool:
    if not (len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE'):
        return False
    try:
        import io
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (EOFError, wave.Error):
        return False


def _run_ffmpeg(src: str, dst: str, fmt_flag: str = "", extra: list | None = None) -> bool:
    import subprocess
    cmd = ["ffmpeg", "-y"]
    if extra:
        cmd += extra
    if fmt_flag:
        cmd += ["-f", fmt_flag]
    cmd += ["-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", dst]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _write_audio_tmp(raw_bytes: bytes, fmt: str, tmp_dir: str,
                     sample_rate: int | None = None, channels: int | None = None) -> str:
    start = time.perf_counter()
    hex_head = raw_bytes[:16].hex() if raw_bytes else ""
    print(f"[AUDIO] declared_fmt={fmt!r} size={len(raw_bytes)} head={hex_head}", flush=True, file=sys.stderr)

    ffmpeg_fmt, src_ext = _detect_audio_fmt(raw_bytes, fmt)
    src_path = os.path.join(tmp_dir, f"audio_in_{uuid.uuid4().hex}.{src_ext}")
    wav_path = os.path.join(tmp_dir, f"audio_{uuid.uuid4().hex}.wav")
    with open(src_path, "wb") as f:
        f.write(raw_bytes)

    try:
        if ffmpeg_fmt == "wav" and _is_wav_16k_mono_pcm16(raw_bytes):
            print(f"[AUDIO] fast path 16kHz mono WAV ({(time.perf_counter()-start)*1000:.1f}ms)",
                  flush=True, file=sys.stderr)
            return src_path

        ok = False
        if ffmpeg_fmt == "s16le":
            if sample_rate and channels:
                ok = _run_ffmpeg(src_path, wav_path,
                                 extra=["-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels)])
            if not ok:
                for pcm_fmt, sr, ch in [("s16le","16000","1"),("s16le","48000","1"),
                                          ("s16le","44100","1"),("s16le","48000","2"),
                                          ("f32le","16000","1"),("f32le","48000","1")]:
                    trial = os.path.join(tmp_dir, f"trial_{uuid.uuid4().hex}.wav")
                    if _run_ffmpeg(src_path, trial, extra=["-f", pcm_fmt, "-ar", sr, "-ac", ch]):
                        shutil.move(trial, wav_path)
                        ok = True
                        break
                    try:
                        os.remove(trial)
                    except OSError:
                        pass
        elif ffmpeg_fmt:
            ok = _run_ffmpeg(src_path, wav_path, fmt_flag=ffmpeg_fmt)

        if not ok:
            ok = _run_ffmpeg(src_path, wav_path)
        if not ok:
            ok = _run_ffmpeg(src_path, wav_path, extra=["-f", "s16le", "-ar", "16000", "-ac", "1"])

        if ok:
            try:
                os.remove(src_path)
            except OSError:
                pass
            print(f"[AUDIO] prepare elapsed={(time.perf_counter()-start)*1000:.1f}ms",
                  flush=True, file=sys.stderr)
            return wav_path
        print(f"[WARN] all ffmpeg attempts failed for fmt={fmt!r}", flush=True, file=sys.stderr)
    except FileNotFoundError:
        print("[WARN] ffmpeg not found", flush=True, file=sys.stderr)

    try:
        os.remove(wav_path)
    except OSError:
        pass
    return src_path


# ============================================================
# Message conversion (from serve.py)
# ============================================================

def _compact_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    try:
        return json.dumps(json.loads(stripped), ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return text


def _tool_calls_to_compact_text(tool_calls: Any) -> str | None:
    if not tool_calls:
        return None
    first = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
    fn = first.get("function") if isinstance(first, dict) else getattr(first, "function", None)
    if not fn:
        return None
    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
    args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
    if not isinstance(name, str) or not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args.strip() or "{}")
        except json.JSONDecodeError:
            args = {}
    return json.dumps({"name": name, "arguments": args or {}}, ensure_ascii=False, separators=(",", ":"))


def _message_compact_text_for_model(msg: Message) -> str | None:
    if msg.role == "assistant":
        tc = _tool_calls_to_compact_text(getattr(msg, "tool_calls", None))
        if tc is not None:
            return tc
        if isinstance(msg.content, str):
            parsed = parse_model_output(msg.content)
            if parsed[0] == "tool_call":
                _, tool_name, args_str = parsed
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    return msg.content
                return json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False, separators=(",", ":"))
            if parsed[0] == "noise_do_not_act":
                return json.dumps({"name": "NoiseDoNotAct", "arguments": {}}, ensure_ascii=False, separators=(",", ":"))
    if msg.role == "tool" and isinstance(msg.content, str):
        return _compact_json_text(msg.content)
    return None


def _assistant_message_is_tool_turn(msg: Message) -> bool:
    if msg.role != "assistant":
        return False
    if getattr(msg, "tool_calls", None):
        return True
    content = msg.content
    if isinstance(content, list):
        content = "".join(c.get("text","") for c in content if isinstance(c,dict) and c.get("type")=="text").strip()
    if not isinstance(content, str) or not content:
        return True
    return parse_model_output(content)[0] in {"tool_call", "noise_do_not_act"}


def _strip_historical_tool_messages(messages: List[Message]) -> List[Message]:
    if not messages:
        return messages
    last_user_idx = next((i for i in range(len(messages)-1,-1,-1) if messages[i].role=="user"), -1)
    last_tc_idx = next((i for i in range(len(messages)-1,-1,-1) if _assistant_message_is_tool_turn(messages[i])), -1)

    keep: set[int] = set()
    if last_tc_idx > last_user_idx:
        keep.add(last_tc_idx)
        for i in range(last_tc_idx+1, len(messages)):
            if messages[i].role == "tool":
                keep.add(i)
            else:
                break

    filtered, dropped_tc, dropped_tr = [], 0, 0
    for i, msg in enumerate(messages):
        if i in keep:
            filtered.append(msg)
        elif msg.role == "tool":
            dropped_tr += 1
        elif _assistant_message_is_tool_turn(msg):
            dropped_tc += 1
        else:
            filtered.append(msg)

    if dropped_tc or dropped_tr:
        logger.info("[HISTORY] stripped tool turns: assistant_tc=%d tool_results=%d", dropped_tc, dropped_tr)
    return filtered


def _messages_to_qwen(messages: List[Message], system_prompt: str, tmp_dir: str) -> tuple[list, list]:
    messages = _strip_historical_tool_messages(messages)
    qwen_msgs = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    tmp_files: list[str] = []

    for msg in messages:
        if msg.role == "system":
            continue
        compact = _message_compact_text_for_model(msg)
        if compact is not None:
            qwen_msgs.append({"role": msg.role, "content": [{"type": "text", "text": compact}]})
            continue

        if isinstance(msg.content, str):
            qwen_content = [{"type": "text", "text": _compact_json_text(msg.content)}]
        else:
            qwen_content = []
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype == "text":
                        qwen_content.append({"type": "text", "text": _compact_json_text(part.get("text", ""))})
                    elif ptype == "input_audio":
                        info = part.get("input_audio", {})
                        b64 = info.get("data", "")
                        if b64:
                            raw = _safe_b64decode(b64)
                            sr = int(info["sample_rate"]) if info.get("sample_rate") else None
                            ch = int(info.get("channels") or info.get("channel") or 0) or None
                            p = _write_audio_tmp(raw, info.get("format","wav"), tmp_dir, sr, ch)
                            tmp_files.append(p)
                            qwen_content.append({"type": "audio", "audio": p})
                else:
                    if part.type == "text" and part.text:
                        qwen_content.append({"type": "text", "text": _compact_json_text(part.text)})
                    elif part.type == "input_audio" and part.input_audio:
                        b64 = part.input_audio.get("data", "")
                        if b64:
                            raw = _safe_b64decode(b64)
                            sr = int(part.input_audio["sample_rate"]) if part.input_audio.get("sample_rate") else None
                            ch = int(part.input_audio.get("channels") or part.input_audio.get("channel") or 0) or None
                            p = _write_audio_tmp(raw, part.input_audio.get("format","wav"), tmp_dir, sr, ch)
                            tmp_files.append(p)
                            qwen_content.append({"type": "audio", "audio": p})
        qwen_msgs.append({"role": msg.role, "content": qwen_content})

    return qwen_msgs, tmp_files


def _print_qwen_messages(qwen_messages: list) -> None:
    LABELS = {"system": "SYSTEM", "user": "USER  ", "assistant": "ASST  ", "tool": "TOOL  "}
    print(f"\n{'═'*60}\n  MODEL INPUT  ({len(qwen_messages)} turns)\n{'═'*60}", flush=True, file=sys.stderr)
    for i, msg in enumerate(qwen_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        txt = p["text"]
                        parts.append(f"<text: {txt[:100]}... ({len(txt)} chars)>" if role == "system" else txt)
                    elif p.get("type") == "audio":
                        parts.append(f"<audio: {p.get('audio','')}>")
            text = " | ".join(parts)
        else:
            text = str(content)
            if role == "system":
                text = f"{text[:100]}... ({len(text)} chars)"
        print(f"[{i}] {LABELS.get(role, role[:6].upper())} │ {text}", flush=True, file=sys.stderr)
    print(f"{'═'*60}\n", flush=True, file=sys.stderr)


# ============================================================
# Inference with block-level KV cache
# ============================================================

def run_inference(
    model,
    processor,
    qwen_messages: list,
    max_new_tokens: int,
    temperature: float,
    kv_cache_manager: Optional[KVCacheManager] = None,
) -> tuple[str, int, int]:
    t0 = time.perf_counter()
    _print_qwen_messages(qwen_messages)

    text = processor.apply_chat_template(qwen_messages, add_generation_prompt=True, tokenize=False)
    chat_template_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    audios, images, videos = process_mm_info(qwen_messages, use_audio_in_video=False)
    mm_ms = (time.perf_counter() - t1) * 1000

    audio_ph = text.count("<|AUDIO|>")
    if audio_ph != len(audios or []):
        logger.warning("[MM] audio mismatch: template=%d process_mm_info=%d; cache disabled for this request",
                       audio_ph, len(audios or []))

    t2 = time.perf_counter()
    inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    processor_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    inputs = _inputs_to_model_device(inputs, model)
    move_ms = (time.perf_counter() - t3) * 1000

    # ── Block-level KV cache lookup ──────────────────────────────────────
    audio_ok = (audio_ph == len(audios or []))
    has_media = bool(audios or images or videos)
    can_cache = kv_cache_manager is not None and audio_ok

    # text-only: cache full token prefix
    # audio/media: cache system prompt prefix only (pure text, stable across requests)
    use_full_cache = can_cache and not has_media
    use_system_cache = can_cache and has_media and bool(
        qwen_messages and qwen_messages[0].get("role") == "system"
    )

    prompt_len = int(inputs["input_ids"].shape[-1])
    cached_length = 0
    token_list: list[int] = []
    system_token_list: list[int] = []

    if use_full_cache:
        token_list = inputs["input_ids"][0].tolist()
        t_lookup = time.perf_counter()
        kv_list, cached_length = kv_cache_manager.lookup_prefix(token_list)
        lookup_ms = (time.perf_counter() - t_lookup) * 1000
        if cached_length > 0 and kv_list is not None:
            t_build = time.perf_counter()
            # Keep full input_ids; generate() slices unprocessed tokens via cache_position.
            inputs["past_key_values"] = _build_dynamic_cache(kv_list)
            inputs["attention_mask"] = torch.ones(
                1, prompt_len, dtype=torch.long, device=inputs["input_ids"].device
            )
            build_ms = (time.perf_counter() - t_build) * 1000
            logger.info("[KV_CACHE] full hit: cached=%d total=%d lookup=%.1fms build_cache=%.1fms",
                        cached_length, prompt_len, lookup_ms, build_ms)
        else:
            logger.info("[KV_CACHE] full miss lookup=%.1fms (will prefill+store after generate)",
                        lookup_ms)

    elif use_system_cache:
        # Tokenize only the system message — its token IDs are identical in every request
        # regardless of what audio/user content follows.
        sys_content = qwen_messages[0]["content"]
        system_text = (
            sys_content[0].get("text", "") if isinstance(sys_content, list) and sys_content
            else str(sys_content)
        )
        sys_only = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
        t_sys_tok = time.perf_counter()
        sys_fmt = processor.apply_chat_template(sys_only, add_generation_prompt=False, tokenize=False)
        sys_enc = processor(text=sys_fmt, return_tensors="pt", padding=False, use_audio_in_video=False)
        system_token_list = sys_enc["input_ids"][0].tolist()
        sys_tok_ms = (time.perf_counter() - t_sys_tok) * 1000

        t_lookup = time.perf_counter()
        kv_list, cached_length = kv_cache_manager.lookup_prefix(system_token_list)
        lookup_ms = (time.perf_counter() - t_lookup) * 1000
        if cached_length > 0 and kv_list is not None:
            t_build = time.perf_counter()
            inputs["past_key_values"] = _build_dynamic_cache(kv_list)
            inputs["attention_mask"] = torch.ones(
                1, prompt_len, dtype=torch.long, device=inputs["input_ids"].device
            )
            build_ms = (time.perf_counter() - t_build) * 1000
            logger.info("[KV_CACHE] system hit: cached=%d total=%d "
                        "sys_tok=%.1fms lookup=%.1fms build_cache=%.1fms",
                        cached_length, prompt_len, sys_tok_ms, lookup_ms, build_ms)
        else:
            logger.info("[KV_CACHE] system miss sys_tok=%.1fms lookup=%.1fms "
                        "(will prefill+store system prompt)",
                        sys_tok_ms, lookup_ms)

    # ── Generate ─────────────────────────────────────────────────────────
    # On cache hit for text-only requests, bypass model.generate() (Omni wrapper) and
    # call thinker.generate() directly.  thinker is a standard CausalLM whose
    # prepare_inputs_for_generation() correctly slices input_ids to only the
    # unprocessed suffix via past_key_values.get_seq_length(), giving real prefill
    # savings.  model.generate() does not reliably honour externally injected
    # past_key_values, so using it on cache hit adds cloning overhead without benefit.
    t4 = time.perf_counter()
    cache_active = cached_length > 0 and (use_full_cache or use_system_cache)
    _thinker = _get_thinker_model(model) if (cache_active and not has_media) else None

    if cache_active and _thinker is not None:
        gen_kwargs: dict = {
            "input_ids": inputs["input_ids"],
            "past_key_values": inputs["past_key_values"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            out_ids = _thinker.generate(**gen_kwargs)
    else:
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                thinker_max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_audio=False,
            )
    generate_ms = (time.perf_counter() - t4) * 1000

    t5 = time.perf_counter()
    gen_ids = out_ids[:, prompt_len:]
    decoded = processor.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decode_ms = (time.perf_counter() - t5) * 1000

    # ── Store KV cache on miss ────────────────────────────────────────────
    store_ids: list[int] | None = None
    store_label = ""
    if use_full_cache and cached_length == 0 and token_list:
        store_ids, store_label = token_list, "full"
    elif use_system_cache and cached_length == 0 and system_token_list:
        store_ids, store_label = system_token_list, "system"

    if store_ids is not None:
        thinker = _get_thinker_model(model)
        if thinker is not None:
            try:
                t_prefill = time.perf_counter()
                with torch.no_grad():
                    ids = torch.tensor([store_ids], dtype=torch.long, device=inputs["input_ids"].device)
                    prefill_out = thinker(input_ids=ids, use_cache=True)
                prefill_ms = (time.perf_counter() - t_prefill) * 1000
                kv_list = _extract_kv_list(prefill_out.past_key_values)
                if kv_list:
                    t_store = time.perf_counter()
                    kv_cache_manager.store(store_ids, kv_list, offset=0)
                    store_ms = (time.perf_counter() - t_store) * 1000
                    logger.info("[KV_CACHE] stored %s: tokens=%d blocks=%d "
                                "prefill=%.1fms store=%.1fms",
                                store_label, len(store_ids),
                                kv_cache_manager.stats["cached_blocks"],
                                prefill_ms, store_ms)
            except Exception as exc:
                import traceback
                logger.warning("[KV_CACHE] store failed: %s", exc)
                traceback.print_exc(file=sys.stderr)

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[PERF] chat_template=%.1fms mm=%.1fms processor=%.1fms to_device=%.1fms "
        "generate=%.1fms decode=%.1fms total=%.1fms kv_cache=%s cached_tokens=%d",
        chat_template_ms, mm_ms, processor_ms, move_ms, generate_ms, decode_ms, total_ms,
        "hit" if cached_length > 0 else ("miss" if (use_full_cache or use_system_cache) else "off"),
        cached_length,
    )
    count, avgs = _inference_perf_averages.record({
        "chat_template": chat_template_ms, "mm": mm_ms, "processor": processor_ms,
        "to_device": move_ms, "generate": generate_ms, "decode": decode_ms, "total": total_ms,
    })
    _log_perf_average("inference", count, avgs,
                      ["chat_template","mm","processor","to_device","generate","decode","total"])
    return decoded.strip(), prompt_len, int(gen_ids.shape[-1])


# ============================================================
# Output parsing (from serve.py)
# ============================================================

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*\})", re.DOTALL)


def _parse_tool_call_json(text: str) -> tuple[str, dict] | None:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str):
        return None
    if name == "NoiseDoNotAct":
        return name, {}
    args = data.get("arguments", {})
    return (name, args) if isinstance(args, dict) else None


def _parse_tool_call_json_array(text: str) -> list[tuple[str, dict]] | None:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    calls = []
    for item in data:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        if not isinstance(name, str) or "arguments" not in item:
            return None
        args = item.get("arguments", {})
        if not isinstance(args, dict):
            return None
        calls.append((name, args))
    return calls


def parse_model_output(text: str) -> tuple:
    if text.strip() == "Reject":
        return ("reject",)

    arr = _parse_tool_call_json_array(text)
    if arr is not None:
        if any(n == "NoiseDoNotAct" for n, _ in arr):
            return ("noise_do_not_act",)
        return ("tool_calls", [(n, json.dumps(a, ensure_ascii=False, separators=(",",":"))) for n, a in arr])

    single = _parse_tool_call_json(text)
    if single is not None:
        n, a = single
        if n == "NoiseDoNotAct":
            return ("noise_do_not_act",)
        return ("tool_call", n, json.dumps(a, ensure_ascii=False, separators=(",",":")))

    m = _ACTION_RE.search(text)
    if m:
        n, args_str = m.group(1).strip(), m.group(2).strip()
        if n == "NoiseDoNotAct":
            return ("noise_do_not_act",)
        try:
            args_str = json.dumps(json.loads(args_str), ensure_ascii=False, separators=(",",":"))
        except json.JSONDecodeError:
            pass
        return ("tool_call", n, args_str)

    content = text
    if text.startswith("Final Answer:"):
        content = text[len("Final Answer:"):].strip()
    elif text.startswith("Clarify:"):
        content = text[len("Clarify:"):].strip()
    return ("text", content)


def _choice_from_parsed(parsed: tuple) -> Choice:
    if parsed[0] == "tool_call":
        _, name, args = parsed
        return Choice(
            message=AssistantMessage(tool_calls=[ToolCall(
                id=f"call_{uuid.uuid4().hex[:22]}", function=ToolFunction(name=name, arguments=args))]),
            finish_reason="tool_calls",
        )
    if parsed[0] == "tool_calls":
        _, calls = parsed
        return Choice(
            message=AssistantMessage(tool_calls=[
                ToolCall(id=f"call_{uuid.uuid4().hex[:22]}", index=i,
                         function=ToolFunction(name=n, arguments=a))
                for i, (n, a) in enumerate(calls)
            ]),
            finish_reason="tool_calls",
        )
    if parsed[0] in ("noise_do_not_act", "reject"):
        label = "NOISE_DO_NOT_ACT" if parsed[0] == "noise_do_not_act" else "REJECT"
        print(f"[{label}] suppressed output", flush=True, file=sys.stderr)
        return Choice(message=AssistantMessage(content=""), finish_reason="stop")
    _, content = parsed
    return Choice(message=AssistantMessage(content=content), finish_reason="stop")


# ============================================================
# Request/Response persistence (from serve.py)
# ============================================================

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


def _save_request_artifacts(request_id: str, tmp_files: list[str], response: "ChatResponse",
                              messages: List[Message], model_messages: list | None = None) -> None:
    try:
        req_dir = _SAVE_DIR / request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        for i, fp in enumerate(tmp_files):
            if os.path.exists(fp):
                name = f"audio_{i}{Path(fp).suffix or '.wav'}"
                shutil.copy2(fp, req_dir / name)
                saved.append({"source": fp, "saved": name})
        q = _last_text_query(messages)
        if q:
            (req_dir / "query.txt").write_text(q, encoding="utf-8")
        (req_dir / "response.json").write_text(response.model_dump_json(indent=2), encoding="utf-8")
        if model_messages is not None:
            (req_dir / "model_request.json").write_text(
                json.dumps({"messages": model_messages, "audio_files": saved}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info("[SAVE] artifacts → %s", req_dir)
    except Exception as e:
        logger.warning("[SAVE] failed for %s: %s", request_id, e)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Qwen2.5-Omni Inference Server", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    print(f"[422] {exc.errors()}", flush=True, file=sys.stderr)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Globals set at startup
_model = None
_processor = None
_system_prompt = ""
_model_name = "qwen-omni"
_tmp_dir = tempfile.mkdtemp(prefix="qwen_serve_")
_inference_perf_averages = _PerfAverages()
_request_perf_averages = _PerfAverages()
_kv_cache_manager: Optional[KVCacheManager] = None


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": _model_name, "object": "model", "created": int(time.time()), "owned_by": "local"}
    ]}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    import traceback
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    req_start = time.perf_counter()
    try:
        t = time.perf_counter()
        qwen_msgs, tmp_files = _messages_to_qwen(req.messages, _system_prompt, _tmp_dir)
        convert_ms = (time.perf_counter() - t) * 1000
    except Exception as e:
        print(f"\n[ERROR] _messages_to_qwen: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=422, detail=f"Message conversion error: {e}") from e

    try:
        t = time.perf_counter()
        reply, prompt_tokens, gen_tokens = run_inference(
            _model, _processor, qwen_msgs,
            max_new_tokens=req.resolved_max_tokens(),
            temperature=req.temperature,
            kv_cache_manager=_kv_cache_manager,
        )
        inference_ms = (time.perf_counter() - t) * 1000
    except Exception as e:
        print(f"\n[ERROR] run_inference: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    print(f"[MODEL_OUT] {reply!r}", flush=True, file=sys.stderr)
    t = time.perf_counter()
    choice = _choice_from_parsed(parse_model_output(reply))
    parse_ms = (time.perf_counter() - t) * 1000

    resp = build_chat_response(choice, prompt_tokens, gen_tokens)

    t = time.perf_counter()
    _save_request_artifacts(resp.id, tmp_files, resp, req.messages, qwen_msgs)
    save_ms = (time.perf_counter() - t) * 1000

    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    total_ms = (time.perf_counter() - req_start) * 1000
    logger.info("[PERF] request convert=%.1fms inference=%.1fms parse=%.1fms save=%.1fms total=%.1fms",
                convert_ms, inference_ms, parse_ms, save_ms, total_ms)
    count, avgs = _request_perf_averages.record(
        {"convert": convert_ms, "inference": inference_ms, "parse": parse_ms, "save": save_ms, "total": total_ms}
    )
    _log_perf_average("request", count, avgs, ["convert","inference","parse","save","total"])
    return resp


@app.get("/v1/cache/stats")
async def cache_stats():
    if _kv_cache_manager is None:
        return {"error": "KV cache disabled (--prompt-cache none)"}
    return _kv_cache_manager.stats


@app.post("/v1/cache/clear")
async def cache_clear():
    if _kv_cache_manager:
        _kv_cache_manager.clear()
    return {"status": "ok", "message": "KV cache cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _model_name,
        "kv_cache": _kv_cache_manager.stats if _kv_cache_manager else None,
    }


# ============================================================
# CLI entrypoint
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible Qwen2.5-Omni server with block-level KV cache")
    p.add_argument("--model-dir", required=True, help="Base model directory")
    p.add_argument("--lora-dir", default="", help="Optional LoRA adapter directory")
    p.add_argument("--system-prompt-file", default=str(_DEFAULT_SP_FILE))
    p.add_argument("--system-prompt", default="", help="Inline system prompt (overrides file)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="qwen-omni")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--prompt-cache", default="kv", choices=["none", "kv"],
                   help="'kv' enables block-level LRU KV prefix caching (default: kv)")
    p.add_argument("--cache-blocks", type=int, default=256, help="Max KV cache blocks (LRU eviction)")
    p.add_argument("--block-size", type=int, default=16, help="Tokens per KV cache block")
    p.add_argument("--log-file", default=str(_DEFAULT_LOG_FILE))
    p.add_argument("--log-file-mode", default="a", choices=["a", "w"])
    return p.parse_args()


def main():
    global _model, _processor, _system_prompt, _model_name, _kv_cache_manager

    args = parse_args()
    setup_file_logging(args.log_file, args.log_file_mode)

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
    print(f"[prompt_cache] mode={args.prompt_cache}")

    if args.prompt_cache == "kv":
        _kv_cache_manager = KVCacheManager(max_blocks=args.cache_blocks, block_size=args.block_size)
        print(f"[kv_cache] block_size={args.block_size} max_blocks={args.cache_blocks} "
              f"capacity=~{args.cache_blocks * args.block_size} tokens")

    print(f"[model] loading {args.model_dir} ...")
    _model, _processor = load_model(args.model_dir, args.lora_dir, args.torch_dtype)
    print(f"[model] ready  lora={args.lora_dir or 'none'}")
    print(f"[server] http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

