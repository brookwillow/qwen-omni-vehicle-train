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
import json
import logging
import os
import re
import tempfile
import time
import uuid
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


_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_SP_FILE = _PROJECT_DIR / "data" / "system-prompt.txt"


# ── Pydantic models (OpenAI schema subset) ───────────────────

class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str          # "text" | "input_audio"
    text: Optional[str] = None
    input_audio: Optional[Dict[str, str]] = None  # {"data": "<b64>", "format": "wav"}


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
        attn_impl = "eager"

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
    """Decode base64 (standard or URL-safe), tolerating whitespace and missing padding."""
    # Normalize URL-safe base64 chars (- -> +, _ -> /)
    data = data.replace('-', '+').replace('_', '/')
    # Strip whitespace and all non-base64 chars (including stray '=')
    data = re.sub(r'[^A-Za-z0-9+/]', '', data)
    # Re-pad correctly based on pure data character count
    missing = len(data) % 4
    if missing:
        data += '=' * (4 - missing)
    return base64.b64decode(data)


def _detect_audio_fmt(data: bytes, declared_fmt: str) -> tuple[str, str]:
    """Return (ffmpeg_input_flag, file_ext) based on magic bytes.

    ffmpeg_input_flag is passed as -f <flag> to tell ffmpeg the actual input
    format (important for raw PCM which has no container header).
    """
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
    # Raw PCM detected — declared format or keyword hints
    dl = declared_fmt.lower()
    if any(k in dl for k in ("pcm", "raw", "s16", "f32")):
        return ("s16le", "pcm")   # assume 16-bit LE PCM
    # Unknown: give ffmpeg the declared extension and let it probe
    ext = declared_fmt.lower().lstrip('.')
    return ("", ext)


def _write_audio_tmp(raw_bytes: bytes, fmt: str, tmp_dir: str) -> str:
    """Write audio bytes to a temp WAV (16 kHz mono) that librosa can read.

    Detects actual format from magic bytes, then uses ffmpeg to convert.
    Falls back to the raw file if ffmpeg is unavailable.
    """
    import subprocess
    import sys

    ffmpeg_fmt, src_ext = _detect_audio_fmt(raw_bytes, fmt)
    src_path = os.path.join(tmp_dir, f"audio_in_{uuid.uuid4().hex}.{src_ext}")
    wav_path = os.path.join(tmp_dir, f"audio_{uuid.uuid4().hex}.wav")

    with open(src_path, "wb") as f:
        f.write(raw_bytes)

    # Build ffmpeg command: for raw PCM supply -f + sample rate/channels
    if ffmpeg_fmt == "s16le":
        cmd = ["ffmpeg", "-y",
               "-f", "s16le", "-ar", "16000", "-ac", "1",
               "-i", src_path,
               "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    elif ffmpeg_fmt:
        cmd = ["ffmpeg", "-y", "-f", ffmpeg_fmt, "-i", src_path,
               "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", src_path,
               "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]

    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            try:
                os.remove(src_path)
            except OSError:
                pass
            return wav_path
        else:
            stderr_msg = result.stderr.decode(errors='replace')
            # Show last 500 chars which usually contains the actual error
            print(f"[WARN] ffmpeg failed (fmt={ffmpeg_fmt!r} ext={src_ext!r}): "
                  f"...{stderr_msg[-500:]}", flush=True, file=sys.stderr)
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; passing raw file to librosa", flush=True, file=sys.stderr)

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
            # Override the default system prompt
            text = msg.content if isinstance(msg.content, str) else ""
            for part in (msg.content if isinstance(msg.content, list) else []):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part["text"]
            qwen_msgs[0] = {"role": "system", "content": [{"type": "text", "text": text}]}
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
                            tmp_path = _write_audio_tmp(raw, fmt, tmp_dir)
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
                            tmp_path = _write_audio_tmp(raw, fmt, tmp_dir)
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
) -> tuple[str, int, int]:
    text = processor.apply_chat_template(qwen_messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(qwen_messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device)
    if getattr(model, "dtype", None) is not None:
        inputs = inputs.to(model.dtype)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_audio=False,
        )

    prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    gen_ids = out_ids[:, prompt_len:]
    decoded = processor.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded.strip(), int(prompt_len), int(gen_ids.shape[-1])


# ── Output parsing ───────────────────────────────────────────

_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*\})",
    re.DOTALL,
)


def parse_model_output(text: str) -> tuple:
    """Parse model text output into structured form.

    Returns:
        ("tool_call", tool_name: str, args_json: str)
      | ("text", content: str)
    """
    m = _ACTION_RE.search(text)
    if m:
        tool_name = m.group(1).strip()
        args_str = m.group(2).strip()
        # Validate JSON; keep raw string regardless
        try:
            json.loads(args_str)
        except json.JSONDecodeError:
            pass  # still return as-is; caller receives raw string
        return ("tool_call", tool_name, args_str)

    # Strip "Final Answer: " prefix if present
    if text.startswith("Final Answer:"):
        content = text[len("Final Answer:"):].strip()
    else:
        content = text
    return ("text", content)


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(title="Qwen2.5-Omni Inference Server", version="1.0.0")


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

    try:
        qwen_msgs, tmp_files = _messages_to_qwen(req.messages, _system_prompt, _tmp_dir)
    except Exception as e:
        print(f"\n[ERROR] _messages_to_qwen failed: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=422, detail=f"Message conversion error: {e}") from e

    try:
        reply, prompt_tokens, gen_tokens = run_inference(
            _model, _processor, qwen_msgs,
            max_new_tokens=req.resolved_max_tokens(),
            temperature=req.temperature,
        )
    except Exception as e:
        print(f"\n[ERROR] run_inference failed: {e}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass

    parsed = parse_model_output(reply)
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

    return ChatResponse(
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
    return p.parse_args()


def main():
    global _model, _processor, _system_prompt, _model_name

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

    print(f"[model] loading {args.model_dir} ...")
    _model, _processor = load_model(args.model_dir, args.lora_dir, args.torch_dtype)
    print(f"[model] ready  lora={args.lora_dir or 'none'}")
    print(f"[server] starting on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
