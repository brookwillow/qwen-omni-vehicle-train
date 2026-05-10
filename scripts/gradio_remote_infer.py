#!/usr/bin/env python3
"""Gradio UI for local audio capture and remote Qwen-Omni inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from record_remote_infer import (
    DEFAULT_SERVER,
    analyze_wav,
    build_payload,
    post_json,
    record_wav,
)


MIN_AUDIO_RMS = 5


def _is_16k_mono_pcm16_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == 16000
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
            )
    except (wave.Error, EOFError):
        return False


def _extract_result(resp: dict[str, Any]) -> tuple[str, str, str]:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        fn = call.get("function") or {}
        args_raw = fn.get("arguments", "")
        try:
            args_obj = json.loads(args_raw)
            args_text = json.dumps(args_obj, ensure_ascii=False, indent=2)
        except (TypeError, json.JSONDecodeError):
            args_text = str(args_raw)
        summary = f"Tool Call\n\nname: `{fn.get('name', '')}`"
        return summary, args_text, json.dumps(resp, ensure_ascii=False, indent=2)

    content = message.get("content", "")
    summary = "Text Response"
    return summary, str(content), json.dumps(resp, ensure_ascii=False, indent=2)


def _convert_to_wav_for_analysis(path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to analyze non-WAV browser recordings.")
    fd, name = tempfile.mkstemp(prefix="qwen_omni_analyze_", suffix=".wav")
    os.close(fd)
    wav_path = Path(name)
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg could not decode audio: {detail}")
    return wav_path


def normalize_audio_for_payload(path: Path) -> tuple[Path, bool]:
    """Return a 16kHz mono PCM16 WAV path for upload."""
    if _is_16k_mono_pcm16_wav(path):
        return path, False
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to normalize audio before upload.")
    fd, name = tempfile.mkstemp(prefix="qwen_omni_payload_", suffix=".wav")
    os.close(fd)
    wav_path = Path(name)
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        "-f", "wav", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg could not normalize audio: {detail}")
    return wav_path, True


def analyze_audio_file(path: Path) -> dict[str, Any]:
    try:
        stats = dict(analyze_wav(path))
        stats["analysis_source"] = "wav"
        return stats
    except (wave.Error, EOFError):
        wav_path = _convert_to_wav_for_analysis(path)
        try:
            stats = dict(analyze_wav(wav_path))
            stats["analysis_source"] = "ffmpeg"
            stats["original_size_bytes"] = path.stat().st_size
            stats["original_suffix"] = path.suffix
            return stats
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass


def validate_audio_has_sound(path: Path) -> dict[str, Any]:
    stats = analyze_audio_file(path)
    if stats.get("size_bytes", 0) < 1024 and stats.get("original_size_bytes", 1024) < 1024:
        raise RuntimeError(f"audio file is too small: {path.stat().st_size} bytes")
    if stats.get("rms", 0) < MIN_AUDIO_RMS or stats.get("peak_abs", 0) == 0:
        raise RuntimeError(
            "audio appears to be silent: "
            f"rms={stats.get('rms')} peak_abs={stats.get('peak_abs')} duration={stats.get('duration_sec')}s"
        )
    return stats


def infer_audio(
    audio_path: str | None,
    server: str,
    model: str,
    hint_text: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, str, str]:
    if not audio_path:
        return "No Audio", "Please record or upload a WAV file first.", "{}"

    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return "Audio Missing", f"File not found: {path}", "{}"

    try:
        validate_audio_has_sound(path)
        url = f"{server.rstrip('/')}/v1/chat/completions"
        payload_path, should_cleanup = normalize_audio_for_payload(path)
        try:
            payload = build_payload(
                path,
                model,
                int(max_tokens),
                float(temperature),
                hint_text.strip(),
                payload_audio_path=payload_path,
            )
            resp = post_json(url, payload, float(timeout))
        finally:
            if should_cleanup:
                try:
                    payload_path.unlink()
                except OSError:
                    pass
        return _extract_result(resp)
    except Exception as exc:
        return "Request Failed", str(exc), "{}"


def analyze_audio_for_ui(audio_path: str | None) -> str:
    if not audio_path:
        return "No audio selected."

    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return f"File not found: {path}"

    try:
        stats = analyze_audio_file(path)
        status = "OK" if stats.get("rms", 0) >= 5 and stats.get("peak_abs", 0) else "SILENT"
        msg = f"{status}: {path}\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
        if status == "SILENT":
            msg += (
                "\n\n⚠️ Audio is silent! Check:\n"
                "  1. macOS: System Settings → Privacy & Security → Microphone → enable for your browser\n"
                "  2. Browser: allow microphone access for this page (click lock icon in address bar)\n"
                "  3. Try the 'Backend Record' button instead (uses ffmpeg avfoundation directly)"
            )
        return msg
    except Exception as exc:
        return f"Could not analyze audio: {exc}"


def record_backend(duration: float, sample_rate: int) -> tuple[str | None, str]:
    try:
        fd, name = tempfile.mkstemp(prefix="qwen_omni_gradio_", suffix=".wav")
        try:
            os.close(fd)
        except OSError:
            pass
        path = Path(name)
        path.unlink(missing_ok=True)
        record_wav(path, float(duration), int(sample_rate))
        stats_text = analyze_audio_for_ui(str(path))
        return str(path), f"Recorded {duration:.1f}s.\n{stats_text}"
    except Exception as exc:
        return None, f"Backend recording failed: {exc}"


def record_backend_and_infer(
    duration: float,
    sample_rate: int,
    server: str,
    model: str,
    hint_text: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str | None, str, str, str, str]:
    audio_path, status = record_backend(duration, sample_rate)
    if not audio_path:
        return None, status, "Record Failed", status, "{}"
    summary, parsed, raw = infer_audio(audio_path, server, model, hint_text, max_tokens, temperature, timeout)
    return audio_path, status, summary, parsed, raw


def infer_messages(
    messages_json: str,
    server: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, str, str]:
    """Send a messages array (JSON) to the server and return results."""
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        return "JSON Error", f"Invalid JSON: {exc}", "{}"
    if not isinstance(messages, list) or not messages:
        return "Format Error", "messages must be a non-empty JSON array.", "{}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    try:
        url = f"{server.rstrip('/')}/v1/chat/completions"
        resp = post_json(url, payload, float(timeout))
        return _extract_result(resp)
    except Exception as exc:
        return "Request Failed", str(exc), "{}"


_MESSAGES_PLACEHOLDER = json.dumps(
    [{"role": "user", "content": "帮我把主驾车窗打开"}],
    ensure_ascii=False, indent=2,
)

_MULTITURN_EXAMPLE = json.dumps(
    [
        {"role": "user", "content": "帮我把空调打开"},
        {"role": "assistant", "content": 'Action: ClimateControl\nAction Input: {"action": "打开", "device": "空调"}'},
        {"role": "user", "content": "Tool Result: {\"status\": \"success\"}"},
        {"role": "assistant", "content": "Final Answer: 好的，空调已为您打开。"},
        {"role": "user", "content": "温度调到26度"},
    ],
    ensure_ascii=False, indent=2,
)


def build_app(default_server: str, default_model: str):
    import gradio as gr

    with gr.Blocks(title="Qwen-Omni Remote Inference") as demo:
        gr.Markdown("# Qwen-Omni Remote Inference")

        # Shared server settings at the top
        with gr.Row():
            server = gr.Textbox(label="Server", value=default_server)
            model = gr.Textbox(label="Model", value=default_model)

        with gr.Tabs():

            # ── Tab 1: Multi-turn Text ────────────────────────────
            with gr.Tab("📝 多轮文本测试"):
                gr.Markdown(
                    "直接粘贴 `messages` JSON 数组进行测试，支持多轮上下文。\n\n"
                    "格式：`[{\"role\": \"user\", \"content\": \"...\"}]`"
                )
                with gr.Row():
                    txt_max_tokens = gr.Number(label="Max Tokens", value=128, precision=0)
                    txt_temperature = gr.Number(label="Temperature", value=0.0)
                    txt_timeout = gr.Number(label="Timeout Seconds", value=60)

                messages_input = gr.Code(
                    label="messages (JSON)",
                    language="json",
                    value=_MESSAGES_PLACEHOLDER,
                    lines=12,
                )

                with gr.Row():
                    btn_send_text = gr.Button("Send Messages", variant="primary")
                    btn_example_single = gr.Button("示例：单轮")
                    btn_example_multi = gr.Button("示例：多轮续接")

                txt_summary = gr.Markdown(label="Summary")
                txt_parsed = gr.Code(label="Parsed Output", language="json")
                txt_raw = gr.Code(label="Raw Response", language="json")

                btn_send_text.click(
                    infer_messages,
                    inputs=[messages_input, server, model, txt_max_tokens, txt_temperature, txt_timeout],
                    outputs=[txt_summary, txt_parsed, txt_raw],
                )
                btn_example_single.click(
                    lambda: _MESSAGES_PLACEHOLDER,
                    outputs=[messages_input],
                )
                btn_example_multi.click(
                    lambda: _MULTITURN_EXAMPLE,
                    outputs=[messages_input],
                )

            # ── Tab 2: Audio ──────────────────────────────────────
            with gr.Tab("🎙️ 音频测试"):
                gr.Markdown("Record or upload local audio, then send it to the remote `serve.py` endpoint.")

                audio = gr.Audio(
                    label="Browser Audio",
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                )

                with gr.Row():
                    duration = gr.Number(label="Backend Record Seconds", value=4.0)
                    sample_rate = gr.Number(label="Sample Rate", value=16000, precision=0)
                    backend_record = gr.Button("Backend Record")
                    backend_record_send = gr.Button("Backend Record + Send", variant="secondary")

                record_status = gr.Textbox(label="Record Status", interactive=False)

                with gr.Row():
                    hint_text = gr.Textbox(
                        label="Hint Text",
                        value="",
                        placeholder="Optional. Leave empty for pure audio E2E.",
                    )
                    max_tokens = gr.Number(label="Max Tokens", value=128, precision=0)
                    temperature = gr.Number(label="Temperature", value=0.0)
                    timeout = gr.Number(label="Timeout Seconds", value=120)

                submit = gr.Button("Send Audio", variant="primary")

                result_summary = gr.Markdown(label="Summary")
                parsed_output = gr.Code(label="Parsed Output", language="json")
                raw_response = gr.Code(label="Raw Response", language="json")

                submit.click(
                    infer_audio,
                    inputs=[audio, server, model, hint_text, max_tokens, temperature, timeout],
                    outputs=[result_summary, parsed_output, raw_response],
                )
                audio.change(
                    analyze_audio_for_ui,
                    inputs=[audio],
                    outputs=[record_status],
                )
                backend_record.click(
                    record_backend,
                    inputs=[duration, sample_rate],
                    outputs=[audio, record_status],
                )
                backend_record_send.click(
                    record_backend_and_infer,
                    inputs=[duration, sample_rate, server, model, hint_text, max_tokens, temperature, timeout],
                    outputs=[audio, record_status, result_summary, parsed_output, raw_response],
                )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for remote Qwen-Omni audio inference.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL, e.g. http://10.95.64.153:8000")
    parser.add_argument("--model", default="qwen2.5-omni")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.server, args.model)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
