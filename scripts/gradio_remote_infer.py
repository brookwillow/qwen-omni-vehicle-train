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
        payload = build_payload(path, model, int(max_tokens), float(temperature), hint_text.strip())
        resp = post_json(url, payload, float(timeout))
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
        return f"{status}: {path}\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
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


def build_app(default_server: str, default_model: str):
    import gradio as gr

    with gr.Blocks(title="Qwen-Omni Remote Audio Test") as demo:
        gr.Markdown("# Qwen-Omni Remote Audio Test")
        gr.Markdown("Record or upload local audio, then send it to the remote `serve.py` endpoint.")

        with gr.Row():
            server = gr.Textbox(label="Server", value=default_server)
            model = gr.Textbox(label="Model", value=default_model)

        audio = gr.Audio(
            label="Browser Audio",
            sources=["microphone", "upload"],
            type="filepath",
            format="mp3",
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
