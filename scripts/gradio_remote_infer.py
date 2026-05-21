#!/usr/bin/env python3
"""Gradio UI for local audio capture and remote Qwen-Omni inference."""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


DEFAULT_SERVER = os.environ.get("QWEN_OMNI_SERVER_URL", "http://10.95.64.153:8000")
MIN_AUDIO_RMS = 5


def analyze_wav(path: Path) -> dict[str, float | int]:
    """Return simple PCM WAV stats used to catch empty/silent recordings."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    stats: dict[str, float | int] = {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_sec": round(frames / sample_rate, 3) if sample_rate else 0,
        "size_bytes": path.stat().st_size,
        "rms": 0,
        "peak_abs": 0,
    }
    if not raw or sample_width != 2:
        return stats

    sample_count = len(raw) // 2
    if sample_count <= 0:
        return stats

    sum_sq = 0
    peak = 0
    for i in range(0, len(raw), 2):
        sample = int.from_bytes(raw[i:i + 2], byteorder="little", signed=True)
        abs_sample = abs(sample)
        peak = max(peak, abs_sample)
        sum_sq += sample * sample
    stats["rms"] = round((sum_sq / sample_count) ** 0.5, 2)
    stats["peak_abs"] = peak
    return stats


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"recorder not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"recorder failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def _record_with_ffmpeg_avfoundation(path: Path, duration: float, sample_rate: int) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-i", ":0",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-t", str(duration),
        "-f", "wav",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 15)
        return result.returncode == 0 and path.exists() and path.stat().st_size > 1024
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _record_with_sounddevice(path: Path, duration: float, sample_rate: int) -> bool:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        return False

    devices = sd.query_devices()
    input_devices = [
        idx for idx, dev in enumerate(devices)
        if int(dev.get("max_input_channels", 0)) > 0
    ]
    if not input_devices:
        return False

    default_input = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    device = default_input if isinstance(default_input, int) and default_input >= 0 else input_devices[0]

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16", device=device)
    sd.wait()
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return True


def record_wav(path: Path, duration: float, sample_rate: int) -> None:
    system = platform.system().lower()
    if system == "darwin":
        if shutil.which("afrecord"):
            _run(["afrecord", "-f", "WAVE", "-d", str(duration), "-r", str(sample_rate), "-c", "1", str(path)])
            return
        if _record_with_ffmpeg_avfoundation(path, duration, sample_rate):
            return

    if _record_with_sounddevice(path, duration, sample_rate):
        return

    if shutil.which("arecord"):
        _run(["arecord", "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-d", str(int(duration)), str(path)])
        return

    if shutil.which("rec"):
        _run(["rec", "-q", "-r", str(sample_rate), "-c", "1", "-b", "16", str(path), "trim", "0", str(duration)])
        return

    raise RuntimeError(
        "no local recorder available. Install sounddevice+soundfile, or use macOS afrecord, Linux arecord."
    )


def build_payload(
    audio_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    hint_text: str,
    payload_audio_path: Path | None = None,
) -> dict[str, Any]:
    encoded_path = payload_audio_path or audio_path
    audio_b64 = "data:audio/wav;base64," + base64.b64encode(encoded_path.read_bytes()).decode("ascii")
    content: list[dict[str, Any]] = []
    if hint_text:
        content.append({"type": "text", "text": hint_text})
    content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    return json.loads(body)


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
        {"role": "assistant", "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}'},
        {"role": "tool", "content": "{\"status\":\"success\"}"},
        {"role": "assistant", "content": "好的，空调已为您打开。"},
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
