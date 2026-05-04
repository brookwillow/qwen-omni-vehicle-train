#!/usr/bin/env python3
"""Record local audio and send it to a remote Qwen-Omni inference server.

Examples:
    python scripts/record_remote_infer.py --server http://10.95.64.153:8000 --duration 4
    python scripts/record_remote_infer.py --server http://10.95.64.153:8000 --audio sample.wav
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


DEFAULT_SERVER = os.environ.get("QWEN_OMNI_SERVER_URL", "http://10.95.64.153:8000")
MIN_AUDIO_BYTES = 1024
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


def validate_wav_has_audio(path: Path) -> dict[str, float | int]:
    if path.stat().st_size < MIN_AUDIO_BYTES:
        raise RuntimeError(f"audio file is too small: {path.stat().st_size} bytes")

    stats = analyze_wav(path)
    if stats.get("rms", 0) < MIN_AUDIO_RMS or stats.get("peak_abs", 0) == 0:
        raise RuntimeError(
            "audio appears to be silent: "
            f"rms={stats.get('rms')} peak_abs={stats.get('peak_abs')} duration={stats.get('duration_sec')}s"
        )
    return stats


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"recorder not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"recorder failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def _record_with_sounddevice(path: Path, duration: float, sample_rate: int) -> bool:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        return False

    print(f"[record] recording {duration:.1f}s via sounddevice -> {path}")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return True


def record_wav(path: Path, duration: float, sample_rate: int) -> None:
    if _record_with_sounddevice(path, duration, sample_rate):
        return

    system = platform.system().lower()
    if system == "darwin" and shutil.which("afrecord"):
        print(f"[record] recording {duration:.1f}s via afrecord -> {path}")
        _run(["afrecord", "-f", "WAVE", "-d", str(duration), "-r", str(sample_rate), "-c", "1", str(path)])
        return

    if shutil.which("arecord"):
        print(f"[record] recording {duration:.1f}s via arecord -> {path}")
        _run(["arecord", "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-d", str(int(duration)), str(path)])
        return

    if shutil.which("rec"):
        print(f"[record] recording {duration:.1f}s via sox rec -> {path}")
        _run(["rec", "-q", "-r", str(sample_rate), "-c", "1", "-b", "16", str(path), "trim", "0", str(duration)])
        return

    raise RuntimeError(
        "no local recorder available. Install sounddevice+soundfile, or use macOS afrecord, Linux arecord, "
        "or pass an existing file with --audio."
    )


def build_payload(audio_path: Path, model: str, max_tokens: int, temperature: float, hint_text: str) -> dict[str, Any]:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
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


def print_response(resp: dict[str, Any]) -> None:
    print("\n[response]")
    print(json.dumps(resp, ensure_ascii=False, indent=2))

    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        fn = call.get("function") or {}
        print("\n[tool_call]")
        print(f"name: {fn.get('name', '')}")
        args = fn.get("arguments", "")
        try:
            print("arguments:")
            print(json.dumps(json.loads(args), ensure_ascii=False, indent=2))
        except (TypeError, json.JSONDecodeError):
            print(f"arguments: {args}")
    else:
        print("\n[content]")
        print(message.get("content", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record local audio and send it to a remote Qwen-Omni server.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL, e.g. http://10.95.64.153:8000")
    parser.add_argument("--model", default="qwen2.5-omni")
    parser.add_argument("--duration", type=float, default=4.0, help="Recording duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--audio", default="", help="Use an existing WAV file instead of recording.")
    parser.add_argument("--keep-audio", default="", help="Path to keep the recorded WAV for debugging.")
    parser.add_argument("--hint-text", default="", help="Optional text hint sent with the audio; leave empty for pure audio E2E.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = args.server.rstrip("/")
    url = f"{server}/v1/chat/completions"

    temp_path: Path | None = None
    try:
        if args.audio:
            audio_path = Path(args.audio).expanduser().resolve()
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
        else:
            if args.keep_audio:
                audio_path = Path(args.keep_audio).expanduser().resolve()
                audio_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                fd, name = tempfile.mkstemp(prefix="qwen_omni_record_", suffix=".wav")
                os.close(fd)
                temp_path = Path(name)
                audio_path = temp_path
            record_wav(audio_path, args.duration, args.sample_rate)

        print(f"[request] POST {url}")
        stats = validate_wav_has_audio(audio_path)
        print(f"[audio] {audio_path} ({audio_path.stat().st_size} bytes)")
        print(f"[audio_stats] {json.dumps(stats, ensure_ascii=False)}")
        payload = build_payload(audio_path, args.model, args.max_tokens, args.temperature, args.hint_text)
        resp = post_json(url, payload, args.timeout)
        print_response(resp)
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
