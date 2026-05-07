import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from record_remote_infer import build_payload


def test_build_payload_uses_wav_data_url(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFtest")

    payload = build_payload(audio, "qwen2.5-omni", 128, 0.0, "")
    audio_data = payload["messages"][0]["content"][0]["input_audio"]["data"]

    assert audio_data == "data:audio/wav;base64," + base64.b64encode(b"RIFFtest").decode("ascii")


def test_build_payload_can_normalize_audio_before_encoding(tmp_path):
    source = tmp_path / "source.wav"
    normalized = tmp_path / "normalized.wav"
    source.write_bytes(b"source")
    normalized.write_bytes(b"normalized")

    payload = build_payload(source, "qwen2.5-omni", 128, 0.0, "", payload_audio_path=normalized)
    audio_data = payload["messages"][0]["content"][0]["input_audio"]["data"]

    assert audio_data == "data:audio/wav;base64," + base64.b64encode(b"normalized").decode("ascii")
