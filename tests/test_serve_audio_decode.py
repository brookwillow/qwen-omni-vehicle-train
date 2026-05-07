import base64
import io
import sys
import types
import wave

sys.modules.setdefault("peft", types.SimpleNamespace(PeftModel=object))
sys.modules.setdefault("qwen_omni_utils", types.SimpleNamespace(process_mm_info=lambda *args, **kwargs: ([], [], [])))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(Qwen2_5OmniForConditionalGeneration=object, Qwen2_5OmniProcessor=object),
)
from serve import _is_wav_16k_mono_pcm16, _safe_b64decode, _write_audio_tmp


def test_safe_b64decode_strips_data_url_metadata():
    raw = b"\x01\x02\x03\x04pcm"
    data_url = "data:audio/pcm;base64," + base64.b64encode(raw).decode("ascii")

    assert _safe_b64decode(data_url) == raw


def test_safe_b64decode_still_accepts_plain_base64():
    raw = b"\x10\x20\x30\x40"

    assert _safe_b64decode(base64.b64encode(raw).decode("ascii")) == raw


def _make_wav(sample_rate: int, channels: int, sample_width: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00" * (sample_width * channels * sample_rate // 10))
    return buffer.getvalue()


def test_detects_16k_mono_pcm16_wav():
    assert _is_wav_16k_mono_pcm16(_make_wav(16000, 1, 2))


def test_rejects_non_16k_wav_for_fast_path():
    assert not _is_wav_16k_mono_pcm16(_make_wav(44100, 1, 2))


def test_write_audio_tmp_skips_ffmpeg_for_16k_mono_pcm16_wav(tmp_path, monkeypatch):
    raw = _make_wav(16000, 1, 2)

    def fail_if_ffmpeg_runs(*args, **kwargs):
        raise AssertionError("ffmpeg should not run for already-normalized WAV")

    monkeypatch.setattr("serve._run_ffmpeg", fail_if_ffmpeg_runs)

    wav_path = _write_audio_tmp(raw, "wav", str(tmp_path))

    assert wav_path.endswith(".wav")
    assert (tmp_path / wav_path.split("/")[-1]).exists()
