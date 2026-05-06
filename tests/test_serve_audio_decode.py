import base64
import sys
import types

sys.modules.setdefault("peft", types.SimpleNamespace(PeftModel=object))
sys.modules.setdefault("qwen_omni_utils", types.SimpleNamespace(process_mm_info=lambda *args, **kwargs: ([], [], [])))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(Qwen2_5OmniForConditionalGeneration=object, Qwen2_5OmniProcessor=object),
)
from serve import _safe_b64decode


def test_safe_b64decode_strips_data_url_metadata():
    raw = b"\x01\x02\x03\x04pcm"
    data_url = "data:audio/pcm;base64," + base64.b64encode(raw).decode("ascii")

    assert _safe_b64decode(data_url) == raw


def test_safe_b64decode_still_accepts_plain_base64():
    raw = b"\x10\x20\x30\x40"

    assert _safe_b64decode(base64.b64encode(raw).decode("ascii")) == raw
