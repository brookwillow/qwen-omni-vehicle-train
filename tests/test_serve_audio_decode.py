import base64
import builtins
import io
import sys
import types
import wave

import torch

sys.modules.setdefault("peft", types.SimpleNamespace(PeftModel=object))
sys.modules.setdefault("qwen_omni_utils", types.SimpleNamespace(process_mm_info=lambda *args, **kwargs: ([], [], [])))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(Qwen2_5OmniForConditionalGeneration=object, Qwen2_5OmniProcessor=object),
)
import serve
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


def test_load_model_falls_back_to_sdpa_when_flash_attn_unavailable(monkeypatch):
    captured = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.update(kwargs)
            return cls()

        def eval(self):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    real_import = builtins.__import__

    def import_without_flash_attn(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("flash_attn unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flash_attn)
    monkeypatch.setattr(serve, "Qwen2_5OmniForConditionalGeneration", FakeModel)
    monkeypatch.setattr(serve, "Qwen2_5OmniProcessor", FakeProcessor)

    serve.load_model("model-dir", "")

    assert captured["attn_implementation"] == "sdpa"


def test_perf_averages_record_running_stage_means():
    averages = serve._PerfAverages()

    count, first = averages.record({"convert": 10.0, "total": 30.0})
    assert count == 1
    assert first == {"convert": 10.0, "total": 30.0}

    count, second = averages.record({"convert": 20.0, "total": 50.0})
    assert count == 2
    assert second == {"convert": 15.0, "total": 40.0}


def test_kv_prompt_cache_matches_system_prompt_prefix():
    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 3
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    hit = cache.match("<system>系统提示</system>\n<user>打开车窗</user>", "系统提示")
    miss = cache.match("<system>其他</system>\n<user>打开车窗</user>", "其他")

    assert hit is not None
    assert hit.suffix_text == "<user>打开车窗</user>"
    assert hit.prefix_tokens == 3
    assert miss is None
    assert cache.last_miss_reason.startswith("system_prompt_mismatch")


def test_kv_prompt_cache_records_prefix_mismatch_reason():
    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 3
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    assert cache.match("<different>系统提示</different>\n<user>打开车窗</user>", "系统提示") is None
    assert cache.last_miss_reason.startswith("prefix_text_mismatch")


def test_clone_past_key_values_does_not_share_tensor_storage():
    past = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    cloned = serve._clone_past_key_values(past)
    cloned[0][0].add_(10)

    assert past[0][0].item() == 1.0
    assert cloned[0][0].item() == 11.0


def test_get_thinker_model_unwraps_peft_base_model():
    thinker = object()
    wrapped = types.SimpleNamespace(base_model=types.SimpleNamespace(model=types.SimpleNamespace(thinker=thinker)))

    assert serve._get_thinker_model(wrapped) is thinker


def test_kv_prompt_cache_prepare_uses_thinker_forward():
    captured = {}

    class FakeInputs(dict):
        def to(self, *args, **kwargs):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "<system>系统提示</system>\n"

        def __call__(self, **kwargs):
            return FakeInputs({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])})

    class FakeThinker:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                past_key_values=((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)
            )

    model = types.SimpleNamespace(device=torch.device("cpu"), dtype=None, thinker=FakeThinker())
    cache = serve._KvPromptCache()

    assert cache.prepare(model, FakeProcessor(), "系统提示")
    assert captured["use_cache"] is True
    assert captured["return_dict"] is True
    assert cache.prefix_tokens == 2


def test_run_inference_uses_kv_prompt_cache_on_prefix_hit(monkeypatch):
    captured = {}

    class FakeInputs(dict):
        def to(self, *args, **kwargs):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "<system>系统提示</system>\n<user>打开车窗</user><assistant>"

        def __call__(self, **kwargs):
            captured["processor_text"] = kwargs["text"]
            return FakeInputs(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            )

        def decode(self, ids, **kwargs):
            return "Action: WindowControl"

    class FakeModel:
        device = torch.device("cpu")
        dtype = None

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[10, 11, 12, 99]])

    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 2
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    messages = [{"role": "system", "content": [{"type": "text", "text": "系统提示"}]}]

    reply, prompt_tokens, gen_tokens = serve.run_inference(
        FakeModel(),
        FakeProcessor(),
        messages,
        max_new_tokens=1,
        temperature=0,
        prompt_cache=cache,
    )

    assert captured["processor_text"] == "<user>打开车窗</user><assistant>"
    assert "past_key_values" in captured["generate_kwargs"]
    assert captured["generate_kwargs"]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert reply == "Action: WindowControl"
    assert prompt_tokens == 5
    assert gen_tokens == 1


def test_chat_response_uses_server_model_name_by_default():
    response = serve.build_chat_response(
        choice=serve.Choice(message=serve.AssistantMessage(content="ok")),
        prompt_tokens=10,
        gen_tokens=2,
    )

    assert response.model == "qwen-omni-lora"


def test_noise_do_not_act_is_suppressed_from_client_tool_calls(capsys):
    parsed = serve.parse_model_output("Action: NoiseDoNotAct\nAction Input: {}", "呲啦呲啦")
    choice = serve._choice_from_parsed_output(parsed)

    response = serve.build_chat_response(choice=choice, prompt_tokens=10, gen_tokens=4)

    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == ""
    assert response.choices[0].message.tool_calls is None
    assert "[NOISE_DO_NOT_ACT]" in capsys.readouterr().err
