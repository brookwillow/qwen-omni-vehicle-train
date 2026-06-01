import base64
import builtins
import hashlib
import io
import json
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
    cache.prefix_input_ids = torch.tensor([[1, 2, 3]])
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    full_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5)}

    hit = cache.match("<system>系统提示</system>\n<user>打开车窗</user>", "系统提示", full_inputs)
    miss = cache.match("<system>其他</system>\n<user>打开车窗</user>", "其他", full_inputs)

    assert hit is not None
    assert hit.suffix_text == "<user>打开车窗</user>"
    assert hit.suffix_input_ids.tolist() == [[4, 5]]
    assert hit.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert hit.prefix_tokens == 3
    assert miss is None
    assert cache.last_miss_reason.startswith("system_prompt_mismatch")


def test_strip_historical_tool_messages_drops_tool_chains_before_latest_user():
    messages = [
        serve.Message(role="user", content="先查一下"),
        serve.Message(
            role="assistant",
            content='{"name":"CarUsageSearch","arguments":{"query":"旧查询"}}',
        ),
        serve.Message(role="tool", content='{"status":"success","query":"旧查询"}'),
        serve.Message(role="assistant", content="我查到了，继续说。"),
        serve.Message(role="user", content="再帮我开窗"),
        serve.Message(
            role="assistant",
            content='{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}',
        ),
        serve.Message(role="tool", content='{"status":"success","action":"打开"}'),
        serve.Message(role="assistant", content="已经帮您打开了。"),
        serve.Message(role="user", content="谢谢"),
    ]

    filtered = serve._strip_historical_tool_messages(messages)

    assert [msg.role for msg in filtered] == ["user", "assistant", "user", "assistant", "user"]
    assert filtered[1].tool_calls is None
    assert filtered[3].content == "已经帮您打开了。"


def test_messages_to_qwen_compacts_kept_tool_call_and_tool_result(tmp_path):
    messages = [
        serve.Message(role="user", content="打开车窗"),
        serve.Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "WindowControl",
                        "arguments": '{ "action": "打开", "device": "车窗" }',
                    },
                }
            ],
        ),
        serve.Message(role="tool", content='{ "status": "success", "rawTTS": "车窗调好了" }'),
    ]

    qwen_messages, tmp_files = serve._messages_to_qwen(messages, "系统提示", str(tmp_path))

    assert tmp_files == []
    assert qwen_messages[1]["content"][0]["text"] == "打开车窗"
    assert qwen_messages[2]["content"][0]["text"] == (
        '{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}'
    )
    assert qwen_messages[3]["content"][0]["text"] == '{"status":"success","rawTTS":"车窗调好了"}'


def test_messages_to_qwen_compacts_action_style_assistant_text(tmp_path):
    messages = [
        serve.Message(role="user", content="打开车窗"),
        serve.Message(
            role="assistant",
            content='Action: WindowControl\nAction Input: {"action": "打开", "device": "车窗"}',
        ),
        serve.Message(role="tool", content='{"status":"success"}'),
    ]

    qwen_messages, _ = serve._messages_to_qwen(messages, "系统提示", str(tmp_path))

    assert qwen_messages[2]["content"][0]["text"] == (
        '{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}'
    )


def test_kv_prompt_cache_records_prefix_mismatch_reason():
    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 3
    cache.prefix_input_ids = torch.tensor([[1, 2, 3]])
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)

    full_inputs = {"input_ids": torch.tensor([[9, 2, 3, 4]]), "attention_mask": torch.ones(1, 4)}

    assert cache.match("<different>系统提示</different>\n<user>打开车窗</user>", "系统提示", full_inputs) is None
    assert cache.last_miss_reason.startswith("prefix_text_mismatch")


def test_kv_prompt_cache_rejects_token_prefix_mismatch():
    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 3
    cache.prefix_input_ids = torch.tensor([[1, 2, 3]])
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)
    full_inputs = {"input_ids": torch.tensor([[1, 20, 3, 4]]), "attention_mask": torch.ones(1, 4)}

    assert cache.match("<system>系统提示</system>\n<user>打开车窗</user>", "系统提示", full_inputs) is None
    assert cache.last_miss_reason.startswith("prefix_token_mismatch")


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
                past_key_values=((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),),
                rope_deltas=torch.tensor([[0]]),
            )

    model = types.SimpleNamespace(device=torch.device("cpu"), dtype=None, thinker=FakeThinker())
    cache = serve._KvPromptCache()

    assert cache.prepare(model, FakeProcessor(), "系统提示")
    assert captured["use_cache"] is True
    assert captured["return_dict"] is True
    assert cache.prefix_tokens == 2
    assert cache.prefix_input_ids.tolist() == [[1, 2]]
    assert cache.rope_deltas.tolist() == [[0]]


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
                    "input_ids": torch.tensor([[1, 2, 10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                }
            )

        def decode(self, ids, **kwargs):
            return "Action: WindowControl"

    class FakeModel:
        device = torch.device("cpu")
        dtype = None
        thinker = types.SimpleNamespace(rope_deltas=torch.tensor([[99]]))

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            assert self.thinker.rope_deltas.tolist() == [[0]]
            self.thinker.rope_deltas = torch.tensor([[123]])
            return torch.tensor([[1, 2, 10, 11, 12, 99]])

    cache = serve._KvPromptCache()
    cache.system_prompt = "系统提示"
    cache.prefix_text = "<system>系统提示</system>\n"
    cache.prefix_tokens = 2
    cache.prefix_input_ids = torch.tensor([[1, 2]])
    cache.past_key_values = ((torch.tensor([[[[1.0]]]]), torch.tensor([[[[2.0]]]])),)
    cache.rope_deltas = torch.tensor([[0]])

    messages = [{"role": "system", "content": [{"type": "text", "text": "系统提示"}]}]
    model = FakeModel()

    reply, prompt_tokens, gen_tokens = serve.run_inference(
        model,
        FakeProcessor(),
        messages,
        max_new_tokens=1,
        temperature=0,
        prompt_cache=cache,
    )

    assert captured["processor_text"] == "<system>系统提示</system>\n<user>打开车窗</user><assistant>"
    assert captured["generate_kwargs"]["input_ids"].tolist() == [[1, 2, 10, 11, 12]]
    assert "past_key_values" in captured["generate_kwargs"]
    assert captured["generate_kwargs"]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert "cache_position" not in captured["generate_kwargs"]
    assert captured["generate_kwargs"]["thinker_max_new_tokens"] == 1
    assert "max_new_tokens" not in captured["generate_kwargs"]
    assert reply == "Action: WindowControl"
    assert prompt_tokens == 5
    assert gen_tokens == 1
    assert model.thinker.rope_deltas.tolist() == [[99]]


def test_run_inference_passes_thinker_max_new_tokens_without_cache():
    captured = {}

    class FakeInputs(dict):
        def to(self, *args, **kwargs):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "<system>系统提示</system>\n<user>打开车窗</user><assistant>"

        def __call__(self, **kwargs):
            return FakeInputs({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)})

        def decode(self, ids, **kwargs):
            return "ok"

    class FakeModel:
        device = torch.device("cpu")
        dtype = None

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 3, 4]])

    serve.run_inference(
        FakeModel(),
        FakeProcessor(),
        [{"role": "system", "content": [{"type": "text", "text": "系统提示"}]}],
        max_new_tokens=7,
        temperature=0,
        prompt_cache=None,
    )

    assert captured["thinker_max_new_tokens"] == 7
    assert "max_new_tokens" not in captured


def test_chat_response_uses_server_model_name_by_default():
    response = serve.build_chat_response(
        choice=serve.Choice(message=serve.AssistantMessage(content="ok")),
        prompt_tokens=10,
        gen_tokens=2,
    )

    assert response.model == "qwen-omni-lora"


def test_noise_do_not_act_is_returned_as_client_tool_call():
    parsed = serve.parse_model_output('{"name":"NoiseDoNotAct","arguments":{}}')
    choice = serve._choice_from_parsed_output(parsed)

    response = serve.build_chat_response(choice=choice, prompt_tokens=10, gen_tokens=4)

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.content == ""
    assert response.choices[0].message.tool_calls is not None
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "NoiseDoNotAct"
    assert tool_call.function.arguments == "{}"


def test_reject_is_returned_as_unsupported_boundary():
    parsed = serve.parse_model_output("Reject")
    choice = serve._choice_from_parsed_output(parsed)

    response = serve.build_chat_response(choice=choice, prompt_tokens=10, gen_tokens=1)

    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == ""
    assert response.choices[0].message.supported is False
    assert response.choices[0].message.tool_calls is None
    assert '"supported":false' in response.model_dump_json()


def test_reject_parser_accepts_terminal_punctuation():
    assert serve.parse_model_output("Reject。") == ("reject",)
    assert serve.parse_model_output("reject.") == ("reject",)


def test_parse_model_output_preserves_model_tool_call_arguments():
    raw = '{"name":"SeatControl","arguments":{"action":"关闭","device":"座椅","feature":"通风","position":"主驾"}}'

    parsed = serve.parse_model_output(raw)

    assert parsed == (
        "tool_call",
        "SeatControl",
        '{"action":"关闭","device":"座椅","feature":"通风","position":"主驾"}',
    )


def test_parse_model_output_supports_multi_tool_json_array():
    raw = (
        '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
        '{"name":"LightControl","arguments":{"action":"关闭","device":"阅读灯"}}]'
    )

    parsed = serve.parse_model_output(raw)
    choice = serve._choice_from_parsed_output(parsed)

    assert parsed == (
        "tool_calls",
        [
            ("WindowControl", '{"action":"打开","device":"车窗"}'),
            ("LightControl", '{"action":"关闭","device":"阅读灯"}'),
        ],
    )
    assert choice.finish_reason == "tool_calls"
    assert [call.index for call in choice.message.tool_calls] == [0, 1]
    assert [call.function.name for call in choice.message.tool_calls] == [
        "WindowControl",
        "LightControl",
    ]


def test_save_request_artifacts_writes_model_request(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "_SAVE_DIR", tmp_path)
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(_make_wav(16000, 1, 2))
    response = serve.build_chat_response(
        choice=serve.Choice(message=serve.AssistantMessage(content="ok")),
        prompt_tokens=3,
        gen_tokens=1,
    )
    model_messages = [
        {"role": "system", "content": [{"type": "text", "text": "系统提示"}]},
        {"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]},
    ]

    serve._save_request_artifacts(
        response.id,
        [str(audio_path)],
        response,
        [],
        model_messages,
    )

    request_path = tmp_path / response.id / "model_request.json"
    saved = json.loads(request_path.read_text(encoding="utf-8"))

    assert saved["messages"] == model_messages
    assert saved["audio_files"] == [{"source": str(audio_path), "saved": "audio_0.wav"}]
    assert (tmp_path / response.id / "response.json").exists()
    assert (tmp_path / response.id / "audio_0.wav").exists()


def test_save_request_artifacts_writes_sanitized_raw_request(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "_SAVE_DIR", tmp_path)
    audio_b64 = base64.b64encode(_make_wav(16000, 1, 2)).decode("ascii")
    response = serve.build_chat_response(
        choice=serve.Choice(message=serve.AssistantMessage(content="ok")),
        prompt_tokens=3,
        gen_tokens=1,
    )
    raw_request = serve.ChatRequest(
        model="qwen-omni-lora",
        messages=[
            serve.Message(
                role="user",
                content=[
                    serve.ContentPart(
                        type="input_audio",
                        input_audio={"data": audio_b64, "format": "wav"},
                    )
                ],
            )
        ],
        max_tokens=32,
        temperature=0,
    )

    serve._save_request_artifacts(
        response.id,
        [],
        response,
        raw_request.messages,
        raw_request=raw_request,
    )

    request_path = tmp_path / response.id / "raw_request.json"
    saved = json.loads(request_path.read_text(encoding="utf-8"))
    audio_data = saved["messages"][0]["content"][0]["input_audio"]["data"]

    assert saved["model"] == "qwen-omni-lora"
    assert saved["max_tokens"] == 32
    assert audio_data["omitted"] is True
    assert audio_data["base64_chars"] == len(audio_b64)
    assert audio_data["sha256"] == hashlib.sha256(audio_b64.encode("utf-8")).hexdigest()
    assert audio_b64 not in request_path.read_text(encoding="utf-8")
