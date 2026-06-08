from train_teacher_sft_lora import (
    TeacherSftConfig,
    build_build_data_command,
    build_label_mask,
    normalize_chatml_text,
    parse_args,
    render_chatml_messages,
)


def test_teacher_defaults_target_qwen35_27b():
    args = parse_args([])

    assert args.model == "Qwen/Qwen3.5-27B"
    assert args.train_file == "data/train_final.jsonl"
    assert args.system_prompt == "data/system-prompt.txt"
    assert args.torch_dtype == "bfloat16"
    assert args.max_length == 4096
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.load_in_4bit is False
    assert args.freeze_vision is True


def test_teacher_build_data_command_uses_current_omni_weights():
    command = build_build_data_command(TeacherSftConfig(train_file="data/train_final.jsonl"))
    text = " ".join(command)

    assert "build_train_data.py" in text
    assert "hard_cases/OverNoiseRemaining_20260607.jsonl:4" in text
    assert "hard_cases/*.jsonl:3" in text
    assert "ProfileControl:2" in text
    assert "WindowControl:1.5" in text
    assert "CurrentNoiseWithHistoryTool:0.5" in text
    assert "NoiseDoNotAct_coverage:0.5" in text
    assert "--output data/train_final.jsonl" in text


def test_normalize_chatml_text_makes_generation_markers_consistent():
    text = "<|im_start|>assistant\nReject<|im_end|>\n"

    assert normalize_chatml_text(text) == "<|im_start|>assistant\nReject<|im_end|>"


def test_render_chatml_messages_preserves_tool_role():
    messages = [
        {"role": "system", "content": "SP"},
        {"role": "user", "content": "打开车窗"},
        {"role": "assistant", "content": '{"name":"WindowControl"}'},
        {"role": "tool", "content": '{"ok":true}'},
        {"role": "assistant", "content": "已打开"},
    ]

    rendered = render_chatml_messages(messages)

    assert "<|im_start|>tool\n{\"ok\":true}<|im_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n已打开<|im_end|>")


def test_build_label_mask_supervises_last_assistant_occurrence_only():
    input_ids = [1, 2, 99, 3, 4, 99, 5]
    target_ids = [99]

    labels, matched = build_label_mask(input_ids, target_ids)

    assert matched is True
    assert labels == [-100, -100, -100, -100, -100, 99, -100]
