from scripts.run_training_pipeline import PipelineConfig, build_stage_commands, select_stages


def _command_text(stage: str) -> str:
    cfg = PipelineConfig(
        model_dir="/models/qwen",
        train_file="data/train_final.jsonl",
        sft_output_dir="out/sft",
        dpo_noise_output_dir="out/dpo-noise",
    )
    return "\n".join(" ".join(cmd.argv) for cmd in build_stage_commands(cfg, [stage]))


def test_select_stages_full_pipeline_order():
    assert select_stages("full") == ["validate", "build", "sft", "eval-sft", "dpo-noise", "eval-dpo"]


def test_select_stages_allows_single_stage():
    assert select_stages("dpo-noise") == ["dpo-noise"]


def test_build_stage_uses_latest_over_noise_weighting():
    text = _command_text("build")

    assert "build_train_data.py" in text
    assert "hard_cases/OverNoiseRemaining_20260607.jsonl:4" in text
    assert "CurrentNoiseWithHistoryTool:0.5" in text
    assert "NoiseDoNotAct_coverage:0.5" in text


def test_dpo_noise_starts_from_sft_output_and_writes_dpo_output():
    text = _command_text("dpo-noise")

    assert "train_memory_dpo_lora.py" in text
    assert "--init-lora-dir out/sft" in text
    assert "--output-dir out/dpo-noise" in text
    assert "anti_over_noise_preferences.jsonl:4" in text
    assert "still_over_noise_preferences_round3.jsonl:6" in text
    assert "reference_free" in text


def test_eval_dpo_uses_dpo_lora_dir():
    text = _command_text("eval-dpo")

    assert "eval.py batch" in text
    assert "--lora-dir out/dpo-noise" in text
