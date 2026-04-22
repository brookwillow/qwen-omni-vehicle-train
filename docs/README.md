# 车载语音助手 LoRA 训练方案

> 更新时间：2026-04-22

## 项目概要

基于 Qwen2.5-Omni-3B 的 Thinker-only LoRA 微调，实现车载 ReAct 风格工具调用。

- 基模型：`Qwen2.5-Omni-3B`（`max_position_embeddings=32768`）
- 训练策略：LoRA (r=8, alpha=16)，仅训练 Thinker（语言路径）
- 冻结模块：AUT + Talker + Vocoder（通过关键词审计强制保证）
- 输出格式：`Action` / `Clarify` / `Reject` 三类决策

## 环境

```bash
conda create -y -n qwen-omni python=3.11
conda activate qwen-omni
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "ms-swift[all]" modelscope peft
pip install "qwen-omni-utils[decord]" soundfile
```

## 数据流水线

```
data/splits/{action,clarify,reject,reject_augmented}.jsonl  (已拆分好的数据，无 SP)
  │
  └─ build_train_data.py ──→ data/train_final.jsonl (注入 SP，打散，可过采样)
                                │
                                └─ train_thinker_lora.py ──→ lora_output/
```

> 注：拆分/增强脚本已归档至 `_archive/`，产物 `data/splits/` 已就绪，日常只需运行 `build_train_data.py`。

### 关键文件

| 文件 | 说明 |
|------|------|
| `data/system-prompt.txt` | 紧凑版 System Prompt（~12K chars，~5K tokens） |
| `data/tools.json` | 20 个车载工具定义 |
| `data/splits/` | 按类型拆分的训练数据（无 SP） |
| `data/train_final.jsonl` | 最终训练数据（含 SP） |
| `data/eval/` | 评测数据集（18 个场景 + 音频） |

### 数据分布（v3 基线）

| 类型 | 数量 | 说明 |
|------|------|------|
| Action | ~2399 | 2 轮：Action → FinalAnswer |
| Clarify | ~605 | 2 轮：Clarify → FinalAnswer |
| Reject | ~1030 | 单轮 + 多轮硬负例（已合并） |

## 训练配置

```bash
python train_thinker_lora.py \
  --model models/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output
```

### 默认超参

| 参数 | 值 | 说明 |
|------|-----|------|
| max_length | 16384 | SP ~5K tokens + 对话，留足余量 |
| lr | 2e-5 | 3B 小模型适用 |
| lora_r / alpha | 8 / 16 | effective scaling = 2 |
| batch_size | 1 | RTX 3090 24GB 显存限制 |
| grad_accum | 8 | 等效 batch=8 |
| warmup_ratio | 0.05 | 前 5% steps 线性预热 |
| weight_decay | 0.01 | AdamW 正则化 |
| max_grad_norm | 1.0 | 梯度裁剪防止 loss spike |
| epochs | 3 | 配合 load_best_model_at_end |
| gradient_checkpointing | True | 节省显存 |
| metric_for_best_model | eval_token_acc | 自动选最优 checkpoint |

### 冻结保障

训练脚本通过关键词 `audio,talker,vocoder,audio_decoder,speech_decoder` 自动冻结非 Thinker 参数，并输出审计文件：
- `lora_output/trainable_params.txt`
- `lora_output/freeze_summary.json`

若冻结后仍有禁止参数可训练，脚本会直接报错退出。

## 推理 / 评测

```bash
# 交互式 CLI 推理
python infer_cli_omni.py \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output

# 批量评测（data/eval/ 下所有场景，自动使用音频输入）
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output

# 指定报告输出路径
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --report eval_report.json

# 单条测试（文本）
python eval.py single \
  --model-dir models/Qwen2.5-Omni-3B \
  --prompt "打开主驾车窗"

# 单条测试（音频）
python eval.py single \
  --model-dir models/Qwen2.5-Omni-3B \
  --prompt "打开主驾车窗" \
  --audio data/eval/audio/window/window_001.wav
```

### 评测指标

| 指标 | 说明 |
|------|------|
| `type_acc` | 响应类型准确率（Action/Clarify/Reject 是否选对） |
| `tool_acc` | 工具名称准确率（Action 类型下工具名匹配） |
| `args_em` | 参数精确匹配率（工具名 + 所有参数完全一致） |
| `reject_hit` | Reject 命中数（正确拒绝 / 预测拒绝） |
| `clarify_hit` | Clarify 命中数（正确追问 / 预测追问） |
| `parse_fail` | 输出格式解析失败数 |

### 评测维度

- **Per-file**：每个测试文件（18 个场景）独立统计
- **By Difficulty**：按 easy / medium / hard 分层
- **By Category**：按 category 分组，展示最弱的 10 个

### 评测报告

Batch 模式运行后自动输出 JSON 报告（默认 `eval_report_<timestamp>.json`），包含：
- 时间戳、模型路径、LoRA 路径
- 总体指标 + per-file / per-difficulty / per-category 明细
- 所有错误样本（含 query、gt、pred、err_type）

### 评测数据

- 路径：`data/eval/*_test.json`（18 个文件，1109 条样本）
- 音频：`data/eval/audio/`（1108 条有对应 wav 文件）
- 输入方式：有音频文件时自动用音频输入，无音频时回退到文本
- 支持字段：`expected_type`（显式指定 Action/Clarify/Reject）

## 脚本总览（4 个）

| 脚本 | 用途 |
|------|------|
| `build_train_data.py` | 合并 splits + 注入 SP → 训练集 |
| `train_thinker_lora.py` | LoRA 训练（389 行） |
| `infer_cli_omni.py` | 交互式 CLI 推理 |
| `eval.py` | 统一评测（batch / single），音频输入 + 多维度统计 |

已归档至 `_archive/`：`split_data_by_type.py`、`augment_reject_samples.py`、`build_system_prompt.py`

## 已完成的优化

以下问题在历史迭代中已修复：

- [x] max_length 1024 → 16384（防止样本截断）
- [x] assistant-only loss masking（仅在 assistant 回复上计算 loss）
- [x] SP 压缩 53%（26K → 12K chars）
- [x] SP 统一管理（`data/system-prompt.txt`，训练/推理/评测共用）
- [x] 训练数据不再内嵌 SP，由 build_train_data.py 构建时注入
- [x] lr 1e-4 → 2e-5，alpha 32 → 16，添加 warmup/weight_decay/grad_clip
- [x] load_best_model_at_end，按 eval_token_acc 选最优 checkpoint
- [x] gradient_checkpointing，batch=1 + grad_accum=8（24GB 显存适配）
- [x] Reject 数据增强（103 条硬负例：家电混淆、多轮拒绝、跨域请求）
- [x] 分类逻辑按最后一条 assistant turn 判断（多轮样本正确分类）
- [x] 冻结审计自动化（forbidden keyword → auto-freeze → fail-fast）

## 下一步

- [ ] 补充 Clarify 评测数据（当前 0 条，训练集有 605 条）
- [ ] 补充 Reject 评测数据（当前 1 条，训练集有 ~1030 条）
- [ ] 补齐 9 个无覆盖工具的测试数据（GeneralBack/Exit/Select、NavigationControl 等）
- [ ] 补充 Clarify 训练数据（605 → 目标 ~1500）
- [ ] 工具混淆问题（雨刮→ClimateControl、播放→MediaControl vs MusicSearchPlay）
- [ ] 阶段 B：DPO/ORPO 定向提准
- [ ] 导出部署：合并 LoRA → ONNX/GGUF
