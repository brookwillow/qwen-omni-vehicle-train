# 车载语音助手 LoRA 训练方案

> 更新时间：2026-05-02

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
pip install fastapi uvicorn pydantic   # serve.py 推理服务依赖
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

### 数据分布（当前）

| 类型 | 数量 | 说明 |
|------|------|------|
| Action | 3880 | 2 轮：Action → FinalAnswer；已补充工具混淆对比和参数精确样本 |
| Clarify | 177 | 2 轮：Clarify → FinalAnswer；仅保留缺少必需信息或目标不明确的追问 |
| Reject | 1208 | 单轮 + 多轮硬负例（已合并） |

## 训练配置

```bash
python train_thinker_lora.py \
  --model models/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output
```

### RTX 3090 (24GB) 显存控制

训练脚本所有超参均可通过命令行覆盖。以下是针对 3090 的推荐配置：

**标准配置（稳定，显存约 20GB）**
```bash
python train_thinker_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output \
  --torch-dtype bfloat16 \
  --max-length 8192 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 3
```

**激进省显存配置（显存约 16GB，适合 OOM 时）**
```bash
python train_thinker_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output \
  --torch-dtype bfloat16 \
  --max-length 8192 \
  --train-batch-size 1 \
  --grad-accum 16 \
  --lora-r 4 \
  --lora-alpha 8 \
  --epochs 3
```

> ⚠️ `--max-length` 不能低于 7168。系统提示约 5K tokens，加对话内容约需 7000+。设得过低会直接截断系统提示，导致训练数据损坏。

**关键参数说明**

| 参数 | 省显存方向 | 说明 |
|------|-----------|------|
| `--max-length` | **不可低于 7168** | 系统提示 ~5K tokens，对话 ~1-2K，最低 7168 |
| `--train-batch-size` | 保持 1 | 已是最小值，降不了 |
| `--grad-accum` | ↑ 增大 | 等效 batch 不变，用时间换显存 |
| `--lora-r` | ↓ 降低 | r=4 可节省约 10% 显存，精度略降 |
| `--torch-dtype bfloat16` | 保持 | 3090 原生支持 bf16，不要用 fp32 |
| `gradient_checkpointing` | 已内置 True | 以 30% 速度换显存，不需要手动开 |

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

### OpenAI 兼容推理服务（serve.py）

```bash
# 启动服务（加载 LoRA）
pip install fastapi uvicorn pydantic   # 首次运行需安装
python serve.py \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --host 0.0.0.0 \
  --port 8000
```

服务启动后监听 `http://<ip>:8000`，兼容 OpenAI Chat Completions API：

**文本请求**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-omni",
    "messages": [{"role": "user", "content": "帮我打开主驾车窗"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

**音频请求（base64 编码 WAV）**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-omni",
    "messages": [{
      "role": "user",
      "content": [{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}]
    }],
    "max_tokens": 128,
    "temperature": 0
  }'
```

**Python 客户端（openai SDK）**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="qwen2.5-omni",
    messages=[{"role": "user", "content": "帮我打开主驾车窗"}],
    max_tokens=128,
    temperature=0,
)
print(resp.choices[0].message.content)
```

API 端点：
- `POST /v1/chat/completions` – 推理
- `GET  /v1/models` – 列出模型
- `GET  /health` – 健康检查

### 交互式 CLI 推理

```bash
python infer_cli_omni.py \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output
```

### 评测

```bash
# 批量评测（data/eval/ 下所有场景，自动使用音频输入）
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output

# 指定报告输出路径
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --report eval_report.json

# 提高 GPU 利用率（默认 batch_size=1，推荐 4-8，OOM 则减小）
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --batch-size 4 \
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
- 解析后的工具参数会经过 `tool_postprocess.py` 做确定性修正，例如泛化座椅指令不会默认补 `position=主驾`，并修正部分座椅/车窗参数格式偏差
- `position` 为可选参数；用户未明确位置时不因缺少位置追问，直接省略 `position`，由工具侧按说话人位置补全

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
| `serve.py` | **OpenAI 兼容推理服务**（FastAPI，支持文本+音频） |
| `infer_cli_omni.py` | 交互式 CLI 推理 |
| `eval.py` | 统一评测（batch / single），音频输入 + 多维度统计，支持 `--batch-size` 批量推理 |
| `tool_postprocess.py` | 工具调用参数后处理，修正确定性的模型输出偏差 |
| `scripts/probe_asr_decoder.py` | Qwen 音频编码器 → Whisper 解码器 ASR 探测实验 |
| `scripts/build_r5_augment.py` | R5 数据增强（position/anti-clarify/climate/light） |

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
- [x] R4 数据增强（+358 条：修复过度-Clarify、补齐弱工具媒体/电话/信息）
- [x] eval.py `--batch-size` 批量推理（单 GPU 利用率从 ~30% → ~75%）
- [x] R5 数据增强（+164 条：position 字段覆盖、抗过度-Clarify、Climate/Light 多样性）
- [x] R6 anti-Clarify 清理：位置缺失不追问，口语意图/信息查询/电话与 FM 搜索直接 Action

## 下一步

- [ ] 补充 Clarify 评测数据（当前 0 条，训练集有 177 条）
- [ ] 补充 Reject 评测数据（当前 1 条，训练集有 ~1030 条）
- [ ] 补齐 9 个无覆盖工具的测试数据（GeneralBack/Exit/Select、NavigationControl 等）
- [ ] 工具混淆问题（雨刮→ClimateControl、播放→MediaControl vs MusicSearchPlay）
- [ ] 阶段 B：DPO/ORPO 定向提准
- [ ] 导出部署：合并 LoRA → ONNX/GGUF
