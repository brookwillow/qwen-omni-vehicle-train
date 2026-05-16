# 车载语音助手 LoRA 训练方案

> 更新时间：2026-05-16

## 项目概要

基于 Qwen2.5-Omni-3B 的 Thinker-only LoRA 微调，实现车载 ReAct 风格工具调用。

- 基模型：`Qwen2.5-Omni-3B`（`max_position_embeddings=32768`）
- 训练策略：LoRA (r=8, alpha=16)，仅训练 Thinker（语言路径）
- 冻结模块：AUT + Talker + Vocoder（通过关键词审计强制保证）
- 输出格式：紧凑 JSON 工具调用 / 自然语言 TTS 文本 / `Reject` 三类决策

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
data/splits/**/*.jsonl  (已拆分好的数据，无 SP，包含 by_tool 工具文件)
  │
  └─ build_train_data.py ──→ data/train_final.jsonl (注入 SP，打散，可按错误族加权)
                                │
                                └─ train_thinker_lora.py ──→ lora_output/
```

> 注：拆分/增强脚本已归档至 `_archive/`，产物 `data/splits/` 已就绪；`build_train_data.py` 默认递归合并 `data/splits/**/*.jsonl`。
>
> `build_train_data.py` 默认开启 `--validate-schema`，合并后会对每条样本中的工具调用做 schema 校验（required 字段、enum 值、未知参数），不合格样本会被移除并在 stderr 打印详情。可用 `--no-validate-schema` 跳过校验。

### 关键文件

| 文件 | 说明 |
|------|------|
| `data/system-prompt.txt` | 紧凑版 System Prompt（~5.5K chars，基于当前工具白名单生成） |
| `data/tools.json` | 38 个车载工具定义（新版 `inputSchema` 格式） |
| `data/splits/by_tool/*.jsonl` | 按工具拆分的训练数据；每个工具文件内已拆为 `user -> JSON tool call` 决策样本和独立 `JSON tool call -> tool-role JSON result -> TTS text` 回复样本；`PhoneControl.jsonl` 包含小鹏客服、小鹏救援、儿童手表等官方默认联系人样本；其中 `NoiseDoNotAct.jsonl` 当前为 450 条 |
| `data/splits/clarify.jsonl` | required 字段缺失后的自然语言追问样本 |
| `data/splits/edge_case.jsonl` | 多轮上下文边界、易混淆与任务列表选择样本 |
| `data/splits/multiturn.jsonl` | 最多三轮纯文本历史上下文样本；历史可来自导航/音乐/新闻/百科/AIGC/天气等外部域，当前轮优先，只有代词、省略、纠错、延续或查询缺槽时才参考历史补全 |
| `data/splits/reject.jsonl` | 单轮 + 多轮硬负例 |
| `data/train_final.jsonl` | 最终训练数据（含 SP） |
| `data/eval/` | 评测数据集（当前工具 schema 已清洗，媒体类无新版等价工具样本已置空/移除） |

### 数据分布（当前）

| 类型 | 数量 | 说明 |
|------|------|------|
| By-tool | 7306 | 每个工具文件内混合：`user -> JSON tool call` 决策样本 4979 条，独立 `JSON tool call -> tool-role JSON result -> TTS text` 回复样本 2327 条 |
| Clarify | 198 | 已拆为两类样本：`user -> 追问 TTS`（99 条，教模型何时追问）+ 完整 4 轮 `user -> 追问 -> 用户补齐 -> tool call`（99 条，教模型追问后如何响应）；已移除纯位置追问（音区可自动判定位置）；配合 last-user-anchored 标签监督，两类样本均能被正确监督 |
| Edge case | 100 | 多轮当前轮边界、查询 vs 控制、popup/task 列表 `GeneralSelect` 等易混淆样本 |
| Multiturn | 367 | 最多三轮纯文本历史；历史允许外部域文本；当前轮输出分布：工具 240 条、NoiseDoNotAct 79 条、Reject 36 条、自然语言 TTS 12 条 |
| Reject | 1127 | 单轮 + 多轮硬负例（已合并），最后一条 assistant 均为 `Reject`；已抽稀家居控制负例并移除高风险车控状态拒识 |

`build_train_data.py` 默认递归合并全部 split，当前最终训练集为 9098 条；统计时多轮样本按最后一个有效决策标签计入对应类别。

### Hard Case 加权

`--oversample` 仍按文件 stem 加权；新增 `--sample-weight` 支持按 stem、路径后缀或 glob 给具体 split 文件加权，适合将评测错误族沉淀为独立 hard case 文件后提高采样占比。

```bash
python build_train_data.py \
  --sample-weight 'hard_cases/*.jsonl:3' ProfileControl:2 WindowControl:1.5
```

推荐流程：先用 `scripts/analyze_eval_errors.py --backlog-md` 生成错误族补强清单，再为 P0/P1 错误族补充非评估原句 hard case；训练时对这些 hard case 文件做 2-4 倍采样，避免新增样本在 9K 级训练集中被稀释。

## 训练配置

```bash
python train_thinker_lora.py \
  --model models/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output
```

### RTX 3090 (24GB) 显存控制

训练脚本所有超参均可通过命令行覆盖。以下是针对 3090 的推荐配置：

**batch=2 试验配置（当前 SP 压缩后推荐先试）**
```bash
python train_thinker_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output \
  --torch-dtype bfloat16 \
  --max-length 4096 \
  --train-batch-size 2 \
  --grad-accum 4 \
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

> 当前 SP 约 3032 tokens；按 Qwen chat template 估算，当前训练集最长样本约 3246 tokens。建议 `--max-length 4096` 起步；如继续压到 3584 需要先重新统计长度，避免截断新增长样本。

**关键参数说明**

| 参数 | 省显存方向 | 说明 |
|------|-----------|------|
| `--max-length` | ↓ 降低 | 当前建议 4096；新增长样本后需重新统计 |
| `--train-batch-size` | ↑ 尝试 2 | SP 压缩后可尝试 batch=2；如 OOM 回退 1 |
| `--grad-accum` | 配合 batch 调整 | batch=2 时用 4 可保持等效 batch=8 |
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

训练脚本会在编码后定位**最后一个 user 之后的所有** assistant 回复，对这些回复计算 loss。多轮历史中的 assistant TTS（前轮回复）不参与监督，避免模型学到"看见 query 直接出 TTS"的错误模式。对于当前轮 `user -> tool_call -> tool-result -> TTS` 的样本，tool_call 和 TTS 都会被监督。若截断导致所有回复都找不到，才会过滤 `labels` 全为 `-100` 的空监督样本。

每次训练启动后会清空并重写 `output_dir/train_metrics.jsonl`，避免多次训练追加到同一个指标文件导致曲线抖动。训练过程的 stdout/stderr 会同时写入 `output_dir/train.log`，可用以下命令事后审阅或实时查看：

```bash
tail -f lora_output/train.log
```

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
  --port 8000 \
  --log-file /tmp/qwen_omni_serve.log
```

服务端启动时会打印实际 attention implementation；如果缺少 `flash_attn` 会回退到 PyTorch `sdpa`，避免继续显式使用 `eager`。请求日志会输出 `[PERF]` 单次分段耗时，并在每次成功请求后输出 `[PERF_AVG]` 进程内累计平均耗时，覆盖消息转换、音频/多模态处理、processor、generate、解析和保存等阶段。默认不再运行本地 Whisper ASR 调试；需要额外转写排查音频时再加 `--debug-asr`。
`serve.py` 默认会把 stdout/stderr、`serve` logger 和 uvicorn 日志写到公共路径 `/tmp/qwen_omni_serve.log`，其他用户可直接查看：

```bash
tail -f /tmp/qwen_omni_serve.log
```

需要每次启动覆盖旧日志时加 `--log-file-mode w`；需要关闭文件日志时传 `--log-file ""`。
服务端在送模前会屏蔽历史轮次里的工具调用和 `tool` 结果，包括 `assistant.tool_calls` 以及历史里直接写成 JSON 工具调用的 `assistant.content`；仅当工具链出现在最新用户消息之后时才保留。保留的 `assistant.tool_calls`、旧 `Action:` 文本和 `tool` JSON 结果会统一压缩成训练数据使用的一行紧凑 JSON，避免推理格式与训练格式不一致。
模型输出 `Reject` 或 `NoiseDoNotAct` 时，服务端只打印诊断日志，不向客户端返回文本或工具调用。

可选开启实验性的 system prompt KV cache：

```bash
python serve.py \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --host 0.0.0.0 \
  --port 8000 \
  --prompt-cache kv
```

`--prompt-cache kv` 会在启动时预热固定 system prompt 的 KV，并仅在 text-only 请求的 chat template 前缀和 token 前缀都严格匹配时复用；音频、图片、视频请求会自动 miss 并走原始路径。cache hit 时服务保留完整 tokenized input，让 HF generate 按 KV 长度在内部切出未处理的 suffix token，避免手工切 suffix 导致 `cache_position`/RoPE 错位，并临时恢复预热时记录的 `rope_deltas`，请求结束后再还原模型状态。如果当前模型封装不支持 KV 预热，服务会打印禁用原因并自动回退到 `none`，避免启动失败。该模式会在 `[PERF] inference` 中标记 `prompt_cache=hit|miss|off`、`cache_prefix_tokens=<命中 token 数>` 和 `cache_miss_reason=<原因>`，建议先用同一批 eval 对比 `none` 和 `kv` 的输出一致性后再长期启用。

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

**音频请求（base64 编码音频）**
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

`input_audio.data` 支持纯 base64，也支持 `data:audio/<fmt>;base64,` 前缀的 data URL。服务端会先解码音频，再交给 Qwen Omni processor；如果上传内容已经是 16kHz mono PCM16 WAV，会跳过 `ffmpeg` 转码。其他 WAV/MP3/PCM 输入会用 `ffmpeg` 统一转为 16kHz mono WAV；裸 PCM 建议传 `format: "pcm"`，并带上 `sample_rate: 16000`、`channels: 1`。

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

**本地录音端到端测试**
```bash
# 本地录 4 秒音频，发送到远端 serve.py 服务
python scripts/record_remote_infer.py \
  --server http://10.95.64.153:8000 \
  --duration 4

# 使用已有 WAV 文件测试
python scripts/record_remote_infer.py \
  --server http://10.95.64.153:8000 \
  --audio data/eval/audio/window/window_001.wav
```

`record_remote_infer.py` 会把本地 WAV 转成 OpenAI 兼容的 `data:audio/wav;base64,` 音频请求，并打印返回的 `tool_call` 或文本结果。默认是纯音频端到端测试；如需调试服务端后处理，可额外传 `--hint-text "打开主驾车窗"`。

**Gradio 本地录音界面**
```bash
pip install gradio   # 首次运行需安装
python scripts/gradio_remote_infer.py \
  --server http://10.95.64.153:8000 \
  --host 127.0.0.1 \
  --port 7860
```

打开 `http://127.0.0.1:7860` 后可直接录音或上传 WAV，页面会展示解析后的 `tool_call`、参数和完整响应。
如果浏览器录音按钮没有录上声音，可以使用页面里的 `Backend Record` 或 `Backend Record + Send`，它会绕过浏览器录音，直接由本地 Python 进程调用系统录音工具生成 WAV。
客户端会检查 WAV 的 RMS 和峰值，若录到静音文件会直接提示，不再把空音频发送到远端。
Gradio 发送前会把浏览器录音或上传音频规范化为 16kHz mono PCM16 WAV，因此服务端可直接走音频快路径；若输入已经满足该格式，则不会额外转码。
macOS 下如果存在系统 `afrecord` 会优先使用它；否则使用 Python 音频库时会检查输入设备，避免选错设备后录到静音。浏览器录音完成后页面也会自动显示 RMS/峰值诊断信息。
Chrome 浏览器录音会保存为 `mp3`，避免当前前端 WAV 转换链路产生静音文件；发送到服务端后由 `serve.py` 自动识别并转码。

API 端点：
- `POST /v1/chat/completions` – 推理
- `GET  /v1/models` – 列出模型
- `GET  /health` – 健康检查

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

评测脚本支持少量业务等价答案：例如「车里太闷了」这类未明确指定车窗或空调的通风意图，`ClimateControl` 切外循环和 `WindowControl` 打开车窗都计为正确。
评测和服务端都不做规则后处理修正预测工具或参数，指标与线上响应都反映模型原始输出。

### 评测维度

- **Per-file**：每个测试文件（18 个场景）独立统计
- **By Difficulty**：按 easy / medium / hard 分层
- **By Category**：按 category 分组，展示最弱的 10 个

### 评测报告

Batch 模式运行后自动输出 JSON 报告（默认 `eval_report_<timestamp>.json`），包含：
- 时间戳、模型路径、LoRA 路径
- `evaluation_mode: raw_model_output` 和 `postprocess_applied: false`
- 总体指标 + per-file / per-difficulty / per-category 明细
- 所有错误样本（含 query、gt、pred、err_type）
- 解析后的工具名和参数保持模型原始输出，不做规则后处理修正
- `position` 为可选参数；用户未明确位置时不因缺少位置追问，直接省略 `position`，由工具侧按说话人位置补全
- 多意图样本暂不计入单工具指标；`eval.py` 会跳过 `intent/sub_category=多意图` 或包含多个 `expected_tool_calls` 的样本

### 评测数据

- 路径：`data/eval/*_test.json`（36 个文件，1778 条样本；其中 169 条多意图/多工具样本被单工具评测 mask）
- 音频：`data/eval/audio/`（1599 条有对应 wav 文件）
- 输入方式：有音频文件时自动用音频输入，无音频时回退到文本
- 支持字段：`expected_type`（显式指定 Action/Clarify/Reject）

## 脚本总览

| 脚本 | 用途 |
|------|------|
| `build_train_data.py` | 合并 splits + 注入 SP → 训练集 |
| `train_thinker_lora.py` | LoRA 训练，含冻结审计 + 训练指标记录 |
| `serve.py` | **OpenAI 兼容推理服务**（FastAPI，支持文本+音频） |
| `eval.py` | 统一评测（batch / single），音频输入 + 多维度统计，支持 `--batch-size` 批量推理 |
| `scripts/analyze_eval_errors.py` | 读取 `eval_report*.json`，按类型、工具、文件、类别聚类错误；可用 `--backlog-md` 输出训练补强任务清单 |
| `scripts/validate_splits.py` | 校验 split 样本消息结构、工具调用和响应形态 |
| `scripts/validate_by_tool_schema.py` | 校验 `data/splits/by_tool/*.jsonl` 是否符合 `data/tools.json` schema |
| `scripts/generate_train_report.py` | 从 `train_metrics.jsonl` 生成 HTML 训练可视化报告 |
| `scripts/record_remote_infer.py` | 录音或读取音频并请求远端 OpenAI 兼容推理服务 |
| `scripts/gradio_remote_infer.py` | 远端推理服务的 Gradio 调试界面 |

已归档至 `_archive/`：`split_data_by_type.py`、`augment_reject_samples.py`、`build_system_prompt.py`

评测后的推荐排查顺序：

```bash
python scripts/analyze_eval_errors.py eval_report.json --limit 20
python scripts/analyze_eval_errors.py eval_report.json --limit 20 --backlog-md docs/eval-error-training-backlog.md
```

该流程用于决定下一轮补数据方向；不通过规则后处理抬高线上或评测准确率。

## 已完成的优化

以下问题在历史迭代中已修复：

- [x] max_length 1024 → 16384（防止样本截断）
- [x] last-assistant-only loss masking（仅监督每条样本最后一个 assistant 回复）
- [x] SP 压缩 53%（26K → 12K chars）
- [x] SP 统一管理（`data/system-prompt.txt`，训练/推理/评测共用）
- [x] 训练数据不再内嵌 SP，由 build_train_data.py 构建时注入
- [x] lr 1e-4 → 2e-5，alpha 32 → 16，添加 warmup/weight_decay/grad_clip
- [x] load_best_model_at_end，按 eval_token_acc 选最优 checkpoint
- [x] gradient_checkpointing，batch=1 + grad_accum=8（24GB 显存适配）
- [x] 过滤空监督样本，避免全 `-100` label batch 污染 `eval_loss`
- [x] Reject 数据增强（103 条硬负例：家电混淆、多轮拒绝、跨域请求）
- [x] 分类逻辑按最后一条 assistant turn 判断（多轮样本正确分类）
- [x] 冻结审计自动化（forbidden keyword → auto-freeze → fail-fast）
- [x] R4 数据增强（+358 条：修复过度-Clarify、补齐弱工具媒体/电话/信息）
- [x] eval.py `--batch-size` 批量推理（单 GPU 利用率从 ~30% → ~75%）
- [x] R5 数据增强（+164 条：position 字段覆盖、抗过度-Clarify、Climate/Light 多样性）
- [x] R6 anti-Clarify 清理：位置缺失不追问，口语意图/信息查询/电话与 FM 搜索直接 Action
- [x] 数据质量清洗（2026-05-10）：修正 clarify 中可选 position 误追问、reject 误标工具样本，并将问候/感叹词/道别/闲聊等无意图噪声迁移至 noise；一次性清洗脚本已删除，保留清洗后的 split 产物
- [x] 训练过程指标持久化：`MetricsSaverCallback` 每隔 `--logging-steps` 步追加写入 `{output_dir}/train_metrics.jsonl`（含 step/epoch/loss/lr/grad_norm/eval_loss/eval_token_acc）
- [x] HTML 训练报告生成：`scripts/generate_train_report.py` 从 `train_metrics.jsonl` 生成暗色主题交互式 HTML（Chart.js），包含 train loss、eval loss + token acc 双轴、学习率、grad norm 四图及关键统计卡片

## 训练监控

训练过程中自动记录指标到 `{output_dir}/train_metrics.jsonl`，每行一个 JSON：

```json
{"step": 10, "epoch": 0.05, "loss": 1.2345, "learning_rate": 1.23e-05, "grad_norm": 0.88}
{"step": 100, "epoch": 0.50, "eval_loss": 0.98, "eval_token_acc": 0.72}
```

训练结束后生成 HTML 报告：

```bash
python scripts/generate_train_report.py --metrics lora_output/train_metrics.jsonl
# → lora_output/train_report.html
```

报告包含：Train Loss、Eval Loss + Token Acc 双轴、Learning Rate、Grad Norm 四张交互图，以及总步数/最优 eval loss/最优 eval acc 等统计卡片。无额外 Python 依赖（Chart.js 通过 CDN 加载），浏览器直接打开。

## 下一步

- [ ] 按 `scripts/analyze_eval_errors.py --backlog-md` 的 P0/P1 错误族继续补充非评估原句 hard case，并用 `build_train_data.py --sample-weight` 加权训练
- [ ] 补充 Clarify 评测数据（当前 0 条，训练集有 ~151 条）
- [ ] 补充 Reject 评测数据（当前 1 条，训练集有 ~1132 条）
- [ ] 补齐 9 个无覆盖工具的测试数据（GeneralBack/Exit/Select、NavigationControl 等）
- [ ] 工具混淆问题（雨刮→ClimateControl、播放→MediaControl vs MusicSearchPlay）
- [ ] 阶段 B：DPO/ORPO 定向提准
- [ ] 导出部署：合并 LoRA → ONNX/GGUF
