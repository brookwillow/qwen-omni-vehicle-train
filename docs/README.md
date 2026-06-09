# 车载语音助手 LoRA 训练方案

> 更新时间：2026-05-21

## 项目概要

基于 Qwen2.5-Omni-3B 的 Thinker-only LoRA 微调，实现车载 ReAct 风格工具调用。

- 基模型：`Qwen2.5-Omni-3B`（`max_position_embeddings=32768`）
- 训练策略：LoRA (r=8, alpha=16)，仅保留 Thinker 内的可训练 LoRA；Talker/token2wav 和音频相关模块冻结
- LoRA 挂载层：Attention 的 `q_proj/k_proj/v_proj/o_proj` + MLP 的 `gate_proj/up_proj/down_proj`
- 冻结模块：Audio tower + Talker + token2wav/语音生成链路（通过关键词审计和 target module 匹配范围保证）
- 输出格式：紧凑 JSON 工具调用 / 自然语言 TTS 文本 / `Reject` 三类决策

### 模型结构与参数量简版

以下统计来自直接加载 `Qwen2.5-Omni-3B` checkpoint 后读取模型模块和参数量。`Qwen2.5-Omni-3B` 的“3B”是模型命名口径；完整 Omni checkpoint 实际约 5.54B 参数，因为包含 Thinker 之外的 Talker 和 token2wav 音频生成链路。


| 一级模块    | 类名                                         | 参数量                     | 说明                                                                |
| ----------- | -------------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| 全模型      | `Qwen2_5OmniForConditionalGeneration`        | 5,537,120,640（约 5.537B） | 完整 Omni checkpoint                                                |
| `thinker`   | `Qwen2_5OmniThinkerForConditionalGeneration` | 4,703,464,448（约 4.703B） | 多模态理解 + 文本推理 + LM head，当前 LoRA 可训练参数保留在该模块内 |
| `talker`    | `Qwen2_5OmniTalkerForConditionalGeneration`  | 384,604,928（约 384.6M）   | 语音生成前段，当前训练冻结                                          |
| `token2wav` | `Qwen2_5OmniToken2WavModel`                  | 449,051,264（约 449.1M）   | codec/token 到波形，当前训练冻结                                    |

`thinker` 内部主要模块：


| 模块路径              | 类名                          | 层数/结构           | 参数量                     |
| --------------------- | ----------------------------- | ------------------- | -------------------------- |
| `thinker.audio_tower` | `Qwen2_5OmniAudioEncoder`     | 32 层 audio encoder | 637,676,544（约 637.7M）   |
| `thinker.visual`      | `Qwen2_5OmniVisionEncoder`    | 32 个 vision blocks | 668,684,288（约 668.7M）   |
| `thinker.model`       | `Qwen2_5OmniThinkerTextModel` | 36 层 text decoder  | 3,085,938,688（约 3.086B） |
| `thinker.lm_head`     | `Linear`                      | vocab projection    | 311,164,928（约 311.2M）   |

`thinker.model` 文本路径结构：


| 模块路径                     | 类名             | 参数量                     |
| ---------------------------- | ---------------- | -------------------------- |
| `thinker.model.embed_tokens` | `Embedding`      | 311,164,928（约 311.2M）   |
| `thinker.model.layers`       | `ModuleList[36]` | 2,774,771,712（约 2.775B） |
| `thinker.model.norm`         | `Qwen2RMSNorm`   | 2,048                      |

单个 text decoder layer 约 77,076,992 参数，核心结构如下：


| 单层模块                                   | LoRA 挂载 | 参数量                 |
| ------------------------------------------ | --------- | ---------------------- |
| `self_attn.q_proj`                         | 是        | 4,196,352（约 4.2M）   |
| `self_attn.k_proj`                         | 是        | 524,544（约 0.5M）     |
| `self_attn.v_proj`                         | 是        | 524,544（约 0.5M）     |
| `self_attn.o_proj`                         | 是        | 4,194,304（约 4.2M）   |
| `mlp.gate_proj`                            | 是        | 22,544,384（约 22.5M） |
| `mlp.up_proj`                              | 是        | 22,544,384（约 22.5M） |
| `mlp.down_proj`                            | 是        | 22,544,384（约 22.5M） |
| `input_layernorm/post_attention_layernorm` | 否        | 各 2,048               |

文本路径关键配置：`hidden_size=2048`，`intermediate_size=11008`，`num_hidden_layers=36`，`num_attention_heads=16`，`num_key_value_heads=2`，`max_position_embeddings=32768`，`vocab_size=151936`。

当前 `target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 是按模块名全局匹配。直接读取模型结构可见匹配范围为：`thinker.model` 252 个模块、`thinker.visual` 96 个模块、`thinker.audio_tower` 96 个模块、`talker.model` 168 个模块。训练脚本会通过 `audio,talker,vocoder,audio_decoder,speech_decoder` 关键词把 `thinker.audio_tower` 和 `talker` 相关 LoRA 参数自动冻结；因此最终可训练 LoRA 主要保留在 `thinker.model`，同时包含 `thinker.visual` 中匹配到的 LoRA 参数。当前车控训练以文本/音频指令为主，主要行为收益来自 `thinker.model` 文本 decoder。

`talker` 和 `token2wav` 结构：


| 模块路径                           | 类名                               | 层数/结构                                  | 参数量                   |
| ---------------------------------- | ---------------------------------- | ------------------------------------------ | ------------------------ |
| `talker.thinker_to_talker_proj`    | `Linear`                           | Thinker hidden states 到 Talker 表示的投影 | 1,835,904（约 1.8M）     |
| `talker.model`                     | `Qwen2_5OmniTalkerModel`           | 24 层 decoder，每层约 14.9M                | 375,199,616（约 375.2M） |
| `talker.codec_head`                | `Linear`                           | codec token projection                     | 7,569,408（约 7.6M）     |
| `token2wav.code2wav_dit_model`     | `Qwen2_5OmniToken2WavDiTModel`     | DiT 声学生成模块                           | 333,607,760（约 333.6M） |
| `token2wav.code2wav_bigvgan_model` | `Qwen2_5OmniToken2WavBigVGANModel` | BigVGAN vocoder                            | 115,443,504（约 115.4M） |

Thinker 到 Talker 之间不是简单传递普通文本字符串，而是经 `talker.thinker_to_talker_proj` 接收 Thinker 的高维 hidden states/语义表示。因此训练 Thinker LoRA 会影响工具 JSON 和文本决策，也可能改变 Talker 接收到的语义分布；冻结 Talker/token2wav 是为了避免语音生成链路被车控数据微调带偏。

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
SFT 主线
data/splits/**/*.jsonl  (已拆分好的数据，无 SP，包含 by_tool 工具文件)
  │
  └─ build_train_data.py ──→ data/train_final.jsonl (注入 SP，schema 校验，打散，可按错误族加权)
                                │
                                └─ train_thinker_lora.py ──→ lora_output_sft/
                                                                  │
                                                                  ├─ eval.py / serve.py
                                                                  │
                                                                  └─ scripts/analyze_eval_errors.py
                                                                              │
                                                                              └─ 回流修正 data/splits/

可选 DPO 对齐阶段
data/rl/*_preferences.jsonl  (chosen/rejected 偏好数据产物)
  │
  └─ train_memory_dpo_lora.py --init-lora-dir lora_output_sft/
                                │
                                └─ lora_output_sft_dpo/ ──→ eval.py / serve.py

可选 Teacher SFT 阶段
data/train_final.jsonl  (与 Omni SFT 相同，已注入 SP)
  │
  └─ train_teacher_sft_lora.py ──→ teacher_lora_qwen35_27b_sft/
                                      │
                                      └─ 后续作为 GKD / 标注 / 边界判定 teacher
```

训练产物使用方式：

- 只做 SFT：线上加载 `base model + lora_output_sft/`。
- 做 DPO：DPO 从 `lora_output_sft/` 初始化继续训练，线上加载 `base model + lora_output_sft_dpo/`，不需要同时加载两个 LoRA。
- 评测报告默认写入对应 LoRA 目录，便于把训练产物、评测结果和错误分析绑定在一起。

> 注：阶段性拆分、增强和偏好构造脚本已清理，不再作为稳定入口保留；当前可直接使用的产物在 `data/splits/` 和 `data/rl/` 下。`build_train_data.py` 默认递归合并 `data/splits/**/*.jsonl`。
>
> `build_train_data.py` 默认开启 `--validate-schema`，合并后会对每条样本中的工具调用做 schema 校验（required 字段、enum 值、未知参数），不合格样本会被移除并在 stderr 打印详情。可用 `--no-validate-schema` 跳过校验。

### 决策边界

- `NoiseDoNotAct`：当前轮次没有明显意图，或这句话不是对车载助手说的；如果样本里出现车控/电话/车况等关键词，必须显式写明“不是对小P说的 / 不用执行 / 不是车内指令”，避免覆盖有效工具请求。
- `Reject`：当前轮次有明确意图，但意图不在当前工具和模型能力范围内，例如外部应用内任务、通用百科/新闻/娱乐问答、家居控制等。
- 工具调用：当前轮次必须是有意义且在工具范围内的请求，才输出紧凑 JSON tool call；不要因为历史里有工具调用而继承到当前无意义轮次。

### 关键文件


| 文件                                               | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/system-prompt.txt`                           | 紧凑版 System Prompt（~5.8K chars，基于当前工具白名单生成）                                                                                                                                                                                                                                                                                                                                                                                               |
| `data/tools.json`                                  | 40 个车载工具定义（新版`inputSchema` 格式，已包含 `IdentityControl`）                                                                                                                                                                                                                                                                                                                                                                                     |
| `data/splits/by_tool/*.jsonl`                      | 按工具拆分的训练数据；每个工具文件内已拆为`user -> JSON tool call` 决策样本和独立 `JSON tool call -> tool-role JSON result -> TTS text` 回复样本；`AppControl` 覆盖明确打开/关闭/下载应用的意图，`音乐应用` 和 `媒体` 分别按 schema 保留，播放内容/导航到目的地等应用内任务走 `Reject`；`IdentityControl.jsonl` 覆盖 FaceID 昵称录入；`PhoneControl.jsonl` 包含小鹏客服、小鹏救援、儿童手表等官方默认联系人样本；其中 `NoiseDoNotAct.jsonl` 当前为 449 条 |
| `data/splits/clarify.jsonl`                        | required 字段缺失后的自然语言追问样本                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `data/splits/edge_case.jsonl`                      | 多轮上下文边界、易混淆与任务列表选择样本                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `data/splits/multiturn.jsonl`                      | 最多三轮纯文本历史上下文样本；历史可来自导航/音乐/新闻/百科/AIGC/天气等外部域，当前轮优先，只有代词、省略、纠错、延续或查询缺槽时才参考历史补全                                                                                                                                                                                                                                                                                                           |
| `data/splits/orchestration.jsonl`                  | feature 分支复杂任务编排样本：并行指令、多工具 JSON 数组、显式多步、最近意图继承、列表选择和模糊导航边界                                                                                                                                                                                                                                                                                                                                                  |
| `data/splits/reject.jsonl`                         | 单轮 + 多轮硬负例                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `data/splits/hard_cases/OverNoiseRemaining_20260607.jsonl` | 2026-06-07 最新评估中仍被预测为 `NoiseDoNotAct`、但 GT 为有效工具请求的 42 条 SFT hard case；用于在下一轮 SFT 中直接降低 noise 过召                                                                                                                                                                                                                                                                                                                       |
| `data/rl/memory_preferences.jsonl`                 | 记忆使用偏好数据（299 条），用于 DPO 阶段强化多轮上下文选择                                                                                                                                                                                                                                                                                                                                                                                               |
| `data/rl/memory_contrast_preferences.jsonl`        | 正确工具 JSON vs 错误历史工具 JSON 的记忆强对比偏好数据                                                                                                                                                                                                                                                                                                                                                                                                   |
| `data/rl/tool_tts_preferences.jsonl`               | 工具调用 vs TTS 回复的输出契约偏好数据（工具 JSON chosen，执行完成话术 rejected）                                                                                                                                                                                                                                                                                                                                                                         |
| `data/rl/tool_boundary_preferences.jsonl`          | 边界偏好数据（36 条）：`Reject/NoiseDoNotAct/澄清TTS` chosen，错误工具调用 rejected，用于约束 DPO 过度工具化                                                                                                                                                                                                                                                                                                                                              |
| `data/rl/current_noise_boundary_preferences.jsonl` | 多轮边界偏好数据（200 条）：历史存在工具调用但当前轮为“嗯/好/先这样/空输入/不是那个”等无动作 query 时，`NoiseDoNotAct` chosen，错误继承历史工具 rejected                                                                                                                                                                                                                                                                                                |
| `data/rl/noise_false_positive_preferences.jsonl`   | 有效工具请求被误判为 `NoiseDoNotAct` 的偏好数据，`chosen=正确工具`、`rejected=NoiseDoNotAct`，用于修正 noise 过召                                                                                                                                                                                                                                                                                                                                          |
| `data/rl/anti_over_noise_preferences.jsonl`        | 2026-06-06 评估报告中的 70 条 noise 过召偏好数据；其中 24 条已人工 review 确认为 `model_fail`，其余为同一报告中 `pred=NoiseDoNotAct` 且 GT 为有效工具的 over-noise 候选，作为当前 SFT 后的定向 DPO 首选数据                                                                                                                                                                                                                                                  |
| `data/rl/still_over_noise_preferences_round2.jsonl` | 第二轮 DPO 偏好数据（49 条）：`eval_report_20260606_235501.json` 中仍然 `pred=NoiseDoNotAct` 但 GT 为有效工具的样本，继续修正 noise 过召                                                                                                                                                                                                                                                                                                                  |
| `data/rl/still_over_noise_preferences_round3.jsonl` | 第三轮 DPO 偏好数据（45 条）：`eval_report_20260607_005951.json` 中仍然 `pred=NoiseDoNotAct` 但 GT 为有效工具的样本；只用于窄口径 over-noise 继续修正                                                                                                                                                                                                                                                                                                      |
| `data/rl/wrong_tool_preferences.jsonl`             | 第二轮 DPO 偏好数据（49 条）：GT 工具和预测工具不同，且 chosen/rejected 均通过 schema 校验，用于修正工具混淆                                                                                                                                                                                                                                                                                                                                               |
| `data/rl/false_reject_clarify_preferences.jsonl`   | 第二轮 DPO 偏好数据（12 条）：有效工具请求被输出为 `Reject` 或澄清话术，`chosen=正确工具`、`rejected=Reject/Clarify`                                                                                                                                                                                                                                                                                                                                       |
| `data/rl/extra_args_preferences.jsonl`             | 第二轮 DPO 偏好数据（90 条）：预测工具正确但多输出了额外参数，`chosen=精简正确参数`、`rejected=多余参数版本`；训练时建议低权重使用，避免模型过度删槽                                                                                                                                                                                                                                                                                                      |
| `data/train_final.jsonl`                           | 最终训练数据（含 SP）                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `data/eval/`                                       | 评测数据集（当前工具 schema 已清洗，`音乐应用` 和 `媒体` 按新版 schema 分开保留）                                                                                                                                                                                                                                                                                                                                                                         |

数值槽位口径：用户明确说具体数字或百分比时，`value` 保留原始数字字符串，例如音量 `15`、音量 `30`、车窗 `50%`；只有用户说“最高/最低/中等/某模式”时才映射为枚举档位。

### 数据分布（当前）


| 类型                                   | 数量 | 说明                                                                                                                                                                                                                                                                                            |
| -------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| By-tool                                | 7314 | 每个工具文件内混合：Action JSON、NoiseDoNotAct、独立`JSON tool call -> tool-role JSON result -> TTS text` 回复样本；`NoiseDoNotAct` 中高风险裸表达已改写为明确非对小P说/不用执行口径                                                                                                                                                |
| Clarify                                | 189  | 已拆为两类样本：`user -> 追问 TTS`（94 条，教模型何时追问）+ 完整 4 轮 `user -> 追问 -> 用户补齐 -> tool call`（95 条，教模型追问后如何响应）；已移除纯位置追问（音区可自动判定位置）与意图明确的方向性追问（如"太晒→打开/关闭遮阳帘"）；配合 final-assistant 标签监督，两类样本均能被正确监督 |
| Edge case                              | 100  | 多轮当前轮边界、查询 vs 控制、popup/task 列表`GeneralSelect` 等易混淆样本                                                                                                                                                                                                                       |
| Hard case: CurrentNoiseWithHistoryTool | 200  | 历史存在明确工具调用，当前轮为无意义/终止/闲聊/空输入时必须`NoiseDoNotAct`，避免从历史轮次错误继承工具                                                                                                                                                                                          |
| Multiturn                              | 370  | 最多三轮纯文本历史；历史允许外部域文本；当前轮输出分布：工具 243 条、NoiseDoNotAct 79 条、Reject 36 条、自然语言 TTS 12 条                                                                                                                                                                      |
| Orchestration                          | 30   | feature 分支复杂任务编排样本；包含多工具 JSON 数组输出、显式多步、最近意图继承、列表选择与模糊导航拒识                                                                                                                                                                                          |
| Reject                                 | 1135 | 单轮 + 多轮硬负例（已合并），最后一条 assistant 均为`Reject`；已抽稀家居控制负例并移除高风险车控状态拒识                                                                                                                                                                                        |

`build_train_data.py` 默认递归合并全部 split，当前未加权最终训练集为 9755 条；统计时多轮样本按最后一个有效决策标签计入对应类别。当前未加权输出类型为 Action JSON 5431、MultiAction JSON 数组 18、NoiseDoNotAct 736、TTS 2400、Reject 1170。`NoiseDoNotAct` 以工具 JSON 形式训练，但语义上属于不执行动作边界，训练日志会单独统计 `Noise`，避免和普通 Action 混在一起误判工具化倾向。

### Hard Case 加权

`--oversample` 仍按文件 stem 加权；新增 `--sample-weight` 支持按 stem、路径后缀或 glob 给具体 split 文件加权，适合将评测错误族沉淀为独立 hard case 文件后提高采样占比。

```bash
python build_train_data.py \
  --sample-weight 'hard_cases/OverNoiseRemaining_20260607.jsonl:4' 'hard_cases/*.jsonl:3' ProfileControl:2 WindowControl:1.5 CurrentNoiseWithHistoryTool:0.5 NoiseDoNotAct_coverage:0.5 \
  --output data/train_final.jsonl
```

推荐流程：先用 `scripts/analyze_eval_errors.py --backlog-md` 生成错误族补强清单，再为 P0/P1 错误族补充 hard case；训练时对这些 hard case 文件做 2-4 倍采样，避免新增样本在 9K 级训练集中被稀释。多轮无意义 query 和普通 noise 覆盖数据已经足够密集，当前建议在全量 hard case 放大时显式下调 `CurrentNoiseWithHistoryTool` 与 `NoiseDoNotAct_coverage`，并对 `OverNoiseRemaining_20260607` 使用 4 倍采样，防止 `NoiseDoNotAct` 过强导致短但有效的车控/电话/查询请求被误拒识。上述推荐命令试跑后输出 10818 条，Noise 占比约 5.9%。

### DPO 偏好训练

偏好数据构造脚本属于阶段性实验工具，已从稳定脚本入口中移除。仓库只保留可直接训练的偏好数据产物；DPO 训练采用“方案 2”：从已有 SFT LoRA 初始化继续训练，原 SFT 产物不被覆盖，最终输出一个新的 DPO LoRA。线上只加载 `base model + 新 DPO LoRA`，不需要同时加载两个 LoRA。

当前如果目标是修正“有效车控被过度判为 `NoiseDoNotAct`”，先放弃上一轮混合 DPO 数据，只使用 anti-over-noise 正样本偏好和已有 false-positive 偏好；不要把 `current_noise_boundary_preferences.jsonl` 一起放进本轮实验，否则会继续强化 `NoiseDoNotAct` chosen，容易抵消这次修正方向。`--preference-weight` 只适合放大权重，当前脚本不支持用小于 1 的权重做下采样。

```bash
python train_memory_dpo_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --init-lora-dir lora_output_v2_0604 \
  --preference-file data/rl/anti_over_noise_preferences.jsonl,data/rl/noise_false_positive_preferences.jsonl \
  --preference-weight anti_over_noise_preferences.jsonl:4 \
  --output-dir lora_output_v2_0604_dpo_anti_over_noise \
  --prompt-format chat_template \
  --system-prompt data/system-prompt.txt \
  --lr 1e-6 \
  --beta 0.05 \
  --epochs 1 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --sft-loss-weight 0.1 \
  --reference-mode reference_free
```

`train_memory_dpo_lora.py` 默认使用 reference-free DPO-style loss，适合先在单卡上快速验证；服务器显存充足时可用 `--reference-mode frozen_init` 加载一份冻结的 `--init-lora-dir` 作为 reference，代价是显存约翻倍。DPO 不是重新选择 LoRA 挂载层，而是从已有 SFT LoRA adapter 初始化，继续训练其中已经存在的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` LoRA 参数。DPO 默认 `--prompt-format chat_template`，会把 preference 行里的 `history + current_query` 按 `data/system-prompt.txt` 和 tokenizer chat template 组织成与 `serve.py` 一致的真实多轮输入；历史分布实验不要退回旧的 `json_instruction`，否则训练输入和线上输入不一致。为适配 24GB 显存，DPO 默认开启 `--gradient-checkpointing` 和 `--empty-cache-between-pairs`；如果仍 OOM，优先加 `--max-length 3584`，再降到 `3072`。DPO 阶段学习率和训练步数要保守，本轮 anti-over-noise 建议先用 `lr=1e-6`、`beta=0.05`、`epochs=1`，并保留少量 `--sft-loss-weight 0.1` 防止整体工具格式漂移。训练后必须同时回归原始 eval、multiturn、orchestration、reject/noise 边界集和 `noise_history_test.json`；如果真实 noise 被误工具化，先减少 `anti_over_noise_preferences.jsonl` 权重或补一份人工确认的真实 noise guard 偏好，再做下一轮。

第二轮 DPO 可从上一轮 anti-over-noise LoRA 继续，目标是继续修正仍然 over-noise 的样本，同时加入工具混淆、误拒识/误澄清和少量 extra-args 偏好。`extra_args_preferences.jsonl` 只给低权重，避免模型学成“能少填就少填”。

```bash
python train_memory_dpo_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --init-lora-dir lora_output_v2_0605_dpo_anti_over_noise \
  --preference-file data/rl/still_over_noise_preferences_round2.jsonl,data/rl/false_reject_clarify_preferences.jsonl,data/rl/wrong_tool_preferences.jsonl,data/rl/extra_args_preferences.jsonl,data/rl/noise_false_positive_preferences.jsonl \
  --preference-weight still_over_noise_preferences_round2.jsonl:6 false_reject_clarify_preferences.jsonl:4 wrong_tool_preferences.jsonl:3 extra_args_preferences.jsonl:1 \
  --output-dir lora_output_v2_0605_dpo_error_repair_round2 \
  --prompt-format chat_template \
  --system-prompt data/system-prompt.txt \
  --lr 8e-7 \
  --beta 0.05 \
  --epochs 1 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --sft-loss-weight 0.1 \
  --reference-mode reference_free
```

第三轮如果第二轮结果表现为 `over_noise` 继续下降但 `wrong_tool/extra_args` 没有明显收益，应停止 mixed DPO，改为只针对最新剩余 over-noise 做窄训练。不要继续带 `extra_args_preferences.jsonl`，否则容易增加多余参数问题。

```bash
python train_memory_dpo_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --init-lora-dir lora_output_v2_0605_dpo_error_repair_round2 \
  --preference-file data/rl/still_over_noise_preferences_round3.jsonl \
  --preference-weight still_over_noise_preferences_round3.jsonl:6 \
  --output-dir lora_output_v2_0605_dpo_over_noise_round3 \
  --prompt-format chat_template \
  --system-prompt data/system-prompt.txt \
  --lr 6e-7 \
  --beta 0.05 \
  --epochs 1 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --sft-loss-weight 0.1 \
  --reference-mode reference_free
```

新一轮 SFT 后如果报告里仍有 `NoiseDoNotAct` 过召或有效工具请求被误判为 `Reject`，先从评估报告生成当轮边界偏好数据，再跑窄口径 DPO。以下命令基于 `eval_report_20260609_105416.json` 生成了 51 条 over-noise 偏好和 14 条 false-reject 偏好：

```bash
python scripts/build_boundary_preferences_from_eval.py \
  /Users/wangjie/Downloads/serve_logs/eval_report_20260609_105416.json \
  --output-dir data/rl
```

生成文件：

- `data/rl/still_over_noise_preferences_20260609.jsonl`
- `data/rl/false_reject_preferences_20260609.jsonl`

然后从新 SFT LoRA 继续做边界 DPO：

```bash
python train_memory_dpo_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --init-lora-dir lora_output_sft_over_noise_repair \
  --preference-file data/rl/still_over_noise_preferences_20260609.jsonl,data/rl/false_reject_preferences_20260609.jsonl,data/rl/noise_false_positive_preferences.jsonl \
  --preference-weight still_over_noise_preferences_20260609.jsonl:6 false_reject_preferences_20260609.jsonl:4 \
  --output-dir lora_output_sft_over_noise_repair_dpo_boundary_20260609 \
  --prompt-format chat_template \
  --system-prompt data/system-prompt.txt \
  --lr 6e-7 \
  --beta 0.05 \
  --epochs 1 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --sft-loss-weight 0.1 \
  --reference-mode reference_free
```

如果上述边界 DPO 表现为 over-noise 下降但 type/reject/args 副作用变大，不要继续在该 DPO LoRA 上叠训练。回到新 SFT LoRA，只使用最新剩余 over-noise 偏好做窄训练。基于 `eval_report_20260609_143403.json` 已生成：

- `data/rl/still_over_noise_preferences_20260609_round2.jsonl`（42 条）

推荐命令：

```bash
python train_memory_dpo_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --init-lora-dir lora_output_sft_over_noise_repair \
  --preference-file data/rl/still_over_noise_preferences_20260609_round2.jsonl \
  --preference-weight still_over_noise_preferences_20260609_round2.jsonl:6 \
  --output-dir lora_output_sft_over_noise_repair_dpo_over_noise_20260609_round2 \
  --prompt-format chat_template \
  --system-prompt data/system-prompt.txt \
  --lr 4e-7 \
  --beta 0.05 \
  --epochs 1 \
  --train-batch-size 1 \
  --grad-accum 8 \
  --sft-loss-weight 0.1 \
  --reference-mode reference_free
```

## 训练配置

### 标准 Pipeline 入口

后续训练命令以 `scripts/run_training_pipeline.py` 为单一入口维护；每次 SFT/DPO 采样比例、LoRA 输出目录或偏好数据组合变化，优先更新该脚本，再同步本文档。脚本默认 dry-run，只打印将要执行的命令；加 `--run` 才会真正启动训练或评估。

```bash
# 打印完整 SFT -> eval -> DPO -> eval 流程
python scripts/run_training_pipeline.py full

# 真正执行完整流程
python scripts/run_training_pipeline.py full --run

# 只执行 SFT 段：validate -> build -> sft -> eval-sft
python scripts/run_training_pipeline.py sft-only --run

# SFT 已完成后，只跑 noise 修复 DPO 和 DPO 评估
python scripts/run_training_pipeline.py dpo-only --run
```

可单独运行的 stage 包括：`validate`、`build`、`sft`、`eval-sft`、`dpo-noise`、`eval-dpo`。常用覆盖参数如下：

```bash
python scripts/run_training_pipeline.py full --run \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --sft-output-dir lora_output_sft_over_noise_repair \
  --dpo-noise-output-dir lora_output_sft_over_noise_repair_dpo_noise
```

当前 pipeline 固定了以下策略：SFT 构建时 `OverNoiseRemaining_20260607` 4 倍、全量 hard case 3 倍、`ProfileControl` 2 倍、`WindowControl` 1.5 倍，同时将 `CurrentNoiseWithHistoryTool` 和 `NoiseDoNotAct_coverage` 降到 0.5；DPO 第一轮只修有效请求被误判为 `NoiseDoNotAct`，使用 `anti_over_noise_preferences.jsonl`、`noise_false_positive_preferences.jsonl`、`still_over_noise_preferences_round3.jsonl`，不混入 `extra_args_preferences.jsonl`。

### SFT 训练流程

SFT 训练前必须先重建 `data/train_final.jsonl`。推荐优先使用 pipeline 入口：

```bash
python scripts/run_training_pipeline.py sft-only --run
```

等价展开命令如下。该命令会递归合并 `data/splits/**/*.jsonl`、注入 `data/system-prompt.txt`，并对最新 over-noise hard case、全量 hard case、`ProfileControl`、`WindowControl` 做采样加权，同时下调多轮 no-op/noise 覆盖数据，避免补强数据在 9K 级训练集中被稀释。

```bash
python build_train_data.py \
  --sample-weight 'hard_cases/OverNoiseRemaining_20260607.jsonl:4' 'hard_cases/*.jsonl:3' ProfileControl:2 WindowControl:1.5 CurrentNoiseWithHistoryTool:0.5 NoiseDoNotAct_coverage:0.5 \
  --output data/train_final.jsonl
```

### Qwen3.5-27B Teacher SFT

`train_teacher_sft_lora.py` 用于训练一个 teacher，目标不是直接替代 Omni 线上模型，而是把当前车控工具 schema、`NoiseDoNotAct/Reject/tool_call` 边界和 SP 约束对齐到更大的 Qwen3.5 teacher 上，后续用于 GKD、错误样本判定或候选生成。

该脚本完全复用 Omni SFT 的训练数据入口：`data/train_final.jsonl` 必须由 `build_train_data.py` 生成，里面已经注入 `data/system-prompt.txt`。监督口径也和 `train_thinker_lora.py` 一致：只监督最后一个 user 之后的最后一条 assistant 内容；历史 assistant TTS 不参与 loss。脚本默认从 ModelScope 使用 `Qwen/Qwen3.5-27B`，本地目录为 `/home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B`；如果目录不存在，会自动执行 `modelscope download --model Qwen/Qwen3.5-27B --local_dir <本地目录>`。`Qwen/Qwen3.5-27B` 带 vision encoder，脚本会优先尝试 `AutoModelForCausalLM`，失败后回退到 `AutoModelForImageTextToText`，并默认冻结 `vision/visual/image` 等相关 LoRA 参数，只训练语言决策路径；如确实要训练视觉路径，可显式加 `--no-freeze-vision`。

H800 上推荐先用 BF16 LoRA 训练 `Qwen/Qwen3.5-27B`：

```bash
python train_teacher_sft_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B \
  --modelscope-model Qwen/Qwen3.5-27B \
  --train-file data/train_final.jsonl \
  --system-prompt data/system-prompt.txt \
  --output-dir teacher_lora_qwen35_27b_sft \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --max-length 4096 \
  --train-batch-size 1 \
  --grad-accum 16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 2
```

如果需要在启动前强制重建训练集，可加 `--rebuild-train-data`；它会使用当前 Omni pipeline 相同的 hard case 加权：

```bash
python train_teacher_sft_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B \
  --modelscope-model Qwen/Qwen3.5-27B \
  --output-dir teacher_lora_qwen35_27b_sft \
  --rebuild-train-data
```

5090 32GB 不适合直接 BF16 训练或加载 27B teacher。若只在 5090 上做小规模验证或量化 teacher 推理，可使用 4bit QLoRA/量化加载：

```bash
python train_teacher_sft_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B \
  --modelscope-model Qwen/Qwen3.5-27B \
  --train-file data/train_final.jsonl \
  --output-dir teacher_lora_qwen35_27b_qlora \
  --load-in-4bit \
  --torch-dtype bfloat16 \
  --max-length 4096 \
  --train-batch-size 1 \
  --grad-accum 16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 2
```

用于 GKD logits 蒸馏时，teacher 最好在 H800 上以 BF16 加载；4bit teacher 可以跑通流程，但概率分布会被量化压粗，建议降低 GKD loss 权重并保留 SFT CE anchor。

### RTX 3090 (24GB) 显存控制

训练脚本所有超参均可通过命令行覆盖。以下是 2026-06-04 当前实际使用的 24GB 显存保守配置，目标输出目录为 `lora_output_v2_0604`：

**SFT v2 0604 实际训练命令**

```bash
python train_thinker_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output_v2_0604 \
  --torch-dtype bfloat16 \
  --max-length 4096 \
  --train-batch-size 1 \
  --grad-accum 16 \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 3
```

该配置等效 batch size 为 16，比 `train-batch-size=2/grad-accum=4` 更稳，适合 24GB 单卡在评估阶段可能触发显存峰值的情况。若仍然 OOM，优先降低 `--max-length` 到 3584，再考虑把 `--lora-r` 降到 4。

**更省显存配置（OOM 时兜底）**

```bash
python train_thinker_lora.py \
  --model /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --train-file data/train_final.jsonl \
  --output-dir ./lora_output \
  --torch-dtype bfloat16 \
  --max-length 4096 \
  --train-batch-size 1 \
  --grad-accum 16 \
  --lora-r 4 \
  --lora-alpha 8 \
  --epochs 3
```

> 当前 SP 约 3032 tokens；按 Qwen chat template 估算，当前训练集最长样本约 3246 tokens。建议 `--max-length 4096` 起步；如继续压到 3584 需要先重新统计长度，避免截断新增长样本。

**关键参数说明**


| 参数                     | 省显存方向      | 说明                                   |
| ------------------------ | --------------- | -------------------------------------- |
| `--max-length`           | ↓ 降低         | 当前建议 4096；新增长样本后需重新统计  |
| `--train-batch-size`     | ↑ 尝试 2       | SP 压缩后可尝试 batch=2；如 OOM 回退 1 |
| `--grad-accum`           | 配合 batch 调整 | batch=2 时用 4 可保持等效 batch=8      |
| `--lora-r`               | ↓ 降低         | r=4 可节省约 10% 显存，精度略降        |
| `--torch-dtype bfloat16` | 保持            | 3090 原生支持 bf16，不要用 fp32        |
| `gradient_checkpointing` | 已内置 True     | 以 30% 速度换显存，不需要手动开        |

### 默认超参


| 参数                   | 值                     | 说明                           |
| ---------------------- | ---------------------- | ------------------------------ |
| max_length             | 16384                  | SP ~5K tokens + 对话，留足余量 |
| lr                     | 2e-5                   | 3B 小模型适用                  |
| lora_r / alpha         | 8 / 16                 | effective scaling = 2          |
| target_modules         | q/k/v/o + gate/up/down | Attention 投影层 + MLP 投影层  |
| batch_size             | 1                      | RTX 3090 24GB 显存限制         |
| grad_accum             | 8                      | 等效 batch=8                   |
| warmup_ratio           | 0.05                   | 前 5% steps 线性预热           |
| weight_decay           | 0.01                   | AdamW 正则化                   |
| max_grad_norm          | 1.0                    | 梯度裁剪防止 loss spike        |
| epochs                 | 3                      | 配合 load_best_model_at_end    |
| gradient_checkpointing | True                   | 节省显存                       |
| metric_for_best_model  | eval_token_acc         | 自动选最优 checkpoint          |

训练脚本会在编码后定位**最后一个 user 之后的最后一条** assistant 回复，并只对这一个 span 计算 loss。多轮历史中的 assistant TTS（前轮回复）不参与监督；当前轮 `user -> tool_call -> tool-result -> TTS` 的样本只监督最后一条 assistant 回复，避免短输出（如 `Reject`）被错误匹配到 system prompt 或历史文本。若截断导致该回复找不到，才会过滤 `labels` 全为 `-100` 的空监督样本。

LoRA 默认按模块名匹配 transformer 的 Attention 和 MLP 线性层：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。脚本仍会用冻结关键词做二次保护，确保 `audio/talker/vocoder/audio_decoder/speech_decoder` 等音频生成相关参数不可训练；因此实际可训练目标保留在 Thinker 内，主要是 `thinker.model` 文本 decoder，同时包括 `thinker.visual` 中匹配到的 LoRA 参数，而不是 Talker/token2wav。

每次训练启动后会清空并重写 `output_dir/train_metrics.jsonl`，避免多次训练追加到同一个指标文件导致曲线抖动。训练过程的 stdout/stderr 会同时写入 `output_dir/train.log`，可用以下命令事后审阅或实时查看：

```bash
tail -f lora_output/train.log
```

### 冻结保障

训练脚本通过关键词 `audio,talker,vocoder,audio_decoder,speech_decoder` 自动冻结命中的音频/Talker/语音生成相关 LoRA 参数，并输出审计文件：

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

服务端启动时会打印实际 attention implementation；如果缺少 `flash_attn` 会回退到 PyTorch `sdpa`，避免继续显式使用 `eager`。请求日志会输出 `[PERF]` 单次分段耗时，并在每次成功请求后输出 `[PERF_AVG]` 进程内累计平均耗时，覆盖消息转换、音频/多模态处理、processor、generate、解析和保存等阶段。默认不再运行本地 Whisper ASR 调试；需要额外转写排查音频时再加 `--debug-asr`。AUT/audio_tower hidden states 直接接 Whisper decoder 的 ASR 路线仍处于实验探测阶段，可用 `scripts/probe_asr_decoder.py` 单独验证，不属于线上推理服务默认链路。
`serve.py` 默认会把 stdout/stderr、`serve` logger 和 uvicorn 日志写到公共路径 `/tmp/qwen_omni_serve.log`，其他用户可直接查看：

```bash
tail -f /tmp/qwen_omni_serve.log
```

需要每次启动覆盖旧日志时加 `--log-file-mode w`；需要关闭文件日志时传 `--log-file ""`。
服务端在送模前会屏蔽历史轮次里的工具调用和 `tool` 结果，包括 `assistant.tool_calls` 以及历史里直接写成 JSON 工具调用的 `assistant.content`；仅当工具链出现在最新用户消息之后时才保留。保留的 `assistant.tool_calls`、旧 `Action:` 文本和 `tool` JSON 结果会统一压缩成训练数据使用的一行紧凑 JSON，避免推理格式与训练格式不一致。
模型输出 `Reject` 或 `NoiseDoNotAct` 时，服务端只打印诊断日志，不向客户端返回文本或工具调用。
feature 分支支持模型输出一行 JSON 数组来表达多个并行工具调用；`serve.py` 会转换为 OpenAI 兼容响应中的多个 `tool_calls`，并保留数组顺序作为 `tool_calls[].index`。

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

# 指定报告输出路径；默认会写入 <lora-dir>/eval_report_<timestamp>.json
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --report lora_output/eval_report.json

# 提高 GPU 利用率（默认 batch_size=1，推荐 4-8，OOM 则减小）
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --batch-size 4

# feature 分支复杂任务编排评测（计入多工具 JSON 数组）
python eval.py batch \
  --model-dir models/Qwen2.5-Omni-3B \
  --lora-dir lora_output \
  --pattern orchestration_test.json \
  --include-multi-tool

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


| 指标                 | 说明                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `type_acc`           | 响应类型准确率（Action/Clarify/Reject 是否选对）                                          |
| `tool_acc`           | 工具名称准确率（Action 类型下工具名匹配）                                                 |
| `args_em`            | 参数精确匹配率（工具名 + 所有参数完全一致）                                               |
| `multi_tool_args_em` | 多工具样本中所有工具和参数完全匹配的数量（仅`--include-multi-tool` 计入多工具行时有意义） |
| `reject_hit`         | Reject 命中数（正确拒绝 / 预测拒绝）                                                      |
| `clarify_hit`        | Clarify 命中数（正确追问 / 预测追问）                                                     |
| `parse_fail`         | 输出格式解析失败数                                                                        |

评测脚本支持少量业务等价答案：例如「车里太闷了」这类未明确指定车窗或空调的通风意图，`ClimateControl` 切外循环和 `WindowControl` 打开车窗都计为正确。
评测和服务端都不做规则后处理修正预测工具或参数，指标与线上响应都反映模型原始输出。
评测解析时只有纯 `Reject`（允许末尾标点）才计为拒识；如果 raw 输出同时包含 `Reject` 前缀和可解析的工具 JSON/`Action:`，优先按模型实际给出的工具调用计入 Action，避免把带工具调用的 raw 输出误记为拒识。
默认单工具评测仍会 mask 多意图/多工具样本；feature 分支需要评估复杂任务编排时，增加 `--include-multi-tool`，此时 JSON 数组形式的多工具输出会按无序集合匹配，样本设置 `ordered_tool_calls: true` 时按顺序匹配。

### 评测维度

- **Per-file**：每个测试文件（37 个场景）独立统计
- **By Difficulty**：按 easy / medium / hard 分层
- **By Category**：按 category 分组，展示最弱的 10 个

### 评测报告

Batch 模式运行后自动输出 JSON 报告；默认有 `--lora-dir` 时写入 `<lora-dir>/eval_report_<timestamp>.json`，没有 LoRA 时写入当前目录，包含：

- 时间戳、模型路径、LoRA 路径
- `evaluation_mode: raw_model_output` 和 `postprocess_applied: false`
- 总体指标 + per-file / per-difficulty / per-category 明细
- 所有错误样本（含 query、gt、pred、err_type）
- 解析后的工具名和参数保持模型原始输出，不做规则后处理修正
- `position` 为可选参数；用户未明确位置时不因缺少位置追问，直接省略 `position`，由工具侧按说话人位置补全
- 多意图样本默认不计入单工具指标；`eval.py` 会跳过 `intent/sub_category=多意图` 或包含多个 `expected_tool_calls` 的样本；传入 `--include-multi-tool` 后会计入并解析 JSON 数组多工具输出

### 评测数据

- 路径：`data/eval/*_test.json`（包含 `noise_history_test.json` 多轮无意义 query 边界专项集；多意图/多工具样本默认被单工具评测 mask）
- 音频：1598 条样本带 `query_audio` 字段；当前仓库 `data/eval/audio/` 下包含 1093 个 wav 文件
- 输入方式：有音频文件时自动用音频输入，无音频时回退到文本
- 支持字段：`expected_type`（显式指定 Action/Clarify/Reject）

## 脚本总览


| 脚本                                 | 用途                                                                                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_train_data.py`                | 合并 splits + 注入 SP → 训练集                                                                                                                                    |
| `train_thinker_lora.py`              | LoRA 训练，含冻结审计 + 训练指标记录                                                                                                                               |
| `train_teacher_sft_lora.py`          | Qwen3.5-27B teacher 的 SFT LoRA/QLoRA 训练；复用同一份 `data/train_final.jsonl` 和最终 assistant span 监督口径，默认冻结 vision 相关 LoRA 参数                     |
| `train_memory_dpo_lora.py`           | 从已有 SFT LoRA 初始化，基于 memory chosen/rejected 偏好数据继续做 DPO-style LoRA 训练                                                                             |
| `train_aut_asr_bridge.py`            | 实验脚本：用 eval 音频训练 Qwen AUT/audio_tower hidden states 到 Whisper decoder 的轻量 ASR bridge，默认保留 10% 验证集                                            |
| `serve.py`                           | **OpenAI 兼容推理服务**（FastAPI，支持文本+音频）                                                                                                                  |
| `eval.py`                            | 统一评测（batch / single），音频输入 + 多维度统计，支持`--batch-size` 批量推理                                                                                     |
| `scripts/analyze_eval_errors.py`     | 读取`eval_report*.json`，按类型、工具、文件、类别、参数槽位变化和混淆对聚类错误；可用 `--backlog-md` 输出训练补强任务清单，或用 `--review-html` 生成逐 case 人工复核页面 |
| `scripts/build_boundary_preferences_from_eval.py` | 从评估报告中抽取 `NoiseDoNotAct` 过召和 false-reject case，生成当轮 DPO 边界修复偏好数据 |
| `scripts/schema_coverage_report.py`  | 统计 SFT / eval / RL 的工具调用、参数枚举和完整参数组合覆盖，定位 eval/RL 中有但 SFT 弱覆盖或缺失的 schema 组合                                                    |
| `scripts/validate_splits.py`         | 校验 split 样本消息结构、工具调用和响应形态                                                                                                                        |
| `scripts/validate_by_tool_schema.py` | 校验`data/splits/by_tool/*.jsonl` 是否符合 `data/tools.json` schema                                                                                                |
| `scripts/validate_rl_schema.py`      | 校验`data/rl` 中稳定 RL 训练数据的 chosen/rejected/expected 工具调用是否符合当前 `data/tools.json` schema；候选和审计 artifact 可用 `--include-artifacts` 额外扫描 |
| `scripts/generate_train_report.py`   | 从`train_metrics.jsonl` 生成 HTML 训练可视化报告                                                                                                                   |
| `scripts/gradio_remote_infer.py`     | 远端推理服务的 Gradio 调试界面                                                                                                                                     |
| `scripts/probe_asr_decoder.py`       | 实验脚本：hook Qwen2.5-Omni`thinker.audio_tower` hidden states，尝试交给 Whisper decoder 解码 ASR；用于验证 AUT 表征是否可直接转写，不是稳定线上入口               |

AUT ASR probe 可单条音频运行，也可批量扫描 eval 文件并输出 JSONL：

```bash
python scripts/probe_asr_decoder.py \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --whisper-dir openai/whisper-large-v3 \
  --audio data/eval/audio/window/window_001.wav

python scripts/probe_asr_decoder.py \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --whisper-dir openai/whisper-large-v3 \
  --eval-file data/eval/window_test.json \
  --limit 20 \
  --output data/serve_logs/aut_asr_probe_window.jsonl
```

AUT ASR bridge 训练使用 eval 中带 `query_audio` 的音频样本构造 ASR 监督数据，默认按 seed 保留 10% validation，只训练 bridge，Qwen AUT 和 Whisper 均冻结：

```bash
python train_aut_asr_bridge.py \
  --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --whisper-dir openai/whisper-large-v3 \
  --eval-dir data/eval \
  --output-dir aut_asr_bridge_output \
  --epochs 3 \
  --grad-accum 8 \
  --bridge-dtype float32
```

评测后的推荐排查顺序：

```bash
python scripts/analyze_eval_errors.py lora_output/eval_report_<timestamp>.json --limit 20
python scripts/analyze_eval_errors.py lora_output/eval_report_<timestamp>.json --limit 20 --backlog-md docs/eval-error-training-backlog.md
python scripts/analyze_eval_errors.py lora_output/eval_report_<timestamp>.json --review-html docs/eval-error-review.html
python scripts/schema_coverage_report.py --output-md docs/schema-coverage-report.md --limit 50
python scripts/schema_coverage_report.py --backlog-md docs/schema-coverage-hardcase-backlog.md --limit 80
```

该流程用于决定下一轮补数据方向；不通过规则后处理抬高线上或评测准确率。

## 已完成的优化

以下问题在历史迭代中已修复：

- [X]  max_length 1024 → 16384（防止样本截断）
- [X]  last-assistant-only loss masking（仅监督每条样本最后一个 assistant 回复）
- [X]  SP 压缩并统一到 `data/system-prompt.txt`（当前约 5.8K chars）
- [X]  SP 统一管理（`data/system-prompt.txt`，训练/推理/评测共用）
- [X]  训练数据不再内嵌 SP，由 build_train_data.py 构建时注入
- [X]  lr 1e-4 → 2e-5，alpha 32 → 16，添加 warmup/weight_decay/grad_clip
- [X]  load_best_model_at_end，按 eval_token_acc 选最优 checkpoint
- [X]  gradient_checkpointing，batch=1 + grad_accum=8（24GB 显存适配）
- [X]  过滤空监督样本，避免全 `-100` label batch 污染 `eval_loss`
- [X]  Reject 数据增强（103 条硬负例：家电混淆、多轮拒绝、跨域请求）
- [X]  分类逻辑按最后一条 assistant turn 判断（多轮样本正确分类）
- [X]  冻结审计自动化（forbidden keyword → auto-freeze → fail-fast）
- [X]  R4 数据增强（+358 条：修复过度-Clarify、补齐弱工具媒体/电话/信息）
- [X]  eval.py `--batch-size` 批量推理（单 GPU 利用率从 ~30% → ~75%）
- [X]  R5 数据增强（+164 条：position 字段覆盖、抗过度-Clarify、Climate/Light 多样性）
- [X]  R6 anti-Clarify 清理：位置缺失不追问，口语意图/信息查询/电话与 FM 搜索直接 Action
- [X]  数据质量清洗（2026-05-10）：修正 clarify 中可选 position 误追问、reject 误标工具样本，并将问候/感叹词/道别/闲聊等无意图噪声迁移至 noise；一次性清洗脚本已删除，保留清洗后的 split 产物
- [X]  训练过程指标持久化：`MetricsSaverCallback` 每隔 `--logging-steps` 步追加写入 `{output_dir}/train_metrics.jsonl`（含 step/epoch/loss/lr/grad_norm/eval_loss/eval_token_acc）
- [X]  HTML 训练报告生成：`scripts/generate_train_report.py` 从 `train_metrics.jsonl` 生成暗色主题交互式 HTML（Chart.js），包含 train loss、eval loss + token acc 双轴、学习率、grad norm 四图及关键统计卡片

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

- [ ]  按 `scripts/analyze_eval_errors.py --backlog-md` 的 P0/P1 错误族继续补充非评估原句 hard case，并用 `build_train_data.py --sample-weight` 加权训练
- [ ]  补充 Clarify 评测数据（当前 0 条，训练集有 ~151 条）
- [ ]  补充 Reject 评测数据（当前 1 条，训练集有 ~1132 条）
- [ ]  补齐 9 个无覆盖工具的测试数据（GeneralBack/Exit/Select、NavigationControl 等）
- [ ]  工具混淆问题（雨刮→ClimateControl、播放→MediaControl vs MusicSearchPlay）
- [ ]  阶段 B：DPO/ORPO 定向提准
- [ ]  导出部署：合并 LoRA → ONNX/GGUF
