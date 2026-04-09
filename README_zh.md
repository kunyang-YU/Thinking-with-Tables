[English](README.md) | **中文**

# Thinking with Tables (TWT)

> **Thinking with Tables: Enhancing Multi-Modal Tabular Understanding via Neuro-Symbolic Reasoning**  
> 基于代码的神经符号推理框架，面向表格-视觉多模态理解（TVMU）任务。

按照论文中的定义，TVMU（Tabular-Vision Multi-Modal Understanding）旨在从文档图像或扫描件中解析表格结构，联合利用表格相关的视觉信息与文本上下文，完成表格解析、信息抽取和复杂数值/逻辑推理，主要涵盖**多模态表问答**和**多模态表预测**两大类任务。  
**Specifically**，在本代码仓库中：  

- **多模态表问答**：只向模型提供表头/表片段图像 + 相关文本描述 + 外部环境中的原始表格文件路径，模型需通过生成代码在沙箱中读取并操作完整表格，完成查表、聚合、计算等推理；  
- **多模态表预测**：提供样本图像 + 元数据/任务描述 + 表格/CSV 路径，模型需同时建模表格特征与图像特征，并通过代码在沙箱中完成特征工程与预测。

本 README 主要说明 **TWT 模型的推理流程与使用方法**。

---

## Quick Start

### 1.1 Clone the Repository

```bash
git clone https://github.com/kunyang-YU/Thinking-with-Tables.git
```

### 1.2 Environment Setup & Dependency Installation

建议使用 Conda 创建独立环境并安装依赖：

```bash
conda create -n TwT python=3.10 -y
conda activate TwT

pip install torch torchvision -U
pip install openai tqdm pandas scikit-learn -U

pip install git+https://github.com/modelscope/ms-swift.git -U

pip install "deepspeed<0.17" -U
pip install flash-attn --no-build-isolation --use-pep517
```

**说明**：表预测任务需在沙箱中调用 `ResNet50Predict`，需准备对应任务的 ResNet50 权重，并将 `tools.py` 放在沙箱可导入的路径下（见下文「沙箱执行方式」）。

---

## 2. 推理机制概述

TWT 采用 **程序辅助的神经符号推理**：模型通过与环境（沙箱）的**多轮交互**完成表格理解与问答/预测。

### 2.1 输出协议

模型必须按以下格式输出，便于解析与沙箱对接：


| 标签                               | 含义                     |
| -------------------------------- | ---------------------- |
| `<analy>...</analy>`             | 推理过程与问题分析              |
| `<code>...</code>`               | 需要在沙箱中执行的 Python 代码    |
| `<code_result>...</code_result>` | 沙箱返回的执行结果（由系统注入，模型不生成） |
| `<answer>...</answer>`           | 最终答案                   |


### 2.2 推理循环（单条样本）

```
┌─────────────────────────────────────────────────────────────────┐
│  输入: 图像(表头/样本) + 问题/任务说明 + 表格/数据路径等            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  模型生成 → 若出现 <code>，则抽取代码 → 沙箱执行 → 得到            │
│  <code_result> → 拼回对话上下文 → 再次调用模型 …                  │
│  直到响应中出现 <answer>，解析得到最终答案。                        │
└─────────────────────────────────────────────────────────────────┘
```

即：**每轮**模型可输出 `<analy>` + `<code>`；系统执行代码并把结果以 `<code_result>` 形式反馈给模型；重复直到模型输出 `<answer>`。

### 2.3 沙箱执行方式

本仓库已准备**论文中各数据集推理所需工具**（如表格数据路径、ResNet50 权重与 `tools.py` 等），可直接用于复现实验。

代码在**受控沙箱**中执行（超时、持久化 `env`），核心逻辑等价于：

- **表 QA**：仅需能执行 pandas 等读表、计算的 Python 环境。
- **表预测**：沙箱需能 `from tools import ResNet50Predict`，并在执行前把 `tools.py` 所在目录加入 `sys.path`（或等价方式）；`ResNet50Predict(task, image_path)` 的 `task` 与权重路径需与当前任务一致（如 `pawpularity`、`adoption`、`skinca`、`paintings`）。

项目内沙箱实现位置（逻辑可复用）：

- **表 QA**：
- **表预测**：

---

## 3. 训练过程（SFT 与 GRPO）

TWT 采用论文中的两阶段训练：**Task-Oriented SFT（TO-SFT）** 与 **Adaptive Loss-Scaled GRPO（AL-GRPO）**，均在 Swift 框架下完成。

### 3.1 Task-Oriented SFT（TO-SFT）

**做法**：对基座多模态模型（如 Qwen3-VL-8B）进行任务导向的监督微调，使模型学会按 `<analy>` / `<code>` / `<answer>` 格式输出，并掌握表 QA 与表预测两类任务的基本求解流程。训练时对代码执行结果（`<code_result>`）做 **mask**，只对模型生成的文本与代码序列计算 loss，避免记忆执行结果。

**数据集结构**：每条样本为单轮对话，格式与 Swift 多模态 SFT 一致。示例字段：

数据来源包括：表 QA 合成数据（如 `tableqa-python-1interaction.json`、`full-table-inter.json`）、多模态表预测数据（pawpularity、adoption、paintings、skinca 等 JSON）。数据合成流程见论文与附录（大模型生成推理轨迹 + 答案校验）。

```json
{
  "question": "<image>\n\n### **Question**\nwhat is the percentage of women spoke with family members in the past 12 months of 2018?\n\n### Table file path: /path/to/67.csv\n",
  "response": "<analy>...</analy>\n<code>import pandas as pd\n...</code>\n<code_result>...</code_result>\n<analy>...</analy>\n<answer>44.0</answer>\n",
  "image": ["/path/to/67.png"]
}
```

**代码实现位置**：

- Swift SFT 入口：`swift sft`（ms-swift 提供）；
- 本仓库 SFT 脚本：`src/sft1.sh`；
- 系统 prompt（输出格式说明）：`src/prompt.txt`；

**调用脚本示例**：

```bash
cd src
bash sft1.sh
```

`sft1.sh` 中主要参数：`--model` 基座模型路径，`--dataset` 多个 JSON 路径（表 QA + 表预测），`--system` 指向 `prompt.txt`，`--train_type full`，`--freeze_vit true`，`--num_train_epochs 3`，`--learning_rate 1e-5`，`--deepspeed zero3` 等。按需调整 `CUDA_VISIBLE_DEVICES`、`per_device_train_batch_size`、`gradient_accumulation_steps`。

### 3.2 Adaptive Loss-Scaled GRPO（AL-GRPO）

**做法**：在 TO-SFT 得到的 checkpoint 上做 GRPO 强化学习。模型按 prompt 生成多轮回复（含 `<code>`）；通过 **TWT 多轮调度器** 在每轮生成后抽取代码、在沙箱中执行、将 `<code_result>` 拼回上下文并继续生成，直到出现 `<answer>`。奖励基于**最终答案正确性**（如 QA 的匹配规则、分类的 0/1、回归的连续 reward）；采用 **adaptive loss-scaled** 策略：仅对**执行成功**的代码段参与 GRPO loss，执行失败的不回传梯度，以提升代码稳定性并避免 reward hacking。

**数据集结构**：每条样本为 **messages + 图像路径 + solution（GT 答案）+ type（任务类型）**，用于 reward 计算与多轮 rollout。

**字段示例（表 QA）**：


| 字段         | 含义                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `messages` | `[{ "role": "system", "content": "输出格式说明" }, { "role": "user", "content": "### **Image** <image> ... ### **Question** ... ### Table file path: ..." }]` |
| `images`   | 图片路径列表，与 user 中 `<image>` 对应                                                                                                                            |
| `solution` | 标准答案（用于 reward）                                                                                                                                         |
| `type`     | 任务类型：`TAT-QA` / `FinQA` / `HiTab` / `CLS` / `REG`，决定 twtplugin 中评估方式                                                                                    |
| `question` | 问题原文（可选，便于查看）                                                                                                                                           |


```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant. ... <analy>...</analy> <code>...</code> <answer>...</answer>" },
    { "role": "user", "content": "### **Image**\n<image>\n\n### **Question**\nHow much is the 2019 rate of inflation?\n\n### Table file path: /path/to/1.csv\n" }
  ],
  "images": ["/path/to/images/1.png"],
  "solution": "2.9",
  "type": "TAT-QA",
  "question": "How much is the 2019 rate of inflation?"
}
```

**字段示例（表预测）**：


| 字段         | 含义                                                                  |
| ---------- | ------------------------------------------------------------------- |
| `messages` | system + user（user 中含 `<image>`、任务说明、`image path:`、`tabular path:`） |
| `images`   | 样本图路径列表                                                             |
| `solution` | 回归值或分类标签（字符串，如 `"3"`、`"33"`）                                        |
| `type`     | `REG`（回归）或 `CLS`（分类）                                                |


```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant. ... <answer>...</answer>" },
    { "role": "user", "content": "## Question\n\nYou are given an image related to pet pawpularity. ... The image is <image>\n\n## Instructions\n... ResNet50Predict(task, image_path) ...\n## Files\n\nimage path: /path/to/0c282c8a762452f9110e1f88c1ed01a6.jpg\ntabular path: /path/to/train.csv\n" }
  ],
  "images": ["/path/to/pawpularity/dataset/images/0c282c8a762452f9110e1f88c1ed01a6.jpg"],
  "solution": "3",
  "type": "REG"
}
```

表预测数据与上述示例结构一致，只需改动 `solution` 和 `type` 即可适配不同任务。

**代码实现位置**：

- GRPO 入口：`src/rl/myrlhf/ms-swift/scripts/rl.sh`（内部调用 `scripts/rlhf_ds.py`，等价于 `swift rlhf --rlhf_type grpo`）；
- TWT 插件（多轮调度 + 奖励格式）：`src/rl/myrlhf/ms-swift/examples/train/grpo/plugin/twtplugin.py`  
  - `TwTMulltiTurn`：多轮调度，抽取 `<code>`、调用 `_repl_code_run`、将 `<code_result>` 写回 `infer_request.messages`；  
  - `CodeFormat`（注册为 `re_format`）：从生成内容中解析 `<answer>`，按 `type` 调用对应评估函数（TAT-QA / FinQA / HiTab / CLS / REG），返回 reward；
- 沙箱与 ResNet50：`twtplugin.py` 内 `_repl_code_run`；表预测需在运行环境中可导入 `tools.ResNet50Predict`，见 `src/rl/myrlhf/ms-swift/examples/train/grpo/plugin/tools.py`；
- 自适应 loss-scale：`--loss_scale 'default+code'` 由 `swift/plugin/loss_scale/loss_scale.py` 中的 `get_loss_scale` 解析，实现“只对执行成功代码段记梯度”的策略；
- 调用脚本：`src/rl/myrlhf/ms-swift/scripts/rl.sh`。

**调用脚本示例**：

```bash
cd src/rl/myrlhf/ms-swift
# 1）先启动 vLLM 服务，加载 SFT 产出的 checkpoint（见脚本内注释或 vllm.sh）
# 2）修改 scripts/rl.sh 中的 --model、--external_plugins、--dataset、--output_dir、--resume_from_checkpoint 等参数
bash scripts/rl.sh
```

`rl.sh` 关键参数：`--rlhf_type grpo`，`--external_plugins` 指向 `twtplugin.py`，`--reward_funcs re_format`，`--multi_turn_scheduler twt_scheduler`（由插件注册），`--dataset` 为多个 RL 用 JSON（如 `data/RL_data/WTQdata/` 与 `data/RL_data/MMData/` 下各 `train.json`），`--stop_words '</code>' '</answer>'`，`--loss_scale 'default+code'`（adaptive loss-scaled），`--num_generations 4`，`--learning_rate 5e-7` 等。训练前需保证 vLLM 已加载正确模型并在 `--vllm_server_port` 可访问。

论文中使用的全部训练数据将随代码一并开源：SFT 阶段约 **1.5K 条表 QA + 1.2K 条表预测 + 5.0K额外数据**，GRPO 阶段约 **0.5K 条表 QA + 0.4K 条表预测**。

---

## 4. 模型部署（推理服务）

### 4.1 使用 vLLM 启动 OpenAI 兼容接口

```bash
# 示例：vllm.sh
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/TWT\
    --served-model-name twt \
    --host 127.0.0.1 \
    --port 8160 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

---

## 5. 运行推理/评估

在运行评估脚本前，需要确保 **表格 CSV、图片、`tools.py`、ResNet50 权重等资源路径都已按本机环境正确填写**，以保证沙盒环境能够正常访问所有文件。

### 5.1 表问答（Table QA，以 TAT-QA 为例）

1. 启动 vLLM 服务（见上文），记下 `--port`（如 8160）。
2. 准备数据：含 `question`、`paras`（可选）、表格 CSV 路径、对应图片路径等，格式与 `evaluation/TAT-QA/test.json` 及脚本内路径一致。
3. 运行评估脚本（需根据实际路径修改脚本内 `file_path`、`path`、`image_path`）：

```bash
cd evaluation/TAT-QA
python evaluation_tat.py --exp my_run --model_name twt --port 8160
```

脚本会：

- 读入每条样本的图片与问题；
- 按论文中的 prompt 构造「图像 + 表格路径 + 说明」的 user 消息；
- 循环：调用本地 OpenAI 客户端（`base_url=http://127.0.0.1:{port}/v1`），流式接收回复 → 用正则抽取 `<code>...</code>` → 调用 `sandbox_run(env, code)` → 将结果以 `<code_result>` 追加到上下文并再次请求，直到回复中出现 `<answer>`；
- 从 `<answer>` 中解析答案并写入 `output_{EXP_NAME}_{MODEL_NAME}.json`。

### 4.2 表预测（Table Prediction，如 Pawpularity / Adoption / SkinCA / Paintings）

以 Pawpularity 为例，脚本在 `evaluation/multi-modal-evaluation/pawpularity/` 下：

1. 数据：JSON 中每条包含 `image_path`、`instructions`（任务说明与表格/数据路径）、`answer`（GT）等。
2. 与表 QA 相同：用 OpenAI 兼容接口 + 流式接收；每轮抽取 `<code>`，在沙箱中执行前**先拼接** `from tools import ResNet50Predict` 再执行（与 `twtplugin.py` 中一致）。
3. 沙箱需能导入 `tools`，且 `ResNet50Predict` 内权重路径与当前任务匹配（如 `pawpularity` / `adoption` / `skinca` / `paintings`）。
4. 循环直到出现 `<answer>`，解析数值答案并保存。

运行前请：

- 在对应任务目录下准备好 `tools.py`（或确保其路径在 `sys.path` 中）；
- 修改脚本内数据路径（如 `file_path`）和端口等参数。

```bash
cd evaluation/multi-modal-evaluation/pawpularity
# 按脚本内参数修改 --exp, --model_name, --port
python evaluation-paw.py --exp paw_run --model_name twt --port 8160
```

其他预测任务（Adoption、SkinCA、Paintings）结构类似，仅数据与 `ResNet50Predict` 的 `task`/权重不同。

---

## 6. 输入/输出与数据格式小结


| 任务类型                                | 输入                       | 沙箱要求                                               | 答案形式                     |
| ----------------------------------- | ------------------------ | -------------------------------------------------- | ------------------------ |
| 表 QA (WikiTQ/TAT-QA/FinQA/TabMWP 等) | 表头/样本图 + 问题 + CSV 路径     | pandas、常规 Python                                   | 文本/数值，从 `<answer>` 解析    |
| 表预测 (分类/回归)                         | 样本图 + 任务说明 + 表格路径 + 图片路径 | pandas、sklearn、`ResNet50Predict(task, image_path)` | 分类标签或回归值，从 `<answer>` 解析 |


评估时**不直接给模型完整表格内容**，只给表头或样本图与路径，表格数据需由模型通过生成的代码在沙箱中读取。

---

## 8. 引用

若使用本代码或 TWT 方法，请引用论文：

```bibtex
@article{yu2026thinking,
  title={Thinking with Tables: Enhancing Multi-Modal Tabular Understanding via Neuro-Symbolic Reasoning},
  author={Yu, Kun-Yang and Zhou, Zhi and Tian, Shi-Yu and Yang, Xiao-Wen and Jia, Zi-Yi and Yang, Ming and Cheng, Zi-Jian and Guo, Lan-Zhe and Li, Yu-Feng},
  journal={arXiv preprint arXiv:2603.24004},
  year={2026}
}
```

---

**总结**：TWT 推理 = **多轮「模型生成 → 抽代码 → 沙箱执行 → 结果填回上下文」** 直到得到 `<answer>`；表 QA 只需通用 Python 沙箱，表预测还需在沙箱中提供 `ResNet50Predict` 与对应权重。按上述步骤部署模型并运行对应 `evaluation/`* 脚本即可复现论文中的推理流程。

---
If you have any questions regarding file paths or testing, please feel free to open an issue or contact yuky@lamda.nju.edu.cn by email.
