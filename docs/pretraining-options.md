# 开源框架上的预训练项目候选

2026 年 11 月 11 日整理。面向已有 CS336 模型实现、预训练与 SFT 基础，希望进一步形成实习项目的学习者。项目当前处于方案阶段。

## 开源框架选择

| 框架 | 适合承担的工作 | 使用时需要核对的内容 |
| --- | --- | --- |
| nanochat | 可读、可修改的从零预训练与完整训练流程 | 官方 speedrun 按 8×H100 配置，个人硬件需要缩小模型、batch 和训练规模，FP8 路径也有硬件要求 |
| LitGPT | 小模型从零训练与已有基模继续预训练 | 明确加载随机权重还是 initial checkpoint，固定 tokenizer、数据格式与模型配置 |
| Nanotron + SmolLM 配方 | 数据混合、多阶段预训练和并行训练 | 官方完整配置的训练量很大，应重建小规模配置，核对模型架构支持与检查点格式 |
| TorchTitan | PyTorch 分布式训练与性能分析 | 先采用小模型配置，再验证 FSDP、激活检查点和数据恢复，按实际卡型选择精度与内核 |
| Megatron-LM | 更深入的训练系统与模型并行扩展 | 初始化环境、架构与并行配置更复杂，需要具体的系统研究问题 |

官方入口见 [nanochat](https://github.com/karpathy/nanochat)、[LitGPT](https://github.com/Lightning-AI/litgpt)、[LitGPT 预训练说明](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md)、[Nanotron](https://github.com/huggingface/nanotron)、[SmolLM 预训练配方](https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md)、[TorchTitan](https://github.com/pytorch/torchtitan)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。

## A. 小模型预训练的数据质量与配比优化

项目研究固定 token 预算下，数据处理与通用/数学语料比例对小模型学习效果的影响。

采用约 100M–360M 的模型随机初始化，先用 LitGPT 或 nanochat 跑通训练，再按需要使用 Nanotron 的配方与并行能力。先做约 1B token 的筛选实验，正式实验可扩展到每组 2B–5B token。模型参数量以实际配置统计为准。

数据处理采用 DataTrove，保留来源、质量分数、重复簇和文档长度信息。近重复簇保持在同一数据划分。先固定 tokenizer，只改变数据处理或混合方式。

可以分别进行两个控制变量实验。质量实验从可比来源的文本池中比较随机抽样与质量筛选，保持训练 token 相同。配比实验使用同一批已处理数据，比较固定比例和分阶段提高数学语料比例。

候选公开数据包括 FineWeb-Edu 与 FineMath。使用已经筛选的数据时，记录上游处理，并明确本项目新增的跨源去重、抽样和配比实验。[DataTrove](https://github.com/huggingface/datatrove) · [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) · [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)

主要评测为通用与数学 held-out loss、适合小模型的下游任务、tokens/s、数据处理速度与训练成本。比较不同 tokenizer 时使用共同原始文本上的 bits-per-byte 等可比指标，同 tokenizer 的模型可以报告 PPL。增加相同 SFT 数据的小规模下游实验，观察基座差异能否延续。

项目成果包括数据处理流水线、来源统计、随机初始化训练配置、各组检查点、消融报告和模型卡。简历名称可为“小型语言模型预训练与数据配方优化”。

粗略预算为 **300–1,500 元**，按小模型、少数数据配方对照及基础评测估算。单 4090 或同实例双卡可作为候选，正式规模由吞吐和显存试跑确定。

## B. 数学领域继续预训练与后训练协同

项目研究已有基模经过数学文本继续预训练后，其知识建模和后续 SFT/GRPO 效果如何变化。

采用 Qwen3-1.7B-Base 等 Base checkpoint，先用较小规模做验证。LitGPT 已列出 Qwen3 支持，并提供 initial_checkpoint_dir 对应的继续预训练流程。实际实现需核对选定模型的配置与权重转换。[Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) · [LitGPT](https://github.com/Lightning-AI/litgpt)

使用数学原始文本与通用文本做 next-token prediction。可以比较纯数学语料、数学与通用文本混合、分阶段混合。将公开评测题与训练语料交叉去重，保留独立调参验证集。最终评测时固定提示模板与生成预算。

项目先完成领域继续预训练的基线对照，再与已学的 SFT/GRPO 连接。

| 分支 | 作用 |
| --- | --- |
| Base → SFT | 下游基线 |
| Base → CPT → SFT | 检查继续预训练的增量 |
| Base → SFT → GRPO | 直接后训练基线 |
| Base → CPT → SFT → GRPO | 检查继续预训练与 RL 的结合 |

SFT 和 GRPO 的数据、超参数与预算按分支对齐，同时单列 CPT 的额外计算量。基模自身已经具备能力，结果需要相对于加载的原始 checkpoint 衡量。使用全参数更新或 LoRA 继续预训练时明确标注。

核心指标为数学与通用验证 loss、数学题正确率、通用任务变化，以及固定采样预算下 RL 组内全对全错比例。改进点可集中在通用语料回放比例与遗忘控制。

简历名称可为“面向数学推理的领域继续预训练与强化学习协同优化”。该方向能连接现有 CS336 与 Agentic RL 学习路线。

粗略预算为 **500–2,500 元**，覆盖较小规模 CPT、少数回放配比及后训练对照。1.7B 全参数训练的显存明显高于权重本身，按大显存单卡或适当分片的同实例多卡方案验证。

## C. 面向消费级 GPU 的预训练效率与恢复能力优化

项目研究在固定模型、序列长度和有效 batch 下，怎样提高训练吞吐并降低显存，同时验证检查点恢复的正确性。

选择 TorchTitan 或 Nanotron，建立单卡与同实例双卡的小模型基线。先定位数据加载、前向反向、通信、编译或保存中的主要耗时，再选择一个瓶颈深入处理。

实验包括数据 packing 与预分词、微批次与梯度累积、激活检查点和合适的分布式策略。保持处理的有效 token 数与训练目标一致，记录 packing 后的 attention 边界与文档边界。分别报告编译时间、稳态吞吐和端到端用时。

检查断点续训恢复的模型、优化器、学习率、随机状态与 dataloader 状态。验证固定种子下的数据次序与后续 loss，明确哪些操作存在非确定性。该工作可形成性能回归脚本和可复查的恢复测试。

主要指标为有效 tokens/s、峰值显存、端到端训练时间和多卡加速比。MFU 需要明确 FLOPs 估算及 GPU 峰值精度口径。采用框架已有优化时说明配置与评测贡献，新增代码以实质瓶颈为依据。

简历名称可为“基于 TorchTitan 的小模型分布式预训练与性能分析”。更适合 AI Infra、训练平台方向。

粗略预算为 **300–1,500 元**，主要用于性能矩阵与有限训练验证。大规模模型并行和更多卡数的扩展另行估算。

## 建议顺序

希望延伸 CS336 的从零训练经历，可以优先 A。希望尽快把预训练与已学的 RL 连成一个主项目，可以优先 B。希望投递 AI Infra 或训练系统方向，可以优先 C。

以上费用是工程估计，按缩小后的模型、语料与有限对照实验考虑，包含基本调试与评测余量。完整复现 SmolLM 等开源模型的训练规模，需要独立的资源规划。当前尚未执行 GPU 训练和测量。
