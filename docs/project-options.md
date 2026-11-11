# 后训练实习项目候选

2026 年 11 月 11 日整理。选题覆盖 Agentic RL Lab 前十个实验入口中的知识，结合 CS336、小模型实现、预训练和 SFT 基础，重点补充实际训练、控制变量实验、评测和错误分析。

## 知识与选题映射

| 已读内容 | 可形成的项目能力 |
| --- | --- |
| Loss Functions、GRPO、DAPO、GSPO | 策略更新、奖励设计、训练稳定性和采样效率 |
| General OPD、Medical OPD | 学生采样、教师评分、领域与通用能力平衡 |
| OPSD | 同模型不同上下文的自蒸馏与参考解答利用 |
| AgentOPSD | 多轮环境交互、动作 token 对齐和信用分配 |
| PyTrio-Jev | 分类决策、候选概率、温度校准和工具路由 |
| Vision GRPO | 图文输入、可验证奖励、视觉依赖与鲁棒性评测 |

来源见 [教程实验目录](https://datawhalechina.github.io/agentic-rl-lab/)。阅读经历先转化为可运行实现，再将实际完成的内容整理进简历。

## 2026 年面经对应关系

10 月 21 日的百度算法面经公开摘要提到 Agentic-GRPO、奖励维度、训练规模与数据质量。9 月的小米汽车面经提到环境搭建、模型工具交互、奖励作弊、模型选型与评测。2 月的小米日常实习面经涉及 SFT/GRPO、OOM 和量化。它们为项目需要留存的数据、配置、故障和评测材料提供了具体追问方向。面经是作者自述，整理帖中的技术问法也需要对照原理核验。[百度面经](https://www.nowcoder.com/feed/main/detail/ea456eb7238c4e44b099358ee6b5add8?sourceSSR=dynamic) · [小米汽车面经](https://api-cdn.nowcoder.com/feed/main/detail/79f1e686a73d4e1b806c5927b6125771?sourceSSR=enterprise) · [小米日常实习面经](https://www.nowcoder.com/feed/main/detail/69d56315acea489c8afcdb65d754147b)

9 月的牛客 OPD 推荐涵盖多教师、Agent 蒸馏和自蒸馏等方向，可作为扩展选题入口。具体资源需求以实际选用仓库核对。[OPD 推荐](https://www.nowcoder.com/discuss/933127343101476864?sourceSSR=users)

## A. 面向数学推理的小模型强化学习与自蒸馏

**核心问题**是如何在有限生成预算下提高小模型的答题正确率，并解释采样难度、输出长度和训练目标的影响。

建议以 1.7B–4B 的 Qwen 为起点，先做 SFT 冷启动，再完成 GRPO。训练选用 GSM8K 或数学推理数据的适当难度子集，评测保留独立题目，并检查与训练语料的交叉重复。

第二阶段只选一个扩展分支。算法分支比较 GRPO、GSPO 与明确实现的动态采样改动。自蒸馏分支比较同一冷启动后的 GRPO 与 OPSD，使教师在训练时借助参考解答评价学生轨迹，学生评测时只看题目。

个人改进可从同题组全对全错比例、难度分层采样或长度控制中选择一个。记录额外筛选的生成开销，使用相同训练 token 或 GPU 时间预算比较。DAPO 的完整方法包含多项设计，实验中应逐项说明实现范围。

主要指标包括正确率、固定采样预算下的 pass@k、答案可解析率、输出长度、退化组比例和吞吐。验证正确率提升是否主要来自格式修复，并人工检查一部分中间推理。

开源基础包括 [Open R1](https://github.com/huggingface/open-r1)、[OPSD](https://github.com/siyan-zhao/OPSD)、[OPSD 数据](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd) 和 [教程 GRPO](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo)。本地训练可参考 TRL 与 Unsloth 的相应实现。

粗略预算为 **300–1,000 元**，约需 2–4 周。候选算力为单卡 LoRA、小规模双 4090 或单张大显存卡，模型长度和采样量由预实验确定。

完成后可用的项目名称为“面向数学推理的小模型强化学习与采样优化”。简历写清同一模型起点、一个已验证改进和独立评测结果。

## B. 领域知识增强与通用能力保持的在线蒸馏

**核心问题**是领域训练提升专项能力时，怎样保留原有通用能力。

可以延续教程的医学考试问答场景。用领域训练得到的模型 checkpoint 作为领域教师，原始 checkpoint 作为通用教师，学生在自己的回答轨迹上接受教师评分。采用同架构、同 tokenizer 的教师与学生，可以先专注算法机制。

先建立 Base、领域 SFT、单教师 OPD、固定比例双教师 OPD 的对照。个人改进可以采用训练内部验证集驱动的混合比例调度，当通用验证指标下降时提高通用样本比例，恢复后逐步增加领域训练。超参数和调度都在验证集决定，最终测试集保持独立。

记录领域正确率、通用正确率、遗忘幅度、每轮生成量和教师评分成本。若使用更大教师，逐项验证 tokenizer、词表和实际学生 token 的评分对齐。教师离线生成答案后做 SFT 的实验单独标记。

开源基础包括 [教程 General OPD](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/general-opd)、[教程 Medical OPD](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/medical-opd)、[TRL 在线蒸馏](https://huggingface.co/docs/trl/main/distillation_trainer) 和 [领域 SFT 数据](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)。

粗略预算为 **500–2,000 元**，约需 3–5 周。同尺寸教师可以共用基模、轮换适配器或分阶段评分，较大教师需要重新评估显存和评分吞吐。

完成后可用的项目名称为“兼顾领域知识与通用能力的双教师在线蒸馏”。核心贡献应落在自己的数据划分、训练调度和遗忘对照实验上。

## C. 基于技能自蒸馏的多轮任务 Agent

**核心问题**是成功或失败的长轨迹中，哪些动作最值得加强学习。

在 text-only ALFWorld 中完成找物、移动、放置等任务。先评测原始模型，再建立终局奖励训练基线，随后加入 AgentOPSD。学生无技能执行，当前模型的同参数快照带技能评价同一条轨迹，将终局奖励与局部 turn 信号结合。

个人改进可以选择失败状态下的技能选择，或技能证据的权重限制。比较通用技能、任务相关技能与随机技能，观察局部提示的相关性。再比较整条轨迹统一优势与 turn-level 信用分配，其他设置一致。

关键工程工作是保存真实动作 token span、屏蔽 observation 的 loss、校验旧策略快照、限制无效循环，以及实现 seen/unseen 固定评测。推理时学生是否带技能需要分别报告。训练期 Self-Teacher 的特殊上下文与评测输入清楚区分。

指标包括任务成功率、无效动作次数、完成步数、不同任务类型表现和未见任务泛化。它更适合展示 Agent 算法能力，需预留环境与长序列调试时间。

开源基础包括 [AgentOPSD 上游](https://github.com/ZethWang/AgentOPSD)、[教程可运行实现](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/agentopsd) 和 [ALFWorld](https://github.com/alfworld/alfworld)。

粗略预算为 **1,000–3,000 元**，约需 4–6 周，按 1–4B LoRA 和受限轨迹长度估算。先验证单张大显存卡或同实例多卡配置。原论文的长轨迹和训练规模需要另行核算。

完成后可用的项目名称为“基于技能自蒸馏与轮次信用分配的多轮任务 Agent”。

## D. 可校准的小模型工具路由器

**核心问题**是小模型能否可靠决定直接回答、查询资料、调用计算器或追问，并在低置信度时交给更强模型。

从 0.6B–1.7B 的 Qwen 文本模型开始，也可沿用教程 4B。把请求、当前状态和候选动作输入模型，在选项位置计算受限 softmax，使用分类监督训练，再在独立校准集上做温度校准。

个人实现包括路由数据构建、工具标签核验、低置信度回退和错误调用归因。ToolACE 的工具调用记录可以提供数据原料，路由任务需要额外映射标签并抽检。按工具或工作流隔离数据，增加不该调用工具、缺少必要信息和未见工作流的样本。

对照包括普通生成式路由、候选概率分类、校准后分类、阈值回退。报告路由准确率、Macro-F1、ECE、覆盖率与准确率的关系、任务最终成功率和端到端延迟。准确率和置信度可信程度分开衡量。

开源基础包括 [Intern-Decision](https://github.com/InternLM/Intern-Decision)、[教程 Jev](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/pytrio-jev)、[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) 和 [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)。

粗略预算为 **100–400 元**，约需 1–2 周，单 4090 作为优先验证配置。大模型回退如使用付费 API，费用另计并单独报告。

完成后可用的项目名称为“面向工具调用的轻量决策模型与置信度校准”。它适合与一个后训练主项目配合，形成训练与应用两个侧面的经历。

## E. 面向几何问答的视觉推理后训练与鲁棒性评测

**核心问题**是视觉 RL 带来的变化有多少来自答案格式、选项偏好或真正使用图片。

可沿用 Qwen3.5-4B 和 GeoQA，从视觉 SFT、Vision GRPO 建立基线。通过选项重排、图片移除和有记录的图像扰动，建立多组对照。数据划分按原始问题与图片分组，同一题的各种变体保持在同一侧。

个人改进可采用训练集选项随机重排并同步更新答案标签，或基于训练侧采样结果调整样本难度。对照固定题目、推理预算与评测解析方式，避免把格式修复全部归因为视觉推理增强。

主要指标包括最终选项正确率、格式有效率、格式有效样本上的正确率、重排一致性、图像消融后的变化及不同题型表现。图片移除实验用于分析视觉依赖，结合人工错误分类解释结果。

开源基础包括 [教程 Vision GRPO](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/vision-grpo)、[GeoQA](https://huggingface.co/datasets/hz2475/geoQA) 和 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。本地训练框架需要核对该视觉架构、LoRA 和多模态 rollout 的支持情况。

粗略预算为 **500–1,800 元**，约需 3–5 周，先验证 48GB/80GB 单卡或同实例多卡。图片 token、模型架构和输出长度都影响实际显存。

完成后可用的项目名称为“面向几何问答的视觉强化学习与选项偏差分析”。

## 推荐组合

| 目标 | 主项目 | 可搭配的第二项目 |
| --- | --- | --- |
| 尽快把已有知识转成训练经验 | A 数学推理后训练 | D 工具路由决策 |
| Agent 算法与交互训练 | C 多轮任务 Agent | D 工具路由决策 |
| 模型后训练与蒸馏 | B 领域在线蒸馏 | A 中一个明确独立的研究问题 |
| 多模态算法 | E 视觉推理 | D 决策模型，或保留已有 CS336 项目 |

结合当前已读算法、尚未正式跑训练的阶段，建议先做 A，完成真实训练与评测，再在 C 或 D 中选择。五个候选可用于选题比较，每个项目都需要有独立问题、数据和实验贡献。

## 预算与交付口径

预算为工程规划估计，按 1–4B 学生模型、LoRA、数千至约一万条训练数据、少量控制变量实验及基础评测考虑。含有限调试余量，7B 全参数、超长轨迹、较大教师和大量种子实验需额外估算。时间按兼职学习与实现的日历周期估计。

AutoDL 同一实例支持单机多卡。普通容器实例的 Docker 限制、CUDA/框架组合、同机库存与 GPU 拓扑在选定项目后核对。采用原仓库方法时保留上游署名，并记录迁移实现与原方法的差异。[AutoDL 多卡说明](https://www.autodl.com/docs/distributed_training/) · [实例环境](https://www.autodl.com/docs/env/)

每个项目最终交付数据准备与划分、训练配置、评测脚本、逐题结果、失败案例、模型 adapter 和复现说明。简历里的指标填实测值，说明相对于哪个基线、在哪个评测集、使用什么推理预算。当前所有候选均处于设计阶段。
