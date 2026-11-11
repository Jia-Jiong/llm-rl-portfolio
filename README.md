# LLM RL Portfolio

大模型强化学习学习与实验记录。以 [Datawhale Agentic RL Lab](https://datawhalechina.github.io/agentic-rl-lab/) 为参考，逐步实现奖励函数、组采样与评测，并记录实验配置和结果。

## 当前项目规划

**ToolSQL-RL — 基于执行反馈的小模型 SQL 生成与自修正**

使用开源 Qwen 模型，在 SQLite 环境中逐步完成 SFT、GRPO 和受限多轮工具交互，比较跨数据库的查询正确率、生成成本与失败类型。

- [项目方案、开源基础、实验设计与 AutoDL 预算](docs/toolsql-rl-plan.md)
- 训练框架计划采用 TRL、PEFT 和 vLLM。
- 算力预算上限为 500 元，候选为同一 AutoDL 实例内的双 4090 或单张大显存 GPU。

当前处于方案设计阶段，训练配置、资源消耗和模型效果需要后续实验验证。
