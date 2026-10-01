# Continuous Control — Resume and Interview Notes

Use these bullets after linking the public repository. The numbers come from three independent 300-episode training runs and separate 10-episode deterministic evaluations per trained model. “±” is population standard deviation across the three training seeds, not a confidence interval.

## English resume bullets

- Built a reproducible DDPG agent from scratch for Udacity's 20-agent Reacher task, including independent Actor/Critic networks, replay, target-network soft updates, seeded exploration, checkpointing, and train/evaluate CLIs; validated the implementation with 39 automated tests and real Unity runs.
- Solved the +30 rolling 100-episode training criterion in all three independent runs (first solved-window endpoint: **123.0 ± 11.2 episodes**); selected checkpoints by training performance and achieved **37.84 ± 0.71** mean score in separate deterministic evaluations across training seeds.

## 中文简历要点

- 从零实现面向 Udacity 20 机械臂 Reacher 的可复现 DDPG：独立 Actor/Critic、经验回放、目标网络软更新、带 seed 的探索、checkpoint 与训练/评估命令；通过 39 项自动化测试及真实 Unity 环境运行验证。
- 三次独立训练均达到连续 100 episode 平均得分 ≥30 的课程标准，首次达标窗口终点为 **123.0 ± 11.2 episode**；按训练表现选择模型后，在独立无探索评估中取得跨训练 seed 的 **37.84 ± 0.71** 平均得分。

## Interview explanation

- **Why DDPG?** Four continuous torque values cannot be exhaustively enumerated. The Actor proposes an action and the Critic evaluates its long-term return.
- **What entered replay?** The actual action after exploration noise and clipping, paired with the resulting reward and next observation.
- **How did the gradients flow?** The Critic minimizes a detached target-network Bellman error. The Actor minimizes negative online-Critic Q while the Critic's parameters are frozen but its action derivative remains active.
- **What does solved mean?** The first complete 100-episode training window whose 20-arm mean reaches +30; the endpoint is reported. Single best episodes and independent evaluations are separate numbers.
- **What is the key limitation?** The legacy Unity API has one done flag. This build always ended all 20 arms at the measured 1001-step boundary, treated as truncation for bootstrap; an early or asynchronous done stops the collector for review. Three seeds do not establish statistical significance.
