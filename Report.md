# Continuous Control: DDPG on 20-Agent Reacher

## Task and success criterion

Each of 20 independent, noninteracting arms receives a 33-dimensional observation and chooses four continuous torques in `[-1,1]`. The environment awards +0.1 while an arm's hand is at the moving goal. For each complete episode, I sum undiscounted rewards per arm and then average the 20 scores. The Udacity criterion is a mean of at least +30 over 100 consecutive complete episode means. The first qualifying window is recorded by both start and end episode numbers.

## Method

The baseline is Deep Deterministic Policy Gradient (DDPG). One Actor maps observations to bounded actions; one Critic predicts discounted return for a state-action pair. Both use two ReLU hidden layers of width 256. The Actor output has four Tanh units; the Critic output is a single unrestricted scalar. Target copies begin equal to the online networks and undergo soft updates after each learner update.

With `s'` the observed next state and `d` a true task termination flag, the Critic target is `y = r + gamma * (1-d) * Q_target(s', Actor_target(s'))`. The Critic minimizes mean squared error to this detached target. The Actor minimizes `-mean(Q_online(s, Actor_online(s)))`; the online Critic parameters are frozen during this Actor step while the gradient through its action input remains available. The executed behavior action is the Actor output plus seeded Gaussian noise, clipped to `[-1,1]`. Replay stores that executed action, not the unperturbed Actor output.

The legacy API reports only `local_done`, without a termination/truncation distinction. This Mac build produced simultaneous `done` for all 20 arms at step 1001 in real checks. The collector treats that observed fixed horizon as truncation and bootstraps from its pre-reset next observation. An earlier or asynchronous done causes an error; this policy is an explicit assumption based on observed behavior, not an environment-provided label.

## Baseline settings and reproducibility

| Setting | Value |
|---|---:|
| Discount `gamma` | 0.99 |
| Target update `tau` | 0.001 |
| Actor / Critic learning rates | 0.0001 / 0.001 |
| Replay capacity | 1,000,000 transitions |
| Batch size | 128 transitions |
| Warmup | 1,000 environment steps with uniform random actions |
| Update frequency | One gradient update per environment step after warmup |
| Exploration | Gaussian, standard deviation 0.2 per action component |
| Episode horizon | 1,001 environment steps, measured on this build |

The 20 arms share one policy, one Critic, and one replay buffer. They are 20 concurrent experience sources, not 20 independent training seeds. Training, replay sampling, and exploration have separately seeded random generators. Runtime configuration, episode logs, checkpoints, summary JSON, and a learning curve are saved under each run directory.

## Measurement plan

Training score is the 20-arm mean of undiscounted episode rewards while actions include exploration. The rolling average uses only 100 consecutive complete training episodes. A checkpoint selected by rolling training performance is evaluated in separate full episodes with exploration and optimizer updates disabled. I report the evaluation episode scores and their population standard deviation, while clearly separating this variation from variation across independently trained seeds. A one-episode maximum is descriptive only. No significance claim will be made from a small number of seeds.

## Results

Three independent training seeds each ran 300 complete episodes (300,300 Unity environment steps, 6,006,000 arm transitions, and 299,301 optimizer updates per seed). All three met Udacity's rolling 100-episode training criterion. For each run, the checkpoint for evaluation was selected by its highest rolling 100-episode **training** mean. Each selected checkpoint was evaluated for 10 complete episodes with a separate Unity seed, no action exploration, and no optimizer updates. Unity's fast simulation mode was enabled during evaluation; this controls execution speed separately from learning.

| Training seed | First solved window (start–end) | Best rolling 100 training mean | Best single training episode | Independent evaluation mean ± episode SD¹ |
|---|---:|---:|---:|---:|
| 0 | 12–111 | 38.219 | 39.212 (episode 140) | 38.595 ± 0.227 |
| 1 | 39–138 | 38.062 | 39.260 (episode 296) | 36.882 ± 0.394 |
| 2 | 21–120 | 38.328 | 39.180 (episode 225) | 38.038 ± 0.403 |

Across the **three independent training runs**, the first solved-window endpoint was **123.0 ± 11.2 episodes** and the independent evaluation mean was **37.838 ± 0.713**. Both ± values here are population SD across training seeds. ¹The SDs in the table's final column instead describe variation across the 10 evaluation episodes *within that seed*. Neither SD is a confidence interval, and these three runs do not support a claim of statistical significance.

The first solved window uses its endpoint as the solved episode. The single best episode is not substituted for the rolling criterion. Training scores include exploratory actions; independent evaluation scores do not. Evaluation environment seeds were 10000, 10001, and 10002 for training seeds 0, 1, and 2 respectively.

The [combined learning curves](artifacts/learning_curves_comparison.png) show each training seed's 20-arm episode mean (faint) and complete rolling 100-episode mean (solid), with the +30 threshold marked. The [aggregate JSON](artifacts/aggregate.json), per-seed `episodes.jsonl`, `summary.json`, `evaluation.json`, and checkpoints provide the underlying data. Each per-seed directory also includes a separate training curve and config. The raw logs were audited by recomputing every arm mean, rolling 100 value, first qualifying window, and best rolling value; all matched the summaries.

Before full training, end-to-end diagnostics in `artifacts/diagnostic-seed0/` and `artifacts/diagnostic-seed1/` confirmed checkpoint, logging, and gradient-update behavior. They are not included in the three-seed statistics.

A later rendered, deterministic Unity rollout of the seed-0 selected checkpoint completed at environment seed 20000 with mean score 39.0045 (`artifacts/demo_rollout.json`). It is a separate one-episode demonstration and is not included in the ten-episode evaluation or cross-seed statistics. A shareable screen recording remains to be made.

## Limitations and next work

The time-limit interpretation of `legacy_done` is based on behavior observed on one build. A changed build or early done requires a new termination rule before training. The current learner uses one Critic and fixed Gaussian exploration. This baseline already solves the course task across all three tested seeds, so an extra algorithm is not needed to establish the result. A future TD3 or noise comparison would need a specific research question, comparable interaction budgets, and independently trained seeds. Three seeds characterize this small experiment but do not establish broad robustness across platforms or hyperparameters.
