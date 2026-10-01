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

Full baseline training is in progress. No solve claim is available yet. End-to-end diagnostics are recorded in `artifacts/diagnostic-seed0/` and `artifacts/diagnostic-seed1/` and should not be interpreted as trained-policy performance. The former ran one complete episode with only two updates and produced training mean 0.0795 and separate deterministic evaluation mean 0.0. The latter ran two complete episodes with 1,003 updates. These runs validate the pipeline and logging.

## Limitations and next work

The time-limit interpretation of `legacy_done` is based on behavior observed on one build. A changed build or early done requires a new termination rule before training. The current learner uses one Critic and fixed Gaussian exploration; after a credible baseline, instability or value overestimation would motivate a controlled TD3 comparison, while weak exploration would motivate a focused noise comparison. Any enhancement should use comparable interaction budgets and independent training seeds.
