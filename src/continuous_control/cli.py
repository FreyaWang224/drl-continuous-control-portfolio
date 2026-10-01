"""Train, independently evaluate, and plot the Udacity 20-arm Reacher task."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .agent import DDPGAgent, DDPGConfig
from .environment import ReacherEnvironment
from .replay import ReplayBuffer

HORIZON = 1001  # Measured: 20 simultaneous legacy_done flags on step 1001.


def save_checkpoint(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(agent.checkpoint(), temporary)
    temporary.replace(path)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    agent = DDPGAgent(DDPGConfig(**data['config']))
    agent.restore(data)
    return agent


def episode(env, agent, *, training, max_steps=HORIZON, replay=None,
            random_rng=None, warmup_steps=0, global_steps=0,
            batch_size=64, update_every=1, unity_train_mode=True):
    """Collect one run. A short max_steps run is diagnostic, not a full episode."""
    # Unity's train_mode controls simulation speed. Exploration and optimizer
    # updates are controlled separately by the `training` flag here.
    current = env.reset(training=unity_train_mode)
    scores = np.zeros(env.expected_agents, dtype=np.float64)
    losses = []
    completed = False
    for step_number in range(1, max_steps + 1):
        if training and global_steps < warmup_steps:
            actions = random_rng.uniform(-1, 1, (env.expected_agents, 4)).astype(np.float32)
        else:
            actions = agent.act(current.observations, explore=training)
        result = env.step(actions)
        scores += result.rewards
        if result.legacy_done.any() and (step_number != HORIZON or not result.legacy_done.all()):
            raise RuntimeError('Unexpected or asynchronous legacy_done; terminal semantics need review')
        if training:
            replay.add(current.observations, actions, result.rewards,
                       result.observations, result.legacy_done)
            global_steps += 1
            if (global_steps >= warmup_steps and global_steps % update_every == 0
                    and len(replay) >= batch_size):
                sampled = replay.sample(batch_size)
                # Observed all-arm fixed-horizon done is treated as truncation.
                # No true termination has been established for this task.
                terminated = np.zeros((batch_size, 1), dtype=np.float32)
                losses.append(agent.learn(sampled, terminated))
        if result.legacy_done.any():
            completed = True
            break
        current = result
    if max_steps == HORIZON and not completed:
        raise RuntimeError('Expected simultaneous legacy_done at the measured horizon')
    return {'complete_episode': completed, 'steps': step_number,
            'arm_scores': scores.tolist(), 'mean_score': float(scores.mean()),
            'global_environment_steps': global_steps,
            'transitions_seen': global_steps * env.expected_agents,
            'updates': agent.updates,
            'mean_critic_loss': float(np.mean([x['critic_loss'] for x in losses])) if losses else None,
            'mean_actor_loss': float(np.mean([x['actor_loss'] for x in losses])) if losses else None}


def plot_training(log_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line]
    if not rows:
        raise ValueError('empty training log')
    episodes = [row['episode'] for row in rows]
    scores = [row['mean_score'] for row in rows]
    complete = all(row['complete_episode'] for row in rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(episodes, scores, color='#4b6fb2', linewidth=1, marker='o', markersize=3,
            label='20-arm mean score')
    if len(episodes) == 1:
        ax.set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)
    rolling_x = [row['episode'] for row in rows if row.get('rolling_100') is not None]
    rolling_y = [row['rolling_100'] for row in rows if row.get('rolling_100') is not None]
    if rolling_x:
        ax.plot(rolling_x, rolling_y, color='#5c3f94', linewidth=2, label='100-episode mean')
        ax.axhline(30, color='#ac4255', linestyle='--', linewidth=1, label='Udacity threshold')
    ax.set(xlabel='Training episode' if complete else 'Diagnostic partial run',
           ylabel='Undiscounted mean reward',
           title='Reacher DDPG training' if complete else 'Reacher DDPG diagnostic run (incomplete episodes)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def train(args):
    if args.episodes < 1 or not 1 <= args.max_steps <= HORIZON:
        raise ValueError('episodes must be positive and max_steps in [1,1001]')
    if (args.batch_size < 1 or args.capacity < args.batch_size or
            args.update_every < 1 or args.checkpoint_every < 1 or args.warmup_steps < 0):
        raise ValueError('invalid batch/capacity/update frequency')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = DDPGConfig(seed=args.seed, gamma=args.gamma, tau=args.tau,
                        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                        noise_std=args.noise_std, device='cpu')
    (output / 'config.json').write_text(json.dumps({'agent': asdict(config),
        'episodes': args.episodes, 'max_steps': args.max_steps, 'warmup_steps': args.warmup_steps,
        'batch_size': args.batch_size, 'capacity': args.capacity,
        'update_every': args.update_every, 'bootstrap_policy':
        'observed synchronous 1001-step legacy_done treated as time-limit truncation'}, indent=2) + '\n')
    agent = DDPGAgent(config)
    replay = ReplayBuffer(capacity=args.capacity, seed=args.seed + 2)
    random_rng = np.random.default_rng(args.seed + 3)
    rolling = deque(maxlen=100)
    best_rolling = float('-inf')
    solved_episode = None
    global_steps = 0
    start = time.monotonic()
    with ReacherEnvironment(args.environment, seed=args.seed, worker_id=args.worker_id) as env:
        with (output / 'episodes.jsonl').open('w') as log:
            for number in range(1, args.episodes + 1):
                result = episode(env, agent, training=True, max_steps=args.max_steps,
                                 replay=replay, random_rng=random_rng,
                                 warmup_steps=args.warmup_steps, global_steps=global_steps,
                                 batch_size=args.batch_size, update_every=args.update_every)
                global_steps = result['global_environment_steps']
                if result['complete_episode']:
                    rolling.append(result['mean_score'])
                else:
                    rolling.clear()
                rolling_mean = float(np.mean(rolling)) if len(rolling) == 100 else None
                if rolling_mean is not None and rolling_mean >= 30 and solved_episode is None:
                    solved_episode = number
                result.update(episode=number, rolling_100=rolling_mean,
                              solved_episode=solved_episode,
                              elapsed_seconds=time.monotonic() - start)
                log.write(json.dumps(result) + '\n')
                log.flush()
                print(f"episode={number} complete={result['complete_episode']} "
                      f"mean={result['mean_score']:.3f} rolling100={rolling_mean} "
                      f"updates={agent.updates}", flush=True)
                if rolling_mean is not None and rolling_mean > best_rolling:
                    best_rolling = rolling_mean
                    save_checkpoint(agent, output / 'best_rolling_100.pt')
                if number % args.checkpoint_every == 0:
                    save_checkpoint(agent, output / 'last.pt')
    save_checkpoint(agent, output / 'last.pt')
    plot_training(output / 'episodes.jsonl', output / 'training_curve.png')
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    complete_rows = [row for row in rows if row['complete_episode']]
    best_episode = max(complete_rows, key=lambda row: row['mean_score']) if complete_rows else None
    summary = {'mode': 'training', 'seed': args.seed, 'episodes_run': args.episodes,
               'complete_episodes': len(complete_rows),
               'best_single_episode': best_episode['episode'] if best_episode else None,
               'best_single_episode_mean_score': best_episode['mean_score'] if best_episode else None,
               'first_solved_window_end': solved_episode,
               'first_solved_window_start': solved_episode - 99 if solved_episode else None,
               'best_rolling_100': best_rolling if best_rolling != float('-inf') else None,
               'elapsed_seconds': time.monotonic() - start,
               'environment_steps': global_steps, 'transitions': global_steps * 20,
               'updates': agent.updates}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def evaluate(args):
    if args.episodes < 1:
        raise ValueError('episodes must be positive')
    agent = load_checkpoint(args.checkpoint)
    rows = []
    with ReacherEnvironment(args.environment, seed=args.seed, worker_id=args.worker_id) as env:
        for number in range(1, args.episodes + 1):
            result = episode(env, agent, training=False,
                             unity_train_mode=not args.realtime)
            if not result['complete_episode']:
                raise RuntimeError('evaluation did not complete a full episode')
            rows.append({'episode': number, 'mean_score': result['mean_score'],
                         'arm_scores': result['arm_scores'], 'steps': result['steps']})
    means = np.array([row['mean_score'] for row in rows])
    report = {'mode': 'independent deterministic evaluation',
              'checkpoint': str(args.checkpoint), 'environment_seed': args.seed,
              'exploration': False, 'optimizer_updates': 0,
              'unity_train_mode_fast_simulation': not args.realtime,
              'episodes': rows, 'mean_score': float(means.mean()),
              'episode_population_sd': float(means.std(ddof=0))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def demo(args):
    """Render a deterministic rollout for an external window recorder."""
    if not 1 <= args.steps <= HORIZON or args.start_delay < 0:
        raise ValueError('demo steps must be in [1,1001] and delay nonnegative')
    agent = load_checkpoint(args.checkpoint)
    scores = np.zeros(20, dtype=np.float64)
    with ReacherEnvironment(args.environment, seed=args.seed,
                            worker_id=args.worker_id, no_graphics=False) as env:
        current = env.reset(training=False)
        time.sleep(args.start_delay)
        complete = False
        for number in range(1, args.steps + 1):
            result = env.step(agent.act(current.observations, explore=False))
            scores += result.rewards
            if result.legacy_done.any():
                if number != HORIZON or not result.legacy_done.all():
                    raise RuntimeError('unexpected done during demo')
                complete = True
                break
            current = result
    report = {'mode': 'rendered deterministic demo', 'checkpoint': str(args.checkpoint),
              'seed': args.seed, 'steps': number, 'complete_episode': complete,
              'mean_reward_so_far': float(scores.mean()),
              'score_note': 'full evaluation score only if complete_episode is true'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--environment', type=Path, required=True)
    train_parser.add_argument('--output', type=Path, required=True)
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.add_argument('--worker-id', type=int, default=0)
    train_parser.add_argument('--episodes', type=int, default=500)
    train_parser.add_argument('--max-steps', type=int, default=HORIZON)
    train_parser.add_argument('--warmup-steps', type=int, default=1000)
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--capacity', type=int, default=1_000_000)
    train_parser.add_argument('--update-every', type=int, default=1)
    train_parser.add_argument('--gamma', type=float, default=0.99)
    train_parser.add_argument('--tau', type=float, default=0.001)
    train_parser.add_argument('--actor-lr', type=float, default=1e-4)
    train_parser.add_argument('--critic-lr', type=float, default=1e-3)
    train_parser.add_argument('--noise-std', type=float, default=0.2)
    train_parser.add_argument('--checkpoint-every', type=int, default=10)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('--environment', type=Path, required=True)
    eval_parser.add_argument('--checkpoint', type=Path, required=True)
    eval_parser.add_argument('--output', type=Path, required=True)
    eval_parser.add_argument('--seed', type=int, default=10000)
    eval_parser.add_argument('--worker-id', type=int, default=1)
    eval_parser.add_argument('--episodes', type=int, default=10)
    eval_parser.add_argument('--realtime', action='store_true',
                             help='run Unity at display speed instead of fast simulation')
    plot_parser = sub.add_parser('plot')
    plot_parser.add_argument('--log', type=Path, required=True)
    plot_parser.add_argument('--output', type=Path, required=True)
    demo_parser = sub.add_parser('demo')
    demo_parser.add_argument('--environment', type=Path, required=True)
    demo_parser.add_argument('--checkpoint', type=Path, required=True)
    demo_parser.add_argument('--output', type=Path, required=True)
    demo_parser.add_argument('--seed', type=int, default=20000)
    demo_parser.add_argument('--worker-id', type=int, default=60)
    demo_parser.add_argument('--steps', type=int, default=HORIZON)
    demo_parser.add_argument('--start-delay', type=float, default=10)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'demo':
        demo(args)
    else:
        plot_training(args.log, args.output)


if __name__ == '__main__':
    main()
