"""Real Unity reset/step/reset/close check; random actions, no training."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
from .environment import ReacherEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--worker-id', type=int, default=21)
    p.add_argument('--agents', type=int, default=20)
    p.add_argument('--render', action='store_true')
    p.add_argument('--output', type=Path, default=Path('artifacts/smoke.json'))
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    scores = np.zeros(args.agents)
    done_events = []
    with ReacherEnvironment(args.environment, seed=args.seed, worker_id=args.worker_id,
                            expected_agents=args.agents, no_graphics=not args.render) as env:
        first = env.reset()
        for t in range(args.steps):
            actions = rng.uniform(-1, 1, (args.agents, 4)).astype(np.float32)
            result = env.step(actions)
            scores += result.rewards
            if result.legacy_done.any():
                done_events.append({'step': t + 1, 'done_count': int(result.legacy_done.sum())})
                env.reset()
        second = env.reset()
    binaries = list((args.environment / 'Contents' / 'MacOS').glob('*'))
    report = dict(status='passed', mode='random-action smoke check; not evaluation or training',
                  seed=args.seed, steps=args.steps, agents=args.agents,
                  state_shape=list(first.observations.shape), action_shape=[args.agents, 4],
                  reward_shape=list(result.rewards.shape), done_shape=list(result.legacy_done.shape),
                  second_reset_shape=list(second.observations.shape), closed=env._closed,
                  legacy_done_events=done_events,
                  legacy_done_semantics='unresolved: do not assume termination versus truncation',
                  partial_reward_sums=scores.tolist(), elapsed_seconds=time.monotonic()-start,
                  platform=platform.platform(), architecture=platform.machine(),
                  versions={n: importlib.metadata.version(n) for n in ['numpy','grpcio','protobuf','unityagents']},
                  binary_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in binaries if x.is_file()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
