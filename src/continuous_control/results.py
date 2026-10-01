"""Aggregate independent training runs without conflating agents or episodes."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot_comparison(run_paths, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#4b6fb2', '#7253a4', '#3f8b67']
    for index, path in enumerate(map(Path, run_paths)):
        rows = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
        seed = json.loads((path / 'summary.json').read_text())['seed']
        color = colors[index % len(colors)]
        x = [row['episode'] for row in rows]
        ax.plot(x, [row['mean_score'] for row in rows], color=color, alpha=0.17, linewidth=0.8)
        rolling = [row for row in rows if row['rolling_100'] is not None]
        ax.plot([row['episode'] for row in rolling],
                [row['rolling_100'] for row in rolling],
                color=color, linewidth=2, label=f'Seed {seed}: rolling 100')
    ax.axhline(30, color='#b84d5d', linestyle='--', linewidth=1.3,
               label='Udacity threshold')
    ax.set(xlabel='Training episode', ylabel='20-arm mean undiscounted score',
           title='DDPG Reacher: independent training seeds')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def aggregate(run_paths):
    runs = []
    reference_config = None
    seen_seeds = set()
    for path in map(Path, run_paths):
        config = json.loads((path / 'config.json').read_text())
        summary = json.loads((path / 'summary.json').read_text())
        seed = summary['seed']
        if seed in seen_seeds:
            raise ValueError(f'duplicate training seed: {seed}')
        seen_seeds.add(seed)
        comparable = dict(config)
        comparable['agent'] = dict(config['agent'])
        comparable['agent'].pop('seed')
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise ValueError('training configurations differ beyond seed')
        evaluation_path = path / 'evaluation.json'
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
        runs.append({'seed': seed, 'path': str(path),
                     'complete_episodes': summary['complete_episodes'],
                     'first_solved_window_end': summary['first_solved_window_end'],
                     'best_rolling_100': summary['best_rolling_100'],
                     'evaluation_mean_score': evaluation['mean_score'] if evaluation else None,
                     'evaluation_episode_population_sd':
                         evaluation['episode_population_sd'] if evaluation else None})
    if not runs:
        raise ValueError('at least one run is required')
    runs.sort(key=lambda row: row['seed'])
    eval_means = [row['evaluation_mean_score'] for row in runs]
    all_evaluated = all(value is not None for value in eval_means)
    all_solved = all(row['first_solved_window_end'] is not None for row in runs)
    return {'training_seed_count': len(runs), 'runs': runs,
            'all_runs_solved': all_solved,
            'all_runs_evaluated': all_evaluated,
            'solved_episode_mean': float(np.mean([row['first_solved_window_end'] for row in runs]))
                if all_solved else None,
            'solved_episode_population_sd': float(np.std([row['first_solved_window_end'] for row in runs], ddof=0))
                if all_solved and len(runs) > 1 else None,
            'evaluation_mean_across_seeds': float(np.mean(eval_means)) if all_evaluated else None,
            'evaluation_population_sd_across_seeds': float(np.std(eval_means, ddof=0))
                if all_evaluated and len(runs) > 1 else None,
            'uncertainty_note': 'Population SD across independent training seeds; not a confidence interval or significance test.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    result = aggregate(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.plot:
        plot_comparison(args.run, args.plot)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
