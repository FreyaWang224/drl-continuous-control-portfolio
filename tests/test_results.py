import json

import pytest

from continuous_control.results import aggregate


def write_run(path, seed, solved, eval_score):
    path.mkdir()
    (path / 'config.json').write_text(json.dumps({'agent': {'seed': seed, 'gamma': 0.99},
                                                    'episodes': 200}))
    (path / 'summary.json').write_text(json.dumps({'seed': seed, 'complete_episodes': 200,
                                                    'first_solved_window_end': solved,
                                                    'best_rolling_100': 35.0}))
    if eval_score is not None:
        (path / 'evaluation.json').write_text(json.dumps({'mean_score': eval_score,
                                                           'episode_population_sd': 1.0}))


def test_across_seed_variability_uses_independent_run_means(tmp_path):
    paths = [tmp_path / str(i) for i in range(3)]
    for path, seed, solved, score in zip(paths, [0, 1, 2], [100, 110, 120], [30, 32, 34]):
        write_run(path, seed, solved, score)
    result = aggregate(paths)
    assert result['training_seed_count'] == 3
    assert result['solved_episode_mean'] == 110
    assert result['evaluation_mean_across_seeds'] == 32
    assert result['evaluation_population_sd_across_seeds'] == pytest.approx((8 / 3) ** 0.5)


def test_missing_evaluation_and_unsolved_run_do_not_get_partial_statistics(tmp_path):
    paths = [tmp_path / 'a', tmp_path / 'b']
    write_run(paths[0], 0, 111, 30)
    write_run(paths[1], 1, None, None)
    result = aggregate(paths)
    assert result['evaluation_mean_across_seeds'] is None
    assert result['solved_episode_mean'] is None
    assert not result['all_runs_solved']


def test_duplicate_seed_rejected(tmp_path):
    paths = [tmp_path / 'a', tmp_path / 'b']
    write_run(paths[0], 0, 111, 30)
    write_run(paths[1], 0, 115, 31)
    with pytest.raises(ValueError, match='duplicate'):
        aggregate(paths)
