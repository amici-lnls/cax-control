from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.cax_optimization as co


def test_study_configs_have_consistent_knob_settings():
    cases = [
        (co.make_txry_config(), ("tx", "ry")),
        (co.make_tyrx_config(), ("ty", "rx")),
        (co.make_tyrxrz_config(), ("ty", "rx", "rz")),
        (co.make_5d_config(), ("tx", "ty", "rx", "ry", "rz")),
    ]
    for config, knobs in cases:
        assert config.knobs == knobs
        assert len(config.limit_lower) == len(knobs)
        assert len(config.limit_upper) == len(knobs)
        if config.initial_search_directions is not None:
            assert config.initial_search_directions.shape == (len(knobs), len(knobs))

    assert co.make_tyrx_config().fixed_mirror == {"tx": 0.0, "ry": 25.255e-3}
    assert co.make_tyrxrz_config().fixed_mirror == {"tx": 0.0, "ry": 25.255e-3}


def test_sample_initial_position_is_seeded_and_in_bounds():
    config = co.make_txry_config(seed=42)
    first = co.sample_initial_position(config)
    second = co.sample_initial_position(config)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= config.limit_lower)
    assert np.all(first <= config.limit_upper)

    fixed = co.make_txry_config(initial_position=(1.5, -0.075))
    np.testing.assert_array_equal(co.sample_initial_position(fixed), (1.5, -0.075))


def test_apply_position_sets_knobs_and_keeps_prealigned_values():
    config = co.make_tyrxrz_config()
    mirror = SimpleNamespace()
    fake_opt = SimpleNamespace(
        config=config,
        params=SimpleNamespace(caxsim=SimpleNamespace(mirror=mirror)),
    )

    co.CAXRCDS._apply_position(fake_opt, (0.25, -0.02, 0.03))

    assert mirror.tx == 0.0
    assert mirror.ry == 25.255e-3
    assert mirror.ty == 0.25
    assert mirror.rx == -0.02
    assert mirror.rz == 0.03


def test_build_params_applies_fixed_mirror_without_a_real_beamline(monkeypatch):
    class FakeMirror:
        pass

    class FakeCAXSim:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mirror = FakeMirror()

    monkeypatch.setattr(co, "CAXSim", FakeCAXSim)
    config = co.make_tyrx_config(seed=7)
    params = co.build_params(config)

    assert params.caxsim.kwargs == {
        "total_rays": config.total_rays,
        "image_size": config.image_size,
    }
    assert params.caxsim.mirror.tx == 0.0
    assert params.caxsim.mirror.ry == 25.255e-3
    assert params.config is config
    np.testing.assert_array_equal(
        params.initial_position, co.sample_initial_position(config)
    )
    np.testing.assert_array_equal(
        params.initial_search_directions, np.eye(2)
    )


def test_optimization_summary_and_saved_histories(tmp_path):
    opt = SimpleNamespace(
        config=co.make_txry_config(),
        num_objective_evals=3,
        objfuncs_evaluated=[4.0, 3.0, 2.0],
        objfuncs_best=[4.0, 3.0, 2.0],
        positions_evaluated=[[0.0, 0.0], [0.1, 0.01], [0.2, 0.02]],
        positions_best=[[0.0, 0.0], [0.1, 0.01], [0.2, 0.02]],
    )
    summary = co.optimization_summary(opt, f1_grid=np.array([[1.0, 2.0]]))
    assert summary["evaluations"] == 3
    assert summary["initial objective"] == 4.0
    assert summary["best objective"] == 2.0
    assert summary["f1 grid minimum"] == 1.0

    history = pd.DataFrame({"f1": [4.0, 3.0, 2.0]})
    errors = pd.DataFrame({"f1": [0.1, 0.1, 0.1]})
    opt.history_dataframe = lambda: history
    opt.error_dataframe = lambda: errors
    paths = co.save_histories(opt, tmp_path)

    assert [path.name for path in paths] == [
        "TXRY_observables.csv",
        "TXRY_errors.csv",
    ]
    pd.testing.assert_frame_equal(pd.read_csv(paths[0]), history)
