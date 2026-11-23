import numpy as np
import pandas as pd
import pytest

from scripts.knobspace_data import REQUIRED_COLUMNS, load_scans, to_grid
from scripts.objective_functions import (
    f1_from_measurements,
    objective_grids,
    objective_minima,
)


def make_frame(n=2):
    rows = []
    for ix in range(n):
        for iy in range(n):
            rows.append(
                {
                    "point_index": ix * n + iy,
                    "ix": ix,
                    "iy": iy,
                    "TX_mm": float(ix),
                    "RY_mrad": float(iy) / 10,
                    "centroid_x_um": 0.0,
                    "centroid_x_err_um": 0.1,
                    "centroid_y_um": 0.0,
                    "centroid_y_err_um": 0.1,
                    "sigma_x_um": 1000.0 * (ix + 1),
                    "sigma_x_err_um": 10.0,
                    "sigma_y_um": 1000.0 * (iy + 1),
                    "sigma_y_err_um": 10.0,
                    "Intensity": 10.0 * (ix + iy + 1),
                    "Intensity_err": 0.5,
                    "astigmatism_mm": 500.0 * iy,
                    "astigmatism_err_mm": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_to_grid_orders_by_tx_then_ry():
    frame = make_frame()
    tx = to_grid(frame, "TX_mm", 2)
    ry = to_grid(frame, "RY_mrad", 2)
    np.testing.assert_array_equal(tx, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(ry, [[0.0, 0.1], [0.0, 0.1]])


def test_load_scans_builds_shared_grids(tmp_path):
    frame = make_frame()
    z0_path = tmp_path / "tx_ry_scan_81x81_z0_20260101_000000.csv"
    zf_path = tmp_path / "tx_ry_scan_81x81_zfocus_20260101_000000.csv"
    frame[list(REQUIRED_COLUMNS)].to_csv(z0_path, index=False)
    frame[list(REQUIRED_COLUMNS)].to_csv(zf_path, index=False)

    scan = load_scans(
        tmp_path, z0_name=z0_path.name, zfocus_name=zf_path.name, z_focus=-142.0
    )
    assert scan.n == 2
    assert scan.focus_key == "z = -142 mm"
    assert set(scan.datasets) == {"z = 0 mm", "z = -142 mm"}
    np.testing.assert_array_equal(scan.tx, to_grid(frame, "TX_mm", 2))
    np.testing.assert_array_equal(scan.ry, to_grid(frame, "RY_mrad", 2))
    assert "TX -2.000 ... +2.000" not in scan.describe()  # uses the actual grid
    assert "grid 2 x 2" in scan.describe()


def test_load_scans_rejects_mismatched_grids(tmp_path):
    frame = make_frame()
    z0_path = tmp_path / "a_z0.csv"
    zf_path = tmp_path / "a_zfocus.csv"
    frame[list(REQUIRED_COLUMNS)].to_csv(z0_path, index=False)
    altered = frame.copy()
    altered.loc[altered["point_index"] == 0, "TX_mm"] = 99.0
    altered[list(REQUIRED_COLUMNS)].to_csv(zf_path, index=False)

    with pytest.raises(ValueError, match="TX grid differs"):
        load_scans(tmp_path, z0_name=z0_path.name, zfocus_name=zf_path.name)


def test_objective_grids_and_minima():
    frame = make_frame()
    objectives, errors = objective_grids(frame)
    assert set(objectives) == {f"f{i}" for i in range(1, 9)}
    assert all(values.shape == (2, 2) for values in objectives.values())
    assert all(values.shape == (2, 2) for values in errors.values())
    assert np.isfinite(objectives["f1"]).all()
    np.testing.assert_allclose(objectives["f1"][0, 0], 100_000.0, rtol=1e-12)

    minima = objective_minima(frame, to_grid(frame, "TX_mm", 2), to_grid(frame, "RY_mrad", 2))
    assert list(minima["objective"]) == [f"f{i}" for i in range(1, 9)]


def test_f1_from_measurements_propagates_errors():
    value, error = f1_from_measurements(48.0, 18.0, 11.0, 0.5, 0.2, 0.1)
    expected = 18.0 * 11.0 / 48.0
    expected_error = expected * np.sqrt(
        (0.5 / 48.0) ** 2 + (0.2 / 18.0) ** 2 + (0.1 / 11.0) ** 2
    )
    assert value == pytest.approx(expected)
    assert error == pytest.approx(expected_error)

    assert np.isnan(f1_from_measurements(0.0, 18.0, 11.0, 0.5, 0.2, 0.1)[0])
    assert np.isnan(f1_from_measurements(48.0, np.nan, 11.0, 0.5, 0.2, 0.1)[1])
