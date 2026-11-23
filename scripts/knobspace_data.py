"""Loading and reshaping the 81 x 81 TX/RY knob-space scans.

The acquisition writes two timestamped CSV files (one at the DVF rest
position and one retraced to the focus).  ``point_index`` increases as
``ix * N + iy``, so a plain ``(N, N)`` reshape yields arrays indexed
``[TX, RY]``.  All scan-dependent constants used by the objective functions
live here so the notebooks do not need to repeat them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

Z_FOCUS = -142.0  # mm

# Normalisation constants from notebooks/optimization/CAX_optimization.ipynb.
I0 = 10.0
DELTA = 500.0
SIGMAX_0 = 19.9  # um
SIGMAY_0 = 11.5  # um

REQUIRED_COLUMNS = (
    "point_index",
    "TX_mm",
    "RY_mrad",
    "centroid_x_um",
    "centroid_x_err_um",
    "centroid_y_um",
    "centroid_y_err_um",
    "sigma_x_um",
    "sigma_x_err_um",
    "sigma_y_um",
    "sigma_y_err_um",
    "Intensity",
    "Intensity_err",
    "astigmatism_mm",
    "astigmatism_err_mm",
)


@dataclass(frozen=True)
class ScanData:
    """The two scans plus the shared TX/RY coordinate grids."""

    z0: pd.DataFrame
    zfocus: pd.DataFrame
    tx: np.ndarray
    ry: np.ndarray
    n: int
    z_focus: float = Z_FOCUS
    z0_path: Path | None = None
    zfocus_path: Path | None = None

    @property
    def focus_key(self) -> str:
        """Dictionary key used for the retraced scan."""
        return f"z = {self.z_focus:g} mm"

    @property
    def datasets(self) -> dict[str, pd.DataFrame]:
        """Both scans keyed by DVF position."""
        return {"z = 0 mm": self.z0, self.focus_key: self.zfocus}

    def describe(self) -> str:
        """One-line summary convenient for notebook output."""
        return (
            f"grid {self.n} x {self.n}; "
            f"TX {self.tx[0, 0]:+.3f} ... {self.tx[-1, 0]:+.3f} mm; "
            f"RY {self.ry[0, 0]:+.4f} ... {self.ry[0, -1]:+.4f} mrad; "
            f"z_focus {self.z_focus:g} mm"
        )


def find_scan_paths(
    data_dir: str | Path | None = None,
    *,
    z0_name: str | None = None,
    zfocus_name: str | None = None,
) -> tuple[Path, Path]:
    """Return the paths of the two scan CSVs.

    By default the most recent timestamped pair in ``data/`` is used.  Files
    ending in ``-yesterday`` are ignored.  Explicit filenames take precedence.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    if z0_name or zfocus_name:
        if not (z0_name and zfocus_name):
            raise ValueError("Provide both z0_name and zfocus_name, or neither.")
        return root / z0_name, root / zfocus_name

    z0_candidates = sorted(
        path
        for path in root.glob("tx_ry_scan_81x81_z0_*.csv")
        if "yesterday" not in path.stem
    )
    zf_candidates = sorted(
        path
        for path in root.glob("tx_ry_scan_81x81_zfocus_*.csv")
        if "yesterday" not in path.stem
    )
    if not z0_candidates or not zf_candidates:
        raise FileNotFoundError(
            f"Could not find a timestamped TX/RY scan pair in {root}. "
            "Pass data_dir or z0_name/zfocus_name explicitly."
        )
    return z0_candidates[-1], zf_candidates[-1]


def to_grid(frame: pd.DataFrame, column: str, n: int | None = None) -> np.ndarray:
    """Return a column as an ``(n, n)`` array ordered ``[TX, RY]``."""
    values = frame[column].to_numpy(dtype=float)
    if n is None:
        n = int(round(np.sqrt(values.size)))
    if n * n != values.size:
        raise ValueError(
            f"Cannot reshape {values.size} values into a square grid "
            f"(n={n}); pass the correct data frame."
        )
    return values.reshape(n, n)


def _ordered_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    order = "point_index" if "point_index" in frame else ["ix", "iy"]
    return frame.sort_values(order).reset_index(drop=True)


def load_scans(
    data_dir: str | Path | None = None,
    *,
    z0_name: str | None = None,
    zfocus_name: str | None = None,
    z_focus: float = Z_FOCUS,
) -> ScanData:
    """Read both CSVs, verify their ordering and build the shared grids."""
    z0_path, zf_path = find_scan_paths(
        data_dir, z0_name=z0_name, zfocus_name=zfocus_name
    )
    df_z0 = _ordered_frame(z0_path)
    df_zf = _ordered_frame(zf_path)

    if len(df_z0) != len(df_zf):
        raise ValueError(
            f"Scan files have different lengths: {len(df_z0)} vs {len(df_zf)}"
        )

    n = round(np.sqrt(len(df_z0)))
    if n * n != len(df_z0):
        raise ValueError(f"Scan has {len(df_z0)} points, which is not a square grid")

    tx = to_grid(df_z0, "TX_mm", n)
    ry = to_grid(df_z0, "RY_mrad", n)
    if not np.allclose(tx, to_grid(df_zf, "TX_mm", n)):
        raise ValueError("TX grid differs between the z=0 and z=focus scans")
    if not np.allclose(ry, to_grid(df_zf, "RY_mrad", n)):
        raise ValueError("RY grid differs between the z=0 and z=focus scans")

    if {"ix", "iy"}.issubset(df_z0.columns):
        ix = to_grid(df_z0, "ix", n)
        iy = to_grid(df_z0, "iy", n)
        expected_ix, expected_iy = np.meshgrid(
            np.arange(n), np.arange(n), indexing="ij"
        )
        if not (np.array_equal(ix, expected_ix) and np.array_equal(iy, expected_iy)):
            raise ValueError("Scan indices are not ordered as ix * N + iy ([TX, RY])")

    return ScanData(
        z0=df_z0,
        zfocus=df_zf,
        tx=tx,
        ry=ry,
        n=n,
        z_focus=z_focus,
        z0_path=z0_path,
        zfocus_path=zf_path,
    )
