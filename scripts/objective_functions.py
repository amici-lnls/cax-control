"""Objective functions and minima for the knob-space scans.

The definitions of f1..f8 and their normalisation constants are kept exactly
as in the original ``CAX_optimization.ipynb``.  Uncertainties are propagated
with the delta method, assuming independent observables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.knobspace_data import DELTA, I0, SIGMAX_0, SIGMAY_0, to_grid

OBJECTIVE_LABELS = {
    "f1": r"$\sigma_x\sigma_y/I$",
    "f2": r"$\sigma_x/\sigma_x^0 + \sigma_y/\sigma_y^0$",
    "f3": r"$-I/I_0$",
    "f4": r"$f_2(\sigma_x, \sigma_y) + f_3(I)$",
    "f5": r"$|d|/\Delta$",
    "f6": r"$f_2(\sigma_x, \sigma_y) + f_5(d)$",
    "f7": r"$f_3(I) + f_5(d)$",
    "f8": r"$f_4(\sigma_x, \sigma_y, I) + f_5(d)$",
}

OBJECTIVE_NAMES = {
    "f1": "Mixed geometric intensity",
    "f2": "Geometric",
    "f3": "Intensity",
    "f4": "Geometric and intensity",
    "f5": "Astigmatism",
    "f6": "Geometric and astigmatism",
    "f7": "Intensity and astigmatism",
    "f8": "Geometric, intensity and astigmatism",
}

ORDER = [f"f{i}" for i in range(1, 9)]

# (value column, error column, short name, colourbar label, highlight mode)
OBSERVABLES = [
    ("centroid_x_um", "centroid_x_err_um", "centroid x", r"$x_c$ [$\mu$m]", "zero"),
    ("centroid_y_um", "centroid_y_err_um", "centroid y", r"$y_c$ [$\mu$m]", "zero"),
    ("sigma_x_um", "sigma_x_err_um", r"$\sigma_x$", r"$\sigma_x$ [$\mu$m]", "min"),
    ("sigma_y_um", "sigma_y_err_um", r"$\sigma_y$", r"$\sigma_y$ [$\mu$m]", "min"),
    ("Intensity", "Intensity_err", "Intensity", r"$-I$ [count / $\mu$m$^2$]", "neg_min"),
    ("astigmatism_mm", "astigmatism_err_mm", "Astigmatism", r"$|d|$ [mm]", "abs_min"),
]

OBSERVABLE_QUAD = [
    ("sigma_x_um", "sigma_x_err_um", r"$\sigma_x$", r"$\sigma_x$ [$\mu$m]", "min"),
    ("sigma_y_um", "sigma_y_err_um", r"$\sigma_y$", r"$\sigma_y$ [$\mu$m]", "min"),
    ("Intensity", "Intensity_err", "Intensity", r"$-I$ [count / $\mu$m$^2$]", "neg_min"),
    ("Objective", "objective_err", "Objective function", r"$f_1=\sigma_x\sigma_y/I$", "min"),
]


def objective_grids(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute f1..f8 and their propagated uncertainties on a scan grid."""
    sx = to_grid(frame, "sigma_x_um") / 1000.0
    sy = to_grid(frame, "sigma_y_um") / 1000.0
    esx = to_grid(frame, "sigma_x_err_um") / 1000.0
    esy = to_grid(frame, "sigma_y_err_um") / 1000.0

    intensity = to_grid(frame, "Intensity")
    e_intensity = to_grid(frame, "Intensity_err")
    astig = to_grid(frame, "astigmatism_mm")
    e_astig = to_grid(frame, "astigmatism_err_mm")

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = (1000.0) ** 2 * sx * sy / intensity
        f2 = 1000.0 * sx / SIGMAX_0 + 1000.0 * sy / SIGMAY_0
        f3 = -intensity / I0
        f4 = f2 + f3
        f5 = np.abs(astig) / DELTA
        f6 = f2 + f5
        f7 = f3 + f5
        f8 = f4 + f5

        e1 = np.abs(f1) * np.sqrt(
            (esx / sx) ** 2 + (esy / sy) ** 2 + (e_intensity / intensity) ** 2
        )
        e2 = np.sqrt((1000.0 * esx / SIGMAX_0) ** 2 + (1000.0 * esy / SIGMAY_0) ** 2)
        e3 = e_intensity / I0
        e4 = np.sqrt(e2**2 + e3**2)
        e5 = e_astig / DELTA
        e6 = np.sqrt(e2**2 + e5**2)
        e7 = np.sqrt(e3**2 + e5**2)
        e8 = np.sqrt(e2**2 + e3**2 + e5**2)

    objectives = dict(zip(ORDER, (f1, f2, f3, f4, f5, f6, f7, f8)))
    errors = dict(zip(ORDER, (e1, e2, e3, e4, e5, e6, e7, e8)))
    return objectives, errors


def f1_from_measurements(
    peak: float,
    sig_major: float,
    sig_minor: float,
    peak_error: float,
    sig_major_error: float,
    sig_minor_error: float,
) -> tuple[float, float]:
    """Return ``f1`` and its propagated error for a single fitted image.

    Widths and peak are in um and counts/um^2, so the objective has the same
    units as the ``f1`` grid computed from the scan CSVs.
    """
    values = np.asarray(
        (peak, sig_major, sig_minor, peak_error, sig_major_error, sig_minor_error),
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values[:3] <= 0):
        return float("nan"), float("nan")

    objective = sig_major * sig_minor / peak
    error = abs(objective) * np.sqrt(
        (peak_error / peak) ** 2
        + (sig_major_error / sig_major) ** 2
        + (sig_minor_error / sig_minor) ** 2
    )
    return float(objective), float(error)


def observable_minima(
    frame: pd.DataFrame, tx: np.ndarray, ry: np.ndarray
) -> pd.DataFrame:
    """Best knob point for each observable."""
    rows = []
    for column, err_column, name, _, mode in OBSERVABLES:
        values = to_grid(frame, column)
        errors = to_grid(frame, err_column)

        if mode == "neg_min":
            values = -values
        elif mode == "abs_min":
            values = np.abs(values)

        if mode == "zero":
            idx = np.unravel_index(np.nanargmin(np.abs(values)), values.shape)
            criterion = "closest to 0"
            value = values[idx]
        else:
            idx = np.unravel_index(np.nanargmin(values), values.shape)
            criterion = "maximum" if mode == "neg_min" else "minimum"
            value = -values[idx] if mode == "neg_min" else values[idx]

        rows.append(
            {
                "observable": name,
                "criterion": criterion,
                "TX [mm]": tx[idx],
                "RY [mrad]": ry[idx],
                "value": value,
                "error": errors[idx],
            }
        )
    return pd.DataFrame(rows)


def objective_minima(
    frame: pd.DataFrame, tx: np.ndarray, ry: np.ndarray
) -> pd.DataFrame:
    """Best knob point for each of the eight objectives."""
    objectives, errors = objective_grids(frame)
    rows = []
    for key in ORDER:
        values = objectives[key]
        idx = np.unravel_index(np.nanargmin(values), values.shape)
        rows.append(
            {
                "objective": key,
                "TX [mm]": tx[idx],
                "RY [mrad]": ry[idx],
                "minimum": values[idx],
                "propagated error": errors[key][idx],
            }
        )
    return pd.DataFrame(rows)
