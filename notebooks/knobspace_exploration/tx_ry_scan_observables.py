"""TX-RY knob-space scan with uncertainty-aware observables.

This script performs an 81 x 81 scan of the CAX mirror knobs:

- ``TX`` from -2 mm to +2 mm
- ``RY`` from -0.1 mrad to +0.1 mrad (Shadow expects mrad)

At every knob point it acquires beam images at two DVF positions:

1. ``z = 0``, the default ``dvf_B1`` location;
2. ``z = -142 mm``, reached by retracing a copy of the ``dvf_B1`` beam to
   that plane.

For each image, ``caustic.fit_beam_image()`` runs the ``Histogram2DAnalyzer``
pipeline and the script records:

| Observable | Source | Unit |
|---|---|---|
| ``centroid_x/y`` | ``hprm_fitting["mux"/"muy"]`` | um |
| ``sigma_x/y`` | ``hprm_fitting["sigx"/"sigy"]`` | um |
| ``Intensity`` | ``hprm_fitting["amplitude"]`` | counts / um^2 |
| uncertainties | ``hprm_fitting["errors"]`` | same as observable |

A 10-point caustic scan is also run once per knob point. The fitted foci are

- ``q_x = result.focus_x``
- ``q_y = result.focus_y``
- ``q_x_err = sqrt(result.x.covariance[1, 1])``
- ``q_y_err = sqrt(result.y.covariance[1, 1])``

These foci are independent of the DVF z position, so the same values are
written to both output files. Astigmatism is stored as ``q_y - q_x`` with a
simple propagated uncertainty.

The two outputs are timestamped at run start:

- ``tx_ry_scan_81x81_z0_<timestamp>.csv``
- ``tx_ry_scan_81x81_zfocus_<timestamp>.csv``

Both are checkpointed while the scan runs, and the script resumes from the
points already present in both files.
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _find_repo_root(start):
    """Find the checkout that contains caustic.py and image_statistics.py."""
    for candidate in [start, *start.parents]:
        if (candidate / "caustic.py").exists() and (candidate / "image_statistics.py").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path.cwd().resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.caustic import fit_beam_image, image_caustic
from shadowpy.si_beamlines import CAXSim
from shadowpy.utils import save_image

# The analyzer warns when a beam is not visible; failures are recorded as NaN
# and handled explicitly by the acquisition helpers below.
warnings.filterwarnings("ignore", message="Beam not visible.*")

# ----------------------------- scan grid ----------------------------- #
N_TX = 81
N_RY = 81
TX_VALUES = np.linspace(-2.0, 2.0, N_TX)          # mm
RY_VALUES = np.linspace(-0.1, 0.1, N_RY)          # mrad (= -100 ... +100 urad)

# --------------------------- DVF locations --------------------------- #
Z_FOCUS = -142.0                                  # mm

# ------------------------------ caustic ------------------------------ #
CAUSTIC_DISTANCES = np.linspace(-450.0, 150.0, 10)  # mm, 10 points
MAX_WORKERS = 1

# ---------------------------- acquisition ---------------------------- #
TOTAL_RAYS = 100_000
IMAGE_SIZE = (210, 280)

# ------------------------------ outputs ------------------------------ #
OUT_DIR = REPO_ROOT / "notebooks" / "knobspace_exploration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Generated once per run. To resume a previous run, set this to that run's
# timestamp (the value printed below and embedded in the CSV filenames).
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_Z0 = OUT_DIR / f"tx_ry_scan_81x81_z0_{RUN_TIMESTAMP}.csv"
OUT_ZF = OUT_DIR / f"tx_ry_scan_81x81_zfocus_{RUN_TIMESTAMP}.csv"

CHECKPOINT_EVERY = 10
RESUME = True

# ----------------------------- observables --------------------------- #
IMAGE_STAT_NAMES = [
    "centroid_x_um", "centroid_x_err_um",
    "centroid_y_um", "centroid_y_err_um",
    "sigma_x_um", "sigma_x_err_um",
    "sigma_y_um", "sigma_y_err_um",
    "Intensity", "Intensity_err",
    "reduced_chi2",
]
COMMON_COLUMNS = [
    "point_index", "ix", "iy", "TX_mm", "RY_mrad",
    "q_x_mm", "q_x_err_mm", "q_y_mm", "q_y_err_mm",
    "astigmatism_mm", "astigmatism_err_mm",
    "duration_s",
]
INTERNAL_Z0_COLUMNS = [f"z0_{name}" for name in IMAGE_STAT_NAMES]
INTERNAL_ZF_COLUMNS = [f"zf_{name}" for name in IMAGE_STAT_NAMES]
ALL_RECORD_COLUMNS = COMMON_COLUMNS + INTERNAL_Z0_COLUMNS + INTERNAL_ZF_COLUMNS

print(f"Scan: {N_TX} x {N_RY} = {N_TX * N_RY} points")
print(f"TX: {TX_VALUES[0]:+.3f} ... {TX_VALUES[-1]:+.3f} mm")
print(f"RY: {RY_VALUES[0] * 1e3:+.1f} ... {RY_VALUES[-1] * 1e3:+.1f} urad")
print(f"Caustic: {CAUSTIC_DISTANCES[0]:+.1f} ... {CAUSTIC_DISTANCES[-1]:+.1f} mm, "
      f"{CAUSTIC_DISTANCES.size} points")
print(f"z=0 output      : {OUT_Z0}")
print(f"z=focus output  : {OUT_ZF}")

cax = CAXSim(total_rays=TOTAL_RAYS, image_size=IMAGE_SIZE)
print(f"CAXSim ready: {TOTAL_RAYS} rays, image_size={IMAGE_SIZE}")


def _nan_image_stats():
    return {name: np.nan for name in IMAGE_STAT_NAMES}


def _as_float(mapping, key):
    value = mapping.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def image_stats(image):
    """Fit one image and return centroid, sigma, peak amplitude and errors."""
    try:
        ana = fit_beam_image(image)
        fit = getattr(ana, "hprm_fitting", None)
        if fit is None:
            return _nan_image_stats()

        errors = fit.get("errors") or {}
        return {
            "centroid_x_um": _as_float(fit, "mux"),
            "centroid_x_err_um": _as_float(errors, "mux"),
            "centroid_y_um": _as_float(fit, "muy"),
            "centroid_y_err_um": _as_float(errors, "muy"),
            "sigma_x_um": _as_float(fit, "sigx"),
            "sigma_x_err_um": _as_float(errors, "sigx"),
            "sigma_y_um": _as_float(fit, "sigy"),
            "sigma_y_err_um": _as_float(errors, "sigy"),
            "Intensity": _as_float(fit, "amplitude"),
            "Intensity_err": _as_float(errors, "amplitude"),
            "reduced_chi2": _as_float(fit, "reduced_chi2"),
        }
    except Exception as exc:
        print(f"    image fit failed: {exc!r}")
        return _nan_image_stats()


def _nan_caustic_stats():
    return {
        "q_x_mm": np.nan,
        "q_x_err_mm": np.nan,
        "q_y_mm": np.nan,
        "q_y_err_mm": np.nan,
        "astigmatism_mm": np.nan,
        "astigmatism_err_mm": np.nan,
    }


def _focus_uncertainty(axis_result):
    covariance = getattr(axis_result, "covariance", None)
    if covariance is None:
        return np.nan
    try:
        value = float(covariance[1, 1])
    except (TypeError, ValueError, IndexError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return float(np.sqrt(max(value, 0.0)))


def caustic_stats(beam, frame):
    """Run the 10-point caustic scan and return both foci and astigmatism."""
    try:
        result = image_caustic(
            beam,
            frame,
            CAUSTIC_DISTANCES,
            image_size=IMAGE_SIZE,
            max_workers=MAX_WORKERS,
        )
        qx = float(result.focus_x)
        qy = float(result.focus_y)
        qx_err = _focus_uncertainty(result.x)
        qy_err = _focus_uncertainty(result.y)
        astig = qy - qx
        astig_err = (
            float(np.hypot(qx_err, qy_err))
            if np.isfinite(qx_err) and np.isfinite(qy_err)
            else np.nan
        )
        return {
            "q_x_mm": qx,
            "q_x_err_mm": qx_err,
            "q_y_mm": qy,
            "q_y_err_mm": qy_err,
            "astigmatism_mm": astig,
            "astigmatism_err_mm": astig_err,
        }
    except Exception as exc:
        print(f"    caustic failed: {exc!r}")
        return _nan_caustic_stats()


def acquire_point(cax, tx, ry):
    """Measure both DVF locations and the caustic for one (TX, RY) point."""
    cax.mirror.tx = tx
    cax.mirror.ry = ry
    cax.trace()

    # z = 0: the image already produced at the DVF.
    stats_z0 = image_stats(cax.dvf_B1.image)

    # qx/qy are independent of the DVF position: measure them once.
    caus = caustic_stats(cax.dvf_B1.beam, cax.dvf_B1.frame)

    # z = Z_FOCUS: retrace a copy of the DVF beam upstream. Keep this
    # failure isolated so a bad focus image does not discard the caustic.
    try:
        beam_focus = cax.dvf_B1.beam.duplicate()
        beam_focus.retrace(Z_FOCUS)
        image_focus = save_image(cax.dvf_B1, beam_focus, image_size=IMAGE_SIZE)
        del beam_focus
        stats_focus = image_stats(image_focus)
    except Exception as exc:
        print(f"    focus image acquisition failed: {exc!r}")
        stats_focus = _nan_image_stats()

    return stats_z0, stats_focus, caus


def _z0_frame(records):
    frame = records[COMMON_COLUMNS + INTERNAL_Z0_COLUMNS].copy()
    return frame.rename(columns={f"z0_{name}": name for name in IMAGE_STAT_NAMES})


def _zf_frame(records):
    frame = records[COMMON_COLUMNS + INTERNAL_ZF_COLUMNS].copy()
    return frame.rename(columns={f"zf_{name}": name for name in IMAGE_STAT_NAMES})


def write_outputs(records):
    """Write the two requested CSVs from the combined in-memory records."""
    if records.empty:
        return
    records = records.sort_values("point_index").reset_index(drop=True)
    _z0_frame(records)[COMMON_COLUMNS + IMAGE_STAT_NAMES].to_csv(OUT_Z0, index=False)
    _zf_frame(records)[COMMON_COLUMNS + IMAGE_STAT_NAMES].to_csv(OUT_ZF, index=False)


def load_existing_records():
    """Load and recombine previous results, or return an empty table."""
    empty = pd.DataFrame(columns=ALL_RECORD_COLUMNS)
    if not RESUME or not OUT_Z0.exists() or not OUT_ZF.exists():
        return empty

    try:
        z0 = pd.read_csv(OUT_Z0)
        zf = pd.read_csv(OUT_ZF)
    except Exception as exc:
        print(f"Could not read previous outputs ({exc!r}); starting fresh.")
        return empty

    if z0.empty or zf.empty:
        return empty

    z0 = z0.rename(columns={name: f"z0_{name}" for name in IMAGE_STAT_NAMES})
    zf = zf.rename(columns={name: f"zf_{name}" for name in IMAGE_STAT_NAMES})

    zf_columns = ["point_index"] + INTERNAL_ZF_COLUMNS
    merged = z0.merge(zf[zf_columns], on="point_index", how="inner")
    return merged[ALL_RECORD_COLUMNS]


# ----------------------------- run the scan ----------------------------- #
# At roughly 5.5 s per point, the full 6561-point scan takes about 10 hours.
# The loop prints the elapsed time and an ETA after every point.
#
# Both CSVs are checkpointed every CHECKPOINT_EVERY points. If the process
# dies, rerun the script from the top: points already present in both files
# are skipped automatically.

records = load_existing_records()
completed = set(records["point_index"].astype(int)) if not records.empty else set()

if completed:
    print(f"Resuming: {len(completed)} points already present in both CSVs.")

total_points = N_TX * N_RY
starting_count = len(records)
scan_start = time.perf_counter()

for ix, tx in enumerate(TX_VALUES):
    for iy, ry in enumerate(RY_VALUES):
        point_index = ix * N_RY + iy
        if point_index in completed:
            continue

        point_start = time.perf_counter()
        try:
            stats_z0, stats_focus, caus = acquire_point(cax, tx, ry)
        except Exception as exc:
            print(f"[{point_index:4d}] acquisition failed: {exc!r}")
            stats_z0 = _nan_image_stats()
            stats_focus = _nan_image_stats()
            caus = _nan_caustic_stats()

        duration = time.perf_counter() - point_start

        record = {
            "point_index": point_index,
            "ix": ix,
            "iy": iy,
            "TX_mm": float(tx),
            "RY_mrad": float(ry),
            "duration_s": duration,
        }
        record.update(caus)
        record.update({f"z0_{name}": value for name, value in stats_z0.items()})
        record.update({f"zf_{name}": value for name, value in stats_focus.items()})

        records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
        completed.add(point_index)

        processed = len(records) - starting_count
        elapsed = time.perf_counter() - scan_start
        rate = elapsed / max(processed, 1)
        eta = rate * (total_points - len(records))

        print(
            f"[{len(records):4d}/{total_points}] "
            f"TX={tx:+.4f} mm RY={ry * 1e3:+6.1f} urad | "
            f"{duration:5.2f}s | elapsed {elapsed / 3600:5.2f}h | "
            f"ETA {eta / 3600:5.2f}h"
        )

        if processed % CHECKPOINT_EVERY == 0:
            write_outputs(records)
            print(f"    checkpoint written ({len(records)} points)")

write_outputs(records)
print(f"Finished: {len(records)}/{total_points} points written.")
print(f"  z=0       -> {OUT_Z0}")
print(f"  z={Z_FOCUS:g} mm -> {OUT_ZF}")

# ----------------------------- verification ----------------------------- #
df_z0 = pd.read_csv(OUT_Z0)
df_zf = pd.read_csv(OUT_ZF)

print(f"z=0 file     : {df_z0.shape[0]} rows x {df_z0.shape[1]} columns")
print(f"z=focus file : {df_zf.shape[0]} rows x {df_zf.shape[1]} columns")

expected = N_TX * N_RY
assert len(df_z0) == expected, f"Expected {expected} rows in z=0 file, found {len(df_z0)}"
assert len(df_zf) == expected, f"Expected {expected} rows in focus file, found {len(df_zf)}"
assert df_z0["point_index"].is_unique
assert df_zf["point_index"].is_unique
assert np.allclose(df_z0["TX_mm"], df_zf["TX_mm"], equal_nan=True)
assert np.allclose(df_z0["RY_mrad"], df_zf["RY_mrad"], equal_nan=True)
assert np.allclose(df_z0["q_x_mm"], df_zf["q_x_mm"], equal_nan=True)
assert np.allclose(df_z0["q_y_mm"], df_zf["q_y_mm"], equal_nan=True)

print("\nNaN counts (z=0 / zfocus):")
for name in IMAGE_STAT_NAMES:
    print(f"  {name:20s} {df_z0[name].isna().sum():5d} / {df_zf[name].isna().sum():5d}")
for name in ["q_x_mm", "q_y_mm", "astigmatism_mm"]:
    print(f"  {name:20s} {df_z0[name].isna().sum():5d} / {df_zf[name].isna().sum():5d}")

# --------------------------- commit and push --------------------------- #
# NOTE: Disabled to avoid committing/pushing automatically from the script.
#
# from datetime import datetime
#
# timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# commit_message = f"Add 81x81 TX-RY scan results ({timestamp})"
#
# commands = [
#     f'git -C "{REPO_ROOT}" add -A',
#     f'git -C "{REPO_ROOT}" commit -m "{commit_message}"',
#     f'git -C "{REPO_ROOT}" push',
# ]
#
# for command in commands:
#     print(f"$ {command}")
#     exit_code = os.system(command)
#     print(f"  exit code: {exit_code}")
