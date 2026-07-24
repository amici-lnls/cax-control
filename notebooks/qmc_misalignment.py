"""
Quasi-random (low-discrepancy) 5D sampling of mirror misalignments (tx, ty, rx, ry, rz)
for the CAX beamline simulation.

Uses a Sobol sequence (scipy.stats.qmc) to fill the 5D parameter space far more
evenly than uniform random sampling for a given N -- important when each sample
is expensive (a full ray-trace + caustic scan).

Drop this next to your notebook, or paste the functions into a cell. It assumes
`cax`, `CAXSim`, and `acquisition_func` already exist as in cax_misalignments.ipynb.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol, scale
from shadowpy.cax_simulation import CAXSim
from shadowpy.utils import save_image

import os
import sys
sys.path.append(os.path.expanduser("~/repos/cax-scripts"))

from caxscripts.image_statistics import Histogram2DAnalyzer

# Fixed order used everywhere below -- keep consistent with the rest of the notebook
KNOBS = ["tx", "ty", "rx", "ry", "rz"]
OBSERVABLE_NAMES = ["cx", "cy", "fwhmx", "fwhmy", "peak", "astig"]


def sobol_samples(bounds: dict, n: int, seed: int | None = None) -> pd.DataFrame:
    """
    Generate N quasi-random points in 5D knob space using a Sobol sequence.

    Parameters
    ----------
    bounds : dict
        {"tx": (lo, hi), "ty": (lo, hi), "rx": (lo, hi), "ry": (lo, hi), "rz": (lo, hi)}
        Any subset/order of KNOBS is fine as long as all five are present.
    n : int
        Number of points to draw. Sobol sequences are best behaved (guaranteed
        balance properties) when n is a power of 2 -- Sobol() will pad up to the
        next power of 2 internally if you don't and just hand you n rows back, so
        this is a soft recommendation rather than a hard requirement.
    seed : int, optional
        For reproducibility. Sobol sequences are deterministic by construction;
        this only affects the optional scrambling.

    Returns
    -------
    pd.DataFrame with columns tx, ty, rx, ry, rz, one row per sample.
    """
    missing = set(KNOBS) - set(bounds)
    if missing:
        raise ValueError(f"Missing bounds for knob(s): {missing}")

    l_bounds = [bounds[k][0] for k in KNOBS]
    u_bounds = [bounds[k][1] for k in KNOBS]

    sampler = Sobol(d=5, scramble=True, seed=seed)
    unit_samples = sampler.random(n)  # shape (n, 5), values in [0, 1)
    real_samples = scale(unit_samples, l_bounds, u_bounds)

    return pd.DataFrame(real_samples, columns=KNOBS)


def run_qmc_scan(
    cax,
    bounds: dict,
    n: int,
    acquisition_func,
    seed: int | None = None,
    out_csv: str | Path | None = "qmc_scan_results.csv",
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """
    Sample N points over the 5D (tx, ty, rx, ry, rz) space with a Sobol sequence,
    run the simulation/acquisition at each point, and collect the results.

    Parameters
    ----------
    cax : CAXSim
        The simulation object (must expose cax.mirror.{tx,ty,rx,ry,rz}).
    bounds : dict
        Per-knob (lo, hi) ranges, see sobol_samples().
    n : int
        Number of points to sample.
    acquisition_func : callable
        Function of cax -> (cx, cy, fwhmx, fwhmy, peak, astig), same signature
        as in the notebook.
    seed : int, optional
        Seed for the Sobol scrambling (reproducibility).
    out_csv : str or Path or None
        If given, results are checkpointed to this CSV as the scan runs (so you
        don't lose everything if a long scan errors out partway through), and
        the final result is written there too. Pass None to skip file I/O.
    checkpoint_every : int
        Write to out_csv every this-many points (in addition to at the end).

    Returns
    -------
    pd.DataFrame with columns tx, ty, rx, ry, rz, cx, cy, fwhmx, fwhmy, peak,
    astig, plus `point_index` and `duration_s` for bookkeeping.
    """
    samples = sobol_samples(bounds, n, seed=seed)
    records = []

    for i, row in samples.iterrows():
        for knob in KNOBS:
            setattr(cax.mirror, knob, row[knob])

        t0 = time.time()
        try:
            obs = acquisition_func(cax)
        except Exception as exc:
            print(f"[{i+1:4d}/{n}] acquisition failed: {exc!r}")
            obs = (np.nan,) * len(OBSERVABLE_NAMES)
        dt = time.time() - t0

        record = {"point_index": i, **row.to_dict()}
        record.update(dict(zip(OBSERVABLE_NAMES, obs)))
        record["duration_s"] = dt
        records.append(record)

        print(f"[{i+1:4d}/{n}] " + ", ".join(f"{k}={row[k]:+.4f}" for k in KNOBS) +
              f"  ({dt:.1f}s)")

        if out_csv is not None and (i + 1) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(out_csv, index=False)

    df = pd.DataFrame(records)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} points to {out_csv}")

    return df

# ================== ACQUISITION ================== #

def _astigmatism_acquisition(cax: CAXSim):

    # Initial search

    low_lim = -1000
    hgh_lim =  1000

    done = False

    while not done:
        distances, fwhm_x, fwhm_y = \
            cax.dvf_B1.parallel_caustic(cax.dvf_B1.beam.duplicate(),
                                        s_range=(low_lim, hgh_lim),
                                        n_points=50, 
                                        max_workers=10)
        # prior focal lengths
        fx = distances[np.nanargmin(fwhm_x)]
        fy = distances[np.nanargmin(fwhm_y)]

        print(fx, fy)

        done = True

        # check if the search was enought otherwise repeat
        if low_lim in [fx, fy]:
            low_lim -= 100
            done = False

        if hgh_lim in [fx, fy]:
            hgh_lim += 100
            done = False

    # Finer search
    distances, fwhm_x, fwhm_y = \
    cax.dvf_B1.parallel_caustic(cax.dvf_B1.beam.duplicate(),
                                s_range=(np.min([fx, fy])-50, np.max([fx, fy])+50),
                                n_points=100, 
                                max_workers=10)

    # actual focal lengths
    fx = distances[np.nanargmin(fwhm_x)]
    fy = distances[np.nanargmin(fwhm_y)]
    print(fx, fy)

    f = np.mean([fx, fy])
    retr_beam = cax.dvf_B1.beam.duplicate()
    retr_beam.retrace(f)
    ana = save_image(cax.dvf_B1, retr_beam)
    astig = fy - fx

    return distances, fwhm_x, fwhm_y, fx, fy, f, astig, ana 

def image_peak(image: Histogram2DAnalyzer):
    
    
    # print("image_peak: hprm_momenta type -- ", type(image.hprm_momenta))
    total = image.img.sum()

    fx = image.hprm_fitting['fwhmx']
    fy = image.hprm_fitting['fwhmy']

    I = total/(fx*fy*1000**2)
    return I

def acquisition_func(cax: CAXSim):
    
    # What observables do I want? centroid (x, y), fwhm (x, y), peak (roi avg), astig
    distances, fwhm_x, fwhm_y, fx, fy, f, astig, ana = _astigmatism_acquisition(cax)
    if not ana.beam_visible:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    peak = image_peak(ana)
    cx = ana.hprm_fitting['mux']
    cy = ana.hprm_fitting['muy']

    fwhmx = ana.hprm_fitting['fwhmx']
    fwhmy = ana.hprm_fitting['fwhmy']

    return cx, cy, fwhmx, fwhmy, peak, astig




if __name__ == "__main__":
    # Example usage -- adapt to your notebook's CAXSim / acquisition_func imports.
    cax = CAXSim(total_rays=100000)

    # from your_notebook_module import acquisition_func  # or paste it in above

    bounds = {
        "tx": (-1.00, 1.00),     # mm
        "ty": (-1.00, 1.00),     # mm
        "rx": (-0.05, 0.05),     # mrad
        "ry": (-0.05, 0.05),     # mrad
        "rz": (-0.05, 0.05),     # mrad
    }

    N = 128  # power of 2 recommended for Sobol

    df = run_qmc_scan(cax, bounds, N, acquisition_func, seed=42,
                        out_csv="qmc_scan_results.csv", checkpoint_every=10)