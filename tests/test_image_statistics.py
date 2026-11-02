"""Fast regression tests for Histogram2DAnalyzer.

These tests cover the parts of the analysis that used to depend on binning:

* peak amplitude and baseline are reported per unit bin area;
* the fit never returns a negative baseline;
* tilted beams are described correctly by marginal projections;
* the same physical Gaussian gives consistent results after rebinning.

The slower CAXSim/Shadow validation is intentionally not part of this module.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.expanduser("~/repos/cax-control"))
sys.path.append(os.path.expanduser("~/repos/shadowpy"))

from caustic import fit_caustic_hyperbola
from image_statistics import FWHM2SIG, Histogram2DAnalyzer
from shadowpy.utils import Image


HIST_RANGE = [[-8.0, 8.0], [-8.0, 8.0]]


def rotated_cov(sig_major, sig_minor, theta_deg):
    """Return a 2x2 covariance matrix rotated by ``theta_deg``."""
    theta = np.deg2rad(theta_deg)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    return rotation @ np.diag([sig_major**2, sig_minor**2]) @ rotation.T


def test_bin_attributes_and_aliases():
    """Bin sizes and area are exposed and stay consistent with old names."""
    ana = Histogram2DAnalyzer(
        np.ones((10, 20)),
        x_bin_edges=np.linspace(0.0, 2.0, 11),
        y_bin_edges=np.linspace(-1.0, 3.0, 21),
    )

    assert ana.x_bin_size == pytest.approx(0.2)
    assert ana.y_bin_size == pytest.approx(0.2)
    assert ana.bin_area == pytest.approx(0.04)
    assert ana.x_bin_width == ana.x_bin_size
    assert ana.y_bin_width == ana.y_bin_size


def test_peak_and_offset_are_per_bin_area():
    """Projection and moment amplitudes scale back to counts per bin."""
    ana = Histogram2DAnalyzer.from_gaussian(
        bins=90, size=250_000, seed=11
    )

    project_counts = ana.hprm_project["amplitude"] * ana.bin_area
    moment_counts = ana.compute_momenta()["amplitude"] * ana.bin_area

    expected_peak = (250_000 * ana.bin_area
                     / (2 * np.pi * np.sqrt(np.linalg.det(np.eye(2)))))
    assert project_counts == pytest.approx(expected_peak, rel=0.15)
    assert moment_counts == pytest.approx(expected_peak, rel=0.10)


def test_rotated_beam_uses_marginal_projections():
    """Projection widths must not collapse to a narrow central slice."""
    cov = rotated_cov(1.0, 0.3, 45.0)
    ana = Histogram2DAnalyzer.from_gaussian(
        bins=100, size=300_000, cov=cov, hist_range=HIST_RANGE, seed=3
    )

    true_sig = np.sqrt(np.diag(cov))
    proj = ana.hprm_project
    assert proj["sigx"] == pytest.approx(true_sig[0], rel=0.05)
    assert proj["sigy"] == pytest.approx(true_sig[1], rel=0.05)
    assert proj["fwhmx"] == pytest.approx(FWHM2SIG * true_sig[0], rel=0.05)

    # A correct ROI must retain essentially all of the beam.
    roi_fraction = ana.img[ana.roi_mask].sum() / ana.img.sum()
    assert roi_fraction > 0.99


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 45.0, 90.0])
def test_fit_recovers_rotated_gaussian(theta_deg):
    """The bounded fit recovers widths and gives finite uncertainties."""
    cov = rotated_cov(1.0, 0.35, theta_deg)
    ana = Histogram2DAnalyzer.from_gaussian(
        bins=100, size=250_000, cov=cov, hist_range=HIST_RANGE, seed=7
    )
    ana.add_noise(seed=8)

    momenta = ana.compute_momenta()
    fit = ana.fit(hprm=momenta)
    true_sig = np.sqrt(np.diag(cov))

    assert fit["sigx"] == pytest.approx(true_sig[0], rel=0.05)
    assert fit["sigy"] == pytest.approx(true_sig[1], rel=0.05)
    assert fit["offset"] >= 0.0
    assert fit["amplitude"] > 0.0
    assert np.isfinite(fit["reduced_chi2"])
    assert fit["dof"] > 0

    for key in ("offset", "amplitude", "mux", "muy", "sigx", "sigy",
                "rho", "fwhmx", "fwhmy", "sig_major", "sig_minor", "theta"):
        assert np.isfinite(fit["errors"][key]), key
    assert fit["errors"]["sigx"] > 0.0
    assert fit["errors"]["amplitude"] > 0.0


def test_zero_background_does_not_fit_negative_offset():
    """Clean zero-baseline histograms must keep the fitted offset >= 0."""
    ana = Histogram2DAnalyzer.from_gaussian(
        bins=80, size=150_000, seed=19
    )
    fit = ana.fit(hprm=ana.compute_momenta())

    assert fit["offset"] >= 0.0
    # The true baseline is zero; allow a few thousandths of the peak.
    assert fit["offset"] < 0.05 * fit["amplitude"]


def test_fit_is_consistent_under_rebinning():
    """Amplitude density and sigmas stay stable when bins are merged."""
    cov = rotated_cov(1.0, 0.35, 35.0)
    rng = np.random.default_rng(5)
    samples = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=900_000)

    fits = []
    for bins in (50, 100, 200):
        hist, x_edges, y_edges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=bins, range=HIST_RANGE
        )
        ana = Histogram2DAnalyzer(
            hist, x_bin_edges=x_edges, y_bin_edges=y_edges
        )
        fits.append(ana.fit(hprm=ana.compute_momenta()))

    amplitude_density = np.array([f["amplitude"] for f in fits])
    sigx = np.array([f["sigx"] for f in fits])
    sigy = np.array([f["sigy"] for f in fits])

    assert np.ptp(amplitude_density) / amplitude_density.mean() < 0.08
    assert np.ptp(sigx) / sigx.mean() < 0.03
    assert np.ptp(sigy) / sigy.mean() < 0.03


def test_from_gaussian_is_reproducible_with_seed():
    """The optional seed makes generated histograms deterministic."""
    first = Histogram2DAnalyzer.from_gaussian(bins=40, size=5_000, seed=1)
    second = Histogram2DAnalyzer.from_gaussian(bins=40, size=5_000, seed=1)

    np.testing.assert_array_equal(first.img, second.img)


def test_fit_caustic_hyperbola_recovers_focus_and_waist():
    """A weighted hyperbola fit should recover the underlying caustic."""
    distances = np.linspace(-30.0, 30.0, 31)
    true_d0 = 4.5
    true_amp = 18.0
    true_s0 = 12.0
    widths = true_amp * np.sqrt(1.0 + ((distances - true_d0) / true_s0) ** 2)
    sigmas = 0.25 + 0.05 * np.abs(widths)
    rng = np.random.default_rng(7)
    noisy = widths + rng.normal(0.0, sigmas)

    params, _ = fit_caustic_hyperbola(distances, noisy, sigmas, bounds=([
        0.0, -100.0, 1e-6
    ], [100.0, 100.0, 1000.0]))

    assert params[1] == pytest.approx(true_d0, abs=2.0)
    assert params[0] == pytest.approx(true_amp, rel=0.15)
    assert params[2] == pytest.approx(true_s0, rel=0.25)


def test_image_from_ticket_accepts_missing_fwhm_values():
    """Missing confounding FWHM metadata should not crash ticket conversion."""
    image = Image.from_ticket({
        "histogram": np.ones((3, 3)),
        "bin_h_edges": np.linspace(0.0, 3.0, 4),
        "bin_v_edges": np.linspace(0.0, 3.0, 4),
        "bin_h_center": np.linspace(0.5, 2.5, 3),
        "bin_v_center": np.linspace(0.5, 2.5, 3),
        "fwhm_h": None,
        "fwhm_v": None,
        "fwhm_coordinates_h": None,
        "fwhm_coordinates_v": None,
    })

    assert np.isnan(image.sigma_h)
    assert np.isnan(image.sigma_v)
    assert np.isnan(image.centroid_h)
    assert np.isnan(image.centroid_v)
