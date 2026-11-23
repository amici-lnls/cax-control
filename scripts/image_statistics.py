"""Module for generating and analyzing 2D distribution histograms.

All functionality lives inside Histogram2DAnalyzer:

  Construction
  - Histogram2DAnalyzer(img, x_bin_edges, y_bin_edges)
            — from an existing histogram.
  - Histogram2DAnalyzer.from_gaussian(...)
            — generate a Gaussian histogram.

  Quick Analysis (run on init)
  - parameters_from_projections() — marginal-projection centroid and width.
  - beam_visible        — attribute: whether beam is visible.
  - hprm_project        — attribute: projection parameters dict.

  Full Analysis
  - compute_momenta(img)   — weighted moments and principal-axis parameters.
  - fit(hprm)              — bounded 2D Gaussian fit with Poisson weights;
                             reports parameter uncertainties and chi-square.
  - compute_threshold()    — Kapur entropy thresholding within the ROI.

  Peak heights (amplitude and offset) are always reported per unit bin area,
  i.e. counts per bin divided by bin_area. This makes them independent of
  histogram binning.

  ROI
  - compute_roi(hprm)      — rebuild the projection-derived ROI mask.
  - roi_mask               — attribute: boolean ROI mask.
  - roi_bounds             — attribute: (xmin, xmax, ymin, ymax) for plotting.

  Display
  - print_stats(hprm)
  - plot(hprm, ...)        — shows and zooms to the ROI by default.
  - plot_entropy(entropies, bin_edges, optimal_threshold)

  Orchestration
  - analyze(mode)        — run analysis pipeline;
                           mode in ('quick', 'moments', 'fit', 'all').
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------

DEFAULT_MEAN = [0, 0]
DEFAULT_COV = [[1, 0], [0, 1]]
DEFAULT_HIST_RANGE = [[-8, 8], [-8, 8]]
COLORS = {"in": "white", "mid": "green", "out": "orange"}
ELLIPSES = ((1, COLORS["in"]), (2, COLORS["mid"]), (3, COLORS["out"]))
ELLIPSES_TITLE = "2D Histogram with Ellipses"

# Threshold for peak-to-average ratio acceptance of image.
# PEAK2AVG_THRESHOLD = 100
PEAK2AVG_THRESHOLD = 5

# FWHM <-> sigma conversion: FWHM = f2sig * sigma
FWHM2SIG = 2 * np.sqrt(2 * np.log(2))

# ---------------------------------------------------------------------------
# Fitting defaults.
# ---------------------------------------------------------------------------

# A Gaussian cannot be resolved below roughly one bin, so the fitter is not
# allowed to collapse to a sub-bin spike. The floor is expressed in bins.
SIGMA_MIN_BINS = 0.5

# Correlations are kept strictly inside the unit interval so that the
# covariance matrix remains positive definite.
RHO_MAX = 1.0 - 1e-6

# Extra padding, in bins, added to the projection-derived ROI on each side.
DEFAULT_ROI_PAD = 2.0

# Number of moment-based ROI refinements attempted before analysis continues.
ROI_REFINE_ITERATIONS = 3

# Maximum number of function evaluations for the nonlinear fit.
FIT_MAXFEV = 20000

# The fitted offset is bounded above by this many Poisson sigmas above the
# robust background estimate, preventing the baseline from absorbing the beam.
OFFSET_UPPER_NSIGMA = 5.0

# The fitted amplitude is bounded above by this multiple of the brightest bin.
AMPLITUDE_MAX_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Analysis pipeline class.
# ---------------------------------------------------------------------------


class Histogram2DAnalyzer:
    """Self-contained analysis pipeline for a single 2D histogram.

    Stores the image, bin edges, and intermediate results (projection,
    moments, fit) as instance attributes.

    Usage::

        # From an existing histogram:
        ana = Histogram2DAnalyzer(hist, x_bin_edges, y_bin_edges)

        # Or generate a Gaussian test histogram directly:
        ana = Histogram2DAnalyzer.from_gaussian(bins=80, size=200_000)
        ana.add_noise()

        hprm = ana.compute_moments()
        ana.print_stats()
        fig, ax = ana.plot()
        plt.show()

        entropies, bin_edges, thr, nbins = ana.compute_threshold()
        fig = ana.plot_entropy(entropies, bin_edges, thr)
        plt.show()

        hprm = ana.compute_moments(img=ana.img_thresholded)
        hprm = ana.fit()
    """

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    def __init__(
        self,
        img,
        droi=4,
        x_bin_edges=None,
        y_bin_edges=None,
        roi_sigma=5.0,
        roi_pad=DEFAULT_ROI_PAD,
    ):
        """Initialise a 2D histogram with optional physical bin edges.

        Arguments:
            img: 2D histogram array of counts per bin. Amplitude-like
                results are returned per unit bin area.
            x_bin_edges: optional uniform x edges; defaults to bin-index
                coordinates from 0 to img.shape[0].
            y_bin_edges: optional uniform y edges; defaults to bin-index
                coordinates from 0 to img.shape[1].
            droi: nonnegative integer radius, in bins, for local ROIs.
            roi_sigma: half-width of the analysis ROI in projected sigmas.
                Defaults to 5.0, i.e. a 5-sigma rectangular ROI.
            roi_pad: extra padding, in bins, added to each side of the
                roi_sigma ROI. Defaults to DEFAULT_ROI_PAD.

        Runs parameters_from_projections() automatically to populate
        beam_visible and hprm_project.
        """
        self.img = np.asarray(img, dtype=float)
        if self.img.ndim != 2 or min(self.img.shape) == 0:
            raise ValueError("img must be a non-empty 2D array.")
        if not isinstance(droi, (int, np.integer)) or droi < 0:
            raise ValueError("droi must be a nonnegative integer number of bins.")
        try:
            roi_sigma = float(roi_sigma)
        except (TypeError, ValueError) as err:
            raise ValueError("roi_sigma must be a positive finite number.") from err
        if not np.isfinite(roi_sigma) or roi_sigma <= 0:
            raise ValueError("roi_sigma must be a positive finite number.")
        try:
            roi_pad = float(roi_pad)
        except (TypeError, ValueError) as err:
            raise ValueError("roi_pad must be a nonnegative finite number.") from err
        if not np.isfinite(roi_pad) or roi_pad < 0:
            raise ValueError("roi_pad must be a nonnegative finite number.")
        self.droi              = droi
        self.roi_sigma         = roi_sigma
        self.roi_pad           = roi_pad
        self.roi_mask          = None
        self.roi_bounds        = None
        self.beam_visible      = None  # set by parameters_from_projections()
        # projection params: centroid, FWHM, sigma and diagonal covariance
        self.hprm_project      = None
        self.hprm_momenta      = None  # momenta params
        self.hprm_fitting      = None  # fitting params
        self.hprm              = None
        self.optimal_threshold = np.max(img) * 0.1
        self.img_thresholded   = None
        self.data              = None
        self.pixel_size        = 1
        self._bin_centers(x_bin_edges, y_bin_edges)

        # Run quick analysis automatically
        self.parameters_from_projections()

    # ------------------------------------------------------------------
    # Quick analysis block (run on init).
    # ------------------------------------------------------------------

    def _compute_beam_visibility(self, cx, cy):
        """Compute whether the beam is visible based on peak-to-average ratio.

        Args:
            cx: x-coordinate of the beam centroid in edge-coordinate units.
            cy: y-coordinate of the beam centroid in edge-coordinate units.

        Returns:
            bool: True if the beam is considered visible.
        """
        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")
        roi = self.img[
            max(0, ix - self.droi):min(self.img.shape[0], ix + self.droi + 1),
            max(0, iy - self.droi):min(self.img.shape[1], iy + self.droi + 1),
        ]
        roi_avg = np.mean(roi)
        mean = np.mean(self.img)
        ratio = roi_avg / mean if mean != 0 else 0
        return ratio >= PEAK2AVG_THRESHOLD

    def _centroid_from_projection(self):
        """Return the peak-projection centroid in bin-edge coordinates.

        Returns:
            np.array: [cx, cy] bin-center coordinates.
        """
        # image is being transposed before being attributed to the class
        # then, its shape is (nx, ny) and the x projection is along axis=1,
        # while the y projection along axis=0
        cx_sum = np.sum(self.img, axis=1)
        cy_sum = np.sum(self.img, axis=0)
        def smooth_projection(projection):
            window = min(21, projection.size if projection.size % 2 else projection.size - 1)
            if window < 3:
                return projection
            return savgol_filter(projection, window_length=window, polyorder=2)

        ix = np.argmax(smooth_projection(cx_sum))
        iy = np.argmax(smooth_projection(cy_sum))
        return np.array([self.x_bin_centers[ix], self.y_bin_centers[iy]])

    @staticmethod
    def _profile_fwhm(profile, centers):
        """Estimate the FWHM of a one-dimensional profile.

        The profile baseline is estimated from the outer tenth of each
        side, which is robust for a beam surrounded by background. The
        crossing points are linearly interpolated between bin centres.

        Args:
            profile: 1D array of profile values.
            centers: 1D array of matching bin-centre coordinates.

        Returns:
            float: FWHM in the coordinate units of ``centers``.
        """
        profile = np.asarray(profile, dtype=float)
        centers = np.asarray(centers, dtype=float)
        if profile.size < 2 or centers.size != profile.size:
            return 0.0

        edge = max(1, profile.size // 10)
        baseline = 0.5 * (
            np.median(profile[:edge]) + np.median(profile[-edge:])
        )
        values = np.clip(profile - baseline, 0.0, None)
        peak_idx = int(np.argmax(values))
        peak = values[peak_idx]
        if not np.isfinite(peak) or peak <= 0:
            return 0.0

        level = 0.5 * peak
        left = np.flatnonzero(values[:peak_idx] <= level)
        right = np.flatnonzero(values[peak_idx + 1:] <= level)

        if left.size:
            i = left[-1]
            x0, x1 = centers[i], centers[i + 1]
            y0, y1 = values[i], values[i + 1]
            xleft = (x1 if y1 == y0 else
                     x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        else:
            xleft = centers[0]

        if right.size:
            i = peak_idx + 1 + right[0]
            x0, x1 = centers[i - 1], centers[i]
            y0, y1 = values[i - 1], values[i]
            xright = (x0 if y1 == y0 else
                      x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        else:
            xright = centers[-1]

        return max(0.0, xright - xleft)

    def _fwhm_from_projection(self, cx, cy, img=None):
        """Estimate marginal-projection FWHM in edge-coordinate units.

        The marginal projections (sums over the other axis) are used rather
        than a single central slice. A central slice severely underestimates
        the width of a tilted beam, which in turn makes the derived ROI too
        small. The arguments ``cx`` and ``cy`` are accepted for API
        compatibility.

        Args:
            cx: x-coordinate of the beam centroid (unused).
            cy: y-coordinate of the beam centroid (unused).
            img: optional nonnegative weight array to analyze instead of
                self.img (e.g. self.img_thresholded).

        Returns:
            np.array: [fwhm_x, fwhm_y] in bin-edge coordinate units.
        """
        del cx, cy
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        profile_x = img.sum(axis=1)
        profile_y = img.sum(axis=0)
        fwhm_x = self._profile_fwhm(profile_x, self.x_bin_centers)
        fwhm_y = self._profile_fwhm(profile_y, self.y_bin_centers)
        return np.array([fwhm_x, fwhm_y])

    @staticmethod
    def sigma_to_fwhm(sig):
        """Convert sigma to FWHM.

        Args:
            sig: sigma value(s).

        Returns:
            FWHM value(s).
        """
        return sig * FWHM2SIG

    @staticmethod
    def fwhm_to_sigma(fwhm):
        """Convert FWHM to sigma.

        Args:
            fwhm: FWHM value(s).

        Returns:
            sigma value(s).
        """
        return fwhm / FWHM2SIG

    @staticmethod
    def adjust_angle(theta):
        """Adjust angle to be within [-pi, pi].

        Args:
            theta: angle in radians.

        Returns:
            Adjusted angle in radians.
        """
        theta_wrapped = (theta + np.pi) % (2 * np.pi)
        if theta_wrapped > np.pi:
            theta_wrapped -= 2 * np.pi
        return theta_wrapped

    def parameters_from_projections(self):
        """Compute a quick beam estimate from marginal projections.

        Sets ``self.beam_visible``, ``self.hprm_project`` and the
        projection-derived ROI. The ROI is then refined with one or more
        moment-based passes so that tilted or slightly non-Gaussian beams
        are not truncated.

        The returned amplitude and offset are peak values per unit bin
        area (the histogram counts divided by ``self.bin_area``).

        Returns:
            dict or None: projection parameters if the beam is visible,
                otherwise None.
        """
        if self.img.ndim != 2:
            raise ValueError("img must be a 2D array.")
        nx, ny = self.img.shape
        if self.x_bin_edges.size != nx + 1:
            raise ValueError("x_bin_edges length must be img.shape[0] + 1.")
        if self.y_bin_edges.size != ny + 1:
            raise ValueError("y_bin_edges length must be img.shape[1] + 1.")

        cx, cy = self._centroid_from_projection()
        self.beam_visible = self._compute_beam_visibility(cx, cy)

        if not self.beam_visible:
            self.hprm_project = None
            self.compute_roi()
            import warnings
            warnings.warn("Beam not visible; skipping quick analysis.",
                          stacklevel=2)
            return None

        fwhms = self._fwhm_from_projection(cx, cy)
        sigx = self.fwhm_to_sigma(fwhms[0])
        sigy = self.fwhm_to_sigma(fwhms[1])
        if not np.isfinite(sigx) or sigx <= 0:
            sigx = self.x_bin_size
        if not np.isfinite(sigy) or sigy <= 0:
            sigy = self.y_bin_size

        self.hprm_project = {
            "mux"       : cx,
            "muy"       : cy,
            "sigx"      : sigx,
            "sigy"      : sigy,
            "fwhmx"     : fwhms[0],
            "fwhmy"     : fwhms[1],
            # no covariance in the projection estimate
            "cov"       : np.diag([sigx**2, sigy**2]),
            "sig_major" : max(sigx, sigy),
            "sig_minor" : min(sigx, sigy),
            # ellipse aligned to the axes
            "theta"     : 0.5 * np.pi * float(sigx < sigy),
            "amplitude" : 0.0,   # filled below, per unit bin area
            "offset"    : 0.0,   # filled below, per unit bin area
            # identity for the quick estimate
            "evecs"     : np.array([[1.0, 0.0], [0.0, 1.0]]),
            "x_bin_centers"  : self.x_bin_centers,
            "y_bin_centers"  : self.y_bin_centers,
        }

        # Build the projection ROI, then refine it using moments. The
        # refinement fixes the common case where the projection width is
        # slightly small (tilted beams, non-Gaussian tails).
        self.compute_roi()
        self._refine_roi_from_moments(self.img)

        offset_counts, amplitude_counts = self._gaussian_linear_estimate(
            self.img, self.hprm_project, useroi=True
        )
        self.hprm_project["amplitude"] = amplitude_counts / self.bin_area
        self.hprm_project["offset"] = offset_counts / self.bin_area
        return self.hprm_project

    def _roi_from_params(self, hprm):
        """Set the rectangular ROI mask and bounds from Gaussian parameters.

        Args:
            hprm: dict containing at least ``mux``, ``muy``, ``sigx`` and
                ``sigy`` in bin-edge coordinate units.

        Returns:
            np.ndarray: boolean ROI mask.
        """
        x0 = (hprm["mux"] - self.roi_sigma * hprm["sigx"]
              - self.roi_pad * self.x_bin_size)
        x1 = (hprm["mux"] + self.roi_sigma * hprm["sigx"]
              + self.roi_pad * self.x_bin_size)
        y0 = (hprm["muy"] - self.roi_sigma * hprm["sigy"]
              - self.roi_pad * self.y_bin_size)
        y1 = (hprm["muy"] + self.roi_sigma * hprm["sigy"]
              + self.roi_pad * self.y_bin_size)

        xlo, xhi = float(self.x_bin_edges[0]), float(self.x_bin_edges[-1])
        ylo, yhi = float(self.y_bin_edges[0]), float(self.y_bin_edges[-1])
        x0 = max(xlo, min(x0, xhi))
        x1 = max(xlo, min(x1, xhi))
        y0 = max(ylo, min(y0, yhi))
        y1 = max(ylo, min(y1, yhi))

        # Allow a one-bin ROI at an image edge, but never an empty span.
        if not x1 > x0:
            x0, x1 = xlo, xhi
        if not y1 > y0:
            y0, y1 = ylo, yhi

        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")
        roi_mask = (
            (xg >= x0) & (xg <= x1) &
            (yg >= y0) & (yg <= y1)
        )
        if not np.any(roi_mask):
            # Fall back to the full image rather than raising from deep in
            # the analysis pipeline.
            roi_mask = np.ones(self.img.shape, dtype=bool)
            x0, x1 = xlo, xhi
            y0, y1 = ylo, yhi

        self.roi_mask = roi_mask
        self.roi_bounds = (x0, x1, y0, y1)
        return roi_mask

    def _refine_roi_from_moments(self, img, max_iter=ROI_REFINE_ITERATIONS):
        """Refine the ROI using weighted moments until it stabilises.

        The projection estimate can be slightly narrow for tilted or
        non-Gaussian beams. A moment estimate made inside the current ROI is
        less sensitive to that, so the ROI is rebuilt from it. Iteration
        stops when the bounds change by less than a small fraction of a bin.

        Args:
            img: 2D image used for the moment calculation.
            max_iter: maximum number of refinement passes.

        Returns:
            np.ndarray or None: the final ROI mask, or None if it could not
                be refined.
        """
        if self.roi_mask is None:
            return None

        img = np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        scale = max(self.x_bin_size, self.y_bin_size)
        for _ in range(max(0, int(max_iter))):
            (mux, muy), covmat = self._covariance_from_moments(
                img, useroi=True
            )
            if covmat is None:
                break
            sigx = float(np.sqrt(covmat[0, 0]))
            sigy = float(np.sqrt(covmat[1, 1]))
            if (not np.isfinite(sigx) or not np.isfinite(sigy)
                    or sigx <= 0 or sigy <= 0):
                break

            old_bounds = self.roi_bounds
            self._roi_from_params({
                "mux": float(mux),
                "muy": float(muy),
                "sigx": sigx,
                "sigy": sigy,
            })
            if old_bounds is not None and self.roi_bounds is not None:
                change = max(
                    abs(a - b) for a, b in zip(old_bounds, self.roi_bounds)
                )
                if change <= 0.05 * scale:
                    break
        return self.roi_mask

    def compute_roi(self, hprm=None):
        """Rebuild the rectangular ROI from projection parameters.

        Uses the centroid and sigmas from ``hprm`` (or
        ``self.hprm_project``), with half-width
        ``roi_sigma * sigma + roi_pad`` on each axis. Stores the boolean mask
        in ``self.roi_mask`` and plotting bounds in ``self.roi_bounds``.

        Args:
            hprm: optional parameter dict used to define the ROI.

        Returns:
            np.ndarray or None: the boolean ROI mask, or None when no
                parameters are available.
        """
        self.roi_mask = None
        self.roi_bounds = None
        if hprm is None:
            hprm = self.hprm_project
        if hprm is None:
            return None
        return self._roi_from_params(hprm)

    def analyze(self, mode='all', warn=True):
        """Run analysis pipeline.

        Args:
            mode: 'quick' | 'moments' | 'fit' | 'all'. Default 'all'.
            warn: print warning if beam not visible. Default True.

        Returns:
            dict with results based on mode.
        """
        if mode == 'quick':
            return self.parameters_from_projections()

        if not self.beam_visible:
            if warn:
                import warnings
                warnings.warn(
                    f"Beam not visible (mode={mode}); "
                    "skipping advanced analysis.",
                    stacklevel=2
                )
            return None

        result = {}

        if mode in ('momenta', 'all'):
            result['moments'] = self.compute_momenta()

        if mode in ('fitting', 'all'):
            result['fit'] = self.fit()

        return result

    # Obsolete(ish) -- a single function calculating 3 things is 
    # awkward, we must settle on a single definition of intensity.
    # Refer to _amplitude_estimate to get peak count around the 
    # centroid.
    def qck_intensity(self, img, centroid, fwhms, exptime, droi=3):
        """Calculate beam intensity via various normalizations.

        Args:
            img: 2D image array.
            centroid: (cx, cy) coordinates in bin-edge units.
            fwhms: (fwhm_x, fwhm_y).
            exptime: exposure time.
            droi: nonnegative integer ROI radius in bins.

        Returns:
            np.array: [peak, intensity_by_mask, peak_fwhm_norm].
        """
        if not isinstance(droi, (int, np.integer)) or droi < 0:
            raise ValueError("droi must be a nonnegative integer number of bins.")
        cx, cy = centroid
        fx, fy = fwhms
        img = np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")
        roi = img[
            max(0, ix - droi):min(img.shape[0], ix + droi + 1),
            max(0, iy - droi):min(img.shape[1], iy + droi + 1),
        ]
        peak = np.mean(roi)
        peak /= exptime
        peak_fwhm_norm = peak / (fx * fy) if fx * fy != 0 else 0

        mask = img > (peak / 2)
        mask_area = np.sum(mask)
        masked_img = np.where(mask, img, 0)
        masked_img_area = np.sum(masked_img)
        intensity_by_mask = (
            masked_img_area / (mask_area * exptime) if mask_area != 0 else 0
        )

        return np.array([peak, intensity_by_mask, peak_fwhm_norm])

    @classmethod
    def from_gaussian(
        cls,
        bins=150,
        size=1_000_000,
        mean=DEFAULT_MEAN,
        cov=DEFAULT_COV,
        hist_range=DEFAULT_HIST_RANGE,
        seed=None,
    ):
        """Create an analyzer from a randomly sampled Gaussian distribution.

        Arguments:
            cls: Histogram2DAnalyzer class.
            bins: number of bins along each axis.
            size: number of samples to draw.
            mean: 2-element mean vector.
            cov: 2x2 covariance matrix.
            hist_range: [[xmin, xmax], [ymin, ymax]] histogram range.
            seed: optional random seed for reproducible histograms.

        Returns:
            Histogram2DAnalyzer instance.
        """
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(mean=mean, cov=cov, size=size)
        hist, x_bin_edges, y_bin_edges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=bins, range=hist_range
        )
        return cls(hist, x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges)

    # ------------------------------------------------------------------
    # Preprocessing.
    # ------------------------------------------------------------------

    def add_noise(self, noisetype="poisson", level=None, seed=None):
        """Add noise to self.img in-place.

        Arguments:
            noisetype: 'poisson' (default) or 'gaussian'.
            level: noise scale; defaults to self.img for Poisson.
            seed: optional random seed for reproducible noise.
        """
        rng = np.random.default_rng(seed)
        if noisetype not in ("poisson", "gaussian"):
            raise ValueError("noisetype must be 'poisson' or 'gaussian'.")
        if noisetype == "poisson":
            lam = self.img if level is None else level
            self.img = self.img + rng.poisson(lam, size=self.img.shape)
        else:
            self.img = self.img + rng.normal(
                loc=0, scale=np.sqrt(self.img), size=self.img.shape
            )

    # ------------------------------------------------------------------
    # Private math helpers.
    # ------------------------------------------------------------------

    def _bin_centers(self, x_bin_edges, y_bin_edges):
        """Set uniform bin edges and centers, defaulting to bin coordinates.

        Each missing axis gets edges from 0 to its bin count. Supplied edges
        must be finite, strictly increasing, and uniformly spaced.
        """
        def validate_edges(edges, count, axis):
            if edges is None:
                return np.arange(count + 1, dtype=float)
            try:
                edges = np.asarray(edges, dtype=float)
            except (TypeError, ValueError) as err:
                raise ValueError(f"{axis}_bin_edges must be numeric.") from err
            if edges.ndim != 1 or edges.size != count + 1:
                raise ValueError(
                    f"{axis}_bin_edges must be 1D with {count + 1} entries."
                )
            if not np.all(np.isfinite(edges)):
                raise ValueError(f"{axis}_bin_edges must contain finite values.")
            widths = np.diff(edges)
            if np.any(widths <= 0):
                raise ValueError(f"{axis}_bin_edges must be strictly increasing.")
            if not np.allclose(widths, widths[0], rtol=1e-7, atol=1e-12 * max(1.0, abs(widths[0]))):
                raise ValueError(f"{axis}_bin_edges must be uniformly spaced.")
            return edges

        self.x_bin_edges = validate_edges(x_bin_edges, self.img.shape[0], "x")
        self.y_bin_edges = validate_edges(y_bin_edges, self.img.shape[1], "y")
        self.x_bin_centers = 0.5 * (self.x_bin_edges[:-1] + self.x_bin_edges[1:])
        self.y_bin_centers = 0.5 * (self.y_bin_edges[:-1] + self.y_bin_edges[1:])
        # Physical size of one bin. Amplitude-like results (peak height and
        # background) are reported per unit bin area so that they do not
        # change when the histogram is rebinned.
        self.x_bin_size = float(self.x_bin_edges[1] - self.x_bin_edges[0])
        self.y_bin_size = float(self.y_bin_edges[1] - self.y_bin_edges[0])
        self.bin_area = self.x_bin_size * self.y_bin_size
        # Backwards-compatible aliases.
        self.x_bin_width = self.x_bin_size
        self.y_bin_width = self.y_bin_size

    def _nearest_bin_index(self, coordinate, axis):
        """Map a physical/bin coordinate to the nearest bin-center index."""
        if axis == "x":
            centers = self.x_bin_centers
        elif axis == "y":
            centers = self.y_bin_centers
        else:
            raise ValueError("axis must be 'x' or 'y'.")
        return int(np.argmin(np.abs(centers - coordinate)))

    def _covariance_from_moments(self, weight, useroi=False):
        """Compute the 2x2 covariance matrix from weighted bin-center moments.

        Args:
            weight: 2D weight array (same shape as the histogram).
            useroi: if True, use the projection-derived rectangular ROI
                stored in self.roi_mask.

        Returns:
            (mux, muy): means.
            covmat: 2x2 covariance matrix.
        """
        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")

        if useroi:
            if self.roi_mask is None:
                raise ValueError(
                    "ROI is not available; run compute_roi() first."
                )
            if self.roi_mask.shape != weight.shape:
                raise ValueError(
                    "ROI mask shape must match the weight array shape."
                )
            xg     = xg[self.roi_mask]
            yg     = yg[self.roi_mask]
            weight = weight[self.roi_mask]
        
        wsum = weight.sum()
        if wsum <= 0:
            print(f"WARNING: Total weight must be positive. wsum={wsum}")
            return (None, None), None
        mux = (weight * xg).sum() / wsum
        muy = (weight * yg).sum() / wsum
        dx = xg - mux
        dy = yg - muy
        varx  = (weight * dx * dx).sum() / wsum
        vary  = (weight * dy * dy).sum() / wsum
        covxy = (weight * dx * dy).sum() / wsum
        covmat = np.array([[varx, covxy], [covxy, vary]])
        return (mux, muy), covmat

    def _ellipse_params_from_cov(self, cov):
        """Return principal-axis parameters from a 2x2 covariance matrix.

        Arguments:
            cov: 2x2 covariance matrix.

        Returns:
            sig_major: std along the major axis.
            sig_minor: std along the minor axis.
            theta: rotation angle of the major axis (radians).
            evecs: eigenvectors matrix (columns = principal directions).
        """
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        sig_major = np.sqrt(evals[0])
        sig_minor = np.sqrt(evals[1])
        theta = np.arctan2(evecs[1, 0], evecs[0, 0])
        return sig_major, sig_minor, theta, evecs

    @staticmethod
    def _gaussian_2d(coords, offset, amplitude, mux, muy, sigx, sigy, rho):
        """Evaluate a 2D Gaussian with a constant offset.

        The model is evaluated in histogram units (counts per bin). The
        returned amplitude/offset are therefore also in counts per bin and
        must be divided by the bin area before being exposed to the user.

        The covariance is parameterised by the correlation coefficient
        ``rho`` instead of the raw covariance ``covxy``:

            covxy = rho * sigx * sigy

        Bounding ``rho`` to (-1, 1) guarantees a positive-definite
        covariance matrix, which the raw covariance parameter cannot.

        Arguments:
            coords: tuple (x, y) of coordinate arrays.
            offset: constant background level.
            amplitude: peak height above the offset.
            mux: mean in x.
            muy: mean in y.
            sigx: standard deviation in x.
            sigy: standard deviation in y.
            rho: correlation coefficient, -1 < rho < 1.

        Returns:
            Gaussian values at the given coordinates.
        """
        x, y = coords
        covxy = rho * sigx * sigy
        cov = np.array([[sigx**2, covxy], [covxy, sigy**2]])
        inv_cov = np.linalg.inv(cov)
        dx = x - mux
        dy = y - muy
        exponent = -0.5 * (
            inv_cov[0, 0] * dx**2
            + 2 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy**2
        )
        return offset + amplitude * np.exp(exponent)

    def _number_of_bins(self, data=None) -> int:
        """Return a bin count for threshold analysis.

        Uses Freedman-Diaconis as the primary estimate, capped at 1024,
        and falls back to Sturges' rule when FD is smaller. Returns at
        least 2.

        Args:
            data: optional 1D or 2D value array; defaults to self.img.
        """
        values = np.asarray(self.img if data is None else data)
        if values.size == 0:
            raise ValueError("data must contain at least one value.")
        n = values.size
        edgehist = np.histogram_bin_edges(values.ravel(), bins="fd")
        fd_nbins = min(len(edgehist) - 1, 1024)
        sturges_nbins = int(1.0 + np.log2(n))
        return max(fd_nbins, sturges_nbins, 2)

    @staticmethod
    def _kapur_entropy(freq: np.ndarray, k: int) -> float:
        """Kapur cross-class entropy for a histogram split at bin index k.

        Arguments:
            freq: 1D frequency array from np.histogram.
            k: index splitting background (< k) from foreground (>= k).

        Returns:
            Total cross-class entropy (background + foreground).
        """
        freq_b = freq[:k]
        freq_f = freq[k:]
        p_b = freq_b / freq_b.sum() if freq_b.sum() > 0 else np.array([])
        p_f = freq_f / freq_f.sum() if freq_f.sum() > 0 else np.array([])
        h_b = -np.sum(p_b * np.log2(p_b + 1e-10))
        h_f = -np.sum(p_f * np.log2(p_f + 1e-10))
        return h_b + h_f

    # ------------------------------------------------------------------
    # Public pipeline methods.
    # ------------------------------------------------------------------

    def _gaussian_linear_estimate(self, img=None, hprm=None, useroi=True):
        """Estimate offset and peak amplitude with a matched filter.

        The Gaussian centroid and covariance are held fixed from ``hprm``;
        only the constant offset and the amplitude are fitted. This makes
        the problem linear and uses every ROI pixel rather than a single
        noisy maximum or a box average. The result is a good starting point
        for the full nonlinear fit.

        Args:
            img: 2D array of counts per bin. Defaults to ``self.img``.
            hprm: parameter dict providing ``mux``, ``muy`` and either
                ``cov`` or ``sigx``/``sigy``. Defaults to
                ``self.hprm_project``.
            useroi: if True, use the current ROI mask when available.

        Returns:
            tuple: ``(offset_counts, amplitude_counts)`` in counts per bin.
        """
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")
        if hprm is None:
            hprm = self.hprm_project
        if hprm is None:
            # Without a shape estimate, a robust background and maximum are
            # better than failing the whole analysis.
            offset = float(np.median(img))
            amplitude = float(np.max(img) - offset)
            if img.min() >= 0:
                offset = max(0.0, offset)
                amplitude = max(0.0, amplitude)
            return offset, amplitude

        if "cov" in hprm and hprm["cov"] is not None:
            cov = np.asarray(hprm["cov"], dtype=float)
        else:
            cov = np.diag([hprm["sigx"] ** 2, hprm["sigy"] ** 2])
        if (cov.shape != (2, 2) or not np.all(np.isfinite(cov))
                or cov[0, 0] <= 0 or cov[1, 1] <= 0
                or np.linalg.det(cov) <= 0):
            cov = np.diag([
                max(hprm["sigx"] ** 2, self.x_bin_size ** 2),
                max(hprm["sigy"] ** 2, self.y_bin_size ** 2),
            ])

        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")
        if useroi and self.roi_mask is not None:
            if self.roi_mask.shape != img.shape:
                raise ValueError("ROI mask shape must match img shape.")
            mask = self.roi_mask
        else:
            mask = np.ones(img.shape, dtype=bool)

        inv_cov = np.linalg.inv(cov)
        dx = xg - hprm["mux"]
        dy = yg - hprm["muy"]
        shape = np.exp(-0.5 * (
            inv_cov[0, 0] * dx**2
            + 2 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy**2
        ))

        design = np.column_stack([np.ones(mask.sum()), shape[mask]])
        values = img[mask]
        if values.size < 2:
            offset = float(values[0]) if values.size else 0.0
            return offset, 0.0

        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        offset, amplitude = (float(coefficients[0]),
                             float(coefficients[1]))
        if img.min() >= 0:
            offset = max(0.0, offset)
            amplitude = max(0.0, amplitude)
        return offset, amplitude

    def _amplitude_estimate(self, img=None, droi=4):
        """Return a peak-amplitude estimate per unit bin area.

        This compatibility wrapper now uses the matched-filter estimate
        instead of the old local box mean. ``droi`` is retained for API
        compatibility but is no longer used.

        Args:
            img: optional 2D array of counts per bin.
            droi: ignored; kept for backwards compatibility.

        Returns:
            float: peak amplitude in counts per unit bin area.
        """
        del droi
        _, amplitude_counts = self._gaussian_linear_estimate(
            img, self.hprm_project, useroi=True
        )
        return amplitude_counts / self.bin_area

    def compute_momenta(self, img=None, useroi=True):
        """Compute weighted moments in the bin-edge coordinate units.

        The returned amplitude and offset are peak values per unit bin
        area, i.e. the counts-per-bin estimate divided by ``self.bin_area``.

        Arguments:
            img: optional array to analyse instead of self.img
                (e.g. self.img_thresholded).
            useroi: if True, use the projection-derived rectangular ROI.

        Returns:
            hprm: dictionary with centroid, sigma, covariance, amplitude and
                offset in the bin-edge coordinate system; also stored in
                ``self.hprm_momenta``.
        """
        if not self.beam_visible:
            import warnings
            warnings.warn("Beam not visible; skipping momenta analysis.",
                          stacklevel=2)
            return None

        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.ndim != 2:
            raise ValueError("img must be a 2D array.")
        nx, ny = img.shape
        if self.x_bin_edges.size != nx + 1:
            raise ValueError("x_bin_edges length must be img.shape[0] + 1.")
        if self.y_bin_edges.size != ny + 1:
            raise ValueError("y_bin_edges length must be img.shape[1] + 1.")

        (mux, muy), covmat = self._covariance_from_moments(
            img, useroi=useroi
        )
        if covmat is None or mux is None or muy is None:
            import warnings
            warnings.warn(
                "Moment calculation failed; image has no positive weight.",
                stacklevel=2,
            )
            return None

        sigx = float(np.sqrt(covmat[0, 0]))
        sigy = float(np.sqrt(covmat[1, 1]))
        sig_major, sig_minor, theta, evecs = self._ellipse_params_from_cov(
            covmat
        )

        shape_hprm = {
            "mux": float(mux),
            "muy": float(muy),
            "cov": covmat,
            "sigx": sigx,
            "sigy": sigy,
        }
        offset_counts, amplitude_counts = self._gaussian_linear_estimate(
            img, shape_hprm, useroi=useroi
        )

        self.hprm_momenta = {
            "mux"           : mux,
            "muy"           : muy,
            "cov"           : covmat,
            "sigx"          : sigx,
            "sigy"          : sigy,
            "fwhmx"         : self.sigma_to_fwhm(sigx),
            "fwhmy"         : self.sigma_to_fwhm(sigy),
            "sig_major"     : sig_major,
            "sig_minor"     : sig_minor,
            "theta"         : self.adjust_angle(theta),
            "evecs"         : evecs,
            "amplitude"     : amplitude_counts / self.bin_area,
            "offset"        : offset_counts / self.bin_area,
            "x_bin_centers" : self.x_bin_centers,
            "y_bin_centers" : self.y_bin_centers,
        }
        return self.hprm_momenta

    @staticmethod
    def _robust_background(img):
        """Estimate a constant background and its robust scatter.

        The median is used instead of the mean so that the beam itself does
        not bias the estimate. The scatter is the scaled median absolute
        deviation, which is insensitive to a small number of bright bins.

        Args:
            img: 2D array of counts per bin.

        Returns:
            tuple: ``(background, scatter)`` in counts per bin.
        """
        values = np.asarray(img, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 0.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, 1.4826 * mad

    @staticmethod
    def _propagate_uncertainties(func, params, pcov, rel_step=1e-6):
        """Propagate a covariance matrix through derived quantities.

        A numerical Jacobian of ``func`` is formed with central
        differences, and the output covariance is ``J pcov J^T``. This
        lets the caller report uncertainties for quantities that are not
        fit parameters directly (FWHM, covariance elements, principal
        sigmas, angle).

        Args:
            func: callable mapping the parameter vector to a 1D array.
            params: parameter vector at which to evaluate the Jacobian.
            pcov: parameter covariance matrix from ``curve_fit``.
            rel_step: relative finite-difference step.

        Returns:
            tuple: ``(values, errors)`` as 1D arrays. Errors are NaN when
                ``pcov`` is unavailable or not finite.
        """
        params = np.asarray(params, dtype=float)
        values = np.atleast_1d(np.asarray(func(params), dtype=float))
        jacobian = np.zeros((values.size, params.size), dtype=float)
        for i in range(params.size):
            step = rel_step * max(abs(params[i]), 1.0)
            plus = params.copy()
            minus = params.copy()
            plus[i] += step
            minus[i] -= step
            jacobian[:, i] = (
                np.atleast_1d(func(plus)) - np.atleast_1d(func(minus))
            ) / (2.0 * step)

        if pcov is None or not np.all(np.isfinite(pcov)):
            return values, np.full(values.shape, np.nan)
        derived_cov = jacobian @ pcov @ jacobian.T
        errors = np.sqrt(np.clip(np.diag(derived_cov), 0.0, None))
        return values, errors

    def fit(self, hprm=None, img=None, useroi=True):
        """Fit a 2D Gaussian using the bin-edge coordinate system.

        The model is fitted in counts per bin, but the returned amplitude
        and offset are peak values per unit bin area. The initial estimate
        from ``hprm`` is multiplied by ``self.bin_area`` before fitting and
        the fitted values are divided by it again.

        The fit is bounded and correlation-parameterised:

            offset in [0, background + OFFSET_UPPER_NSIGMA * sigma]
            amplitude in [0, AMPLITUDE_MAX_FACTOR * brightest bin]
            mux, muy within the image edges
            sigx, sigy >= SIGMA_MIN_BINS bins and <= the image span
            rho in [-RHO_MAX, RHO_MAX]

        Poisson weights ``sigma = sqrt(max(counts, 1))`` are used with
        ``absolute_sigma=True``, so the returned ``pcov`` gives absolute
        parameter uncertainties. If the weighted fit fails, an unweighted
        fit is attempted as a fallback and the result is flagged with
        ``weighted=False``.

        Arguments:
            hprm: optional pre-computed moments dict. Defaults to
                ``self.hprm_project``.
            img: optional counts-per-bin array to fit instead of self.img
                (e.g. self.img_thresholded); must have the same shape.
            useroi: if True, fit within the projection-derived rectangular
                ROI stored in self.roi_mask. The supplied hprm is still
                used for the initial parameter guess.

        Returns:
            hprm: fitted parameters dictionary; also stored in
                ``self.hprm_fitting``. It includes an ``errors`` dict with
                the uncertainty of every parameter and derived quantity,
                plus ``rho``, ``pcov``, ``chi2``, ``dof``,
                ``reduced_chi2`` and ``weighted`` when available.
        """
        if not self.beam_visible:
            return None

        if hprm is None:
            hprm = self.hprm_project

        # This is a workaround for Pylance, since the clauses above
        # guarantee hprm is calculated.
        if hprm is None:
            raise ValueError("No initial parameters available for fitting.")

        # Default to self.img if no alternative is provided.
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")

        # Fit to ROI.
        if useroi:
            if self.roi_mask is None:
                self.compute_roi(hprm)
            if self.roi_mask is None:
                raise ValueError(
                    "ROI is not available; run compute_roi() first."
                )
            roi_mask = self.roi_mask
        else:
            roi_mask = np.ones(img.shape, dtype=bool)

        x_opt = xg[roi_mask]
        y_opt = yg[roi_mask]
        img_opt = img[roi_mask]

        # Drop non-finite bins; a single NaN would otherwise poison the fit.
        finite = (np.isfinite(x_opt) & np.isfinite(y_opt)
                  & np.isfinite(img_opt))
        x_opt = x_opt[finite]
        y_opt = y_opt[finite]
        img_opt = img_opt[finite]
        if img_opt.size < 8:
            raise ValueError(
                "Not enough finite bins in the ROI to constrain the fit."
            )

        # --- Initial guess. hprm values are densities; convert back to
        # --- counts per bin for the fit model.
        p0_offset = float(hprm.get("offset", 0.0)) * self.bin_area
        p0_amplitude = float(hprm.get("amplitude", 0.0)) * self.bin_area
        if not np.isfinite(p0_offset):
            p0_offset = float(np.median(img_opt))
        if not np.isfinite(p0_amplitude) or p0_amplitude <= 0:
            p0_amplitude = max(
                0.0, float(np.max(img_opt)) - p0_offset
            )

        cov0 = np.asarray(hprm.get("cov", np.diag(
            [hprm["sigx"] ** 2, hprm["sigy"] ** 2]
        )), dtype=float)
        sigx0 = float(hprm["sigx"])
        sigy0 = float(hprm["sigy"])
        rho0 = 0.0
        if cov0.shape == (2, 2) and sigx0 > 0 and sigy0 > 0:
            rho0 = float(cov0[0, 1] / (sigx0 * sigy0))
        if not np.isfinite(rho0):
            rho0 = 0.0
        rho0 = float(np.clip(rho0, -RHO_MAX, RHO_MAX))

        p0 = np.array([
            p0_offset,
            p0_amplitude,
            float(hprm["mux"]),
            float(hprm["muy"]),
            sigx0,
            sigy0,
            rho0,
        ], dtype=float)

        # --- Bounds. The offset upper limit is tied to the robust
        # --- background so the baseline cannot absorb the beam. The
        # --- amplitude limit protects against a runaway peak.
        background, background_scatter = self._robust_background(img)
        offset_lower = min(0.0, float(np.min(img_opt)))
        offset_upper = max(
            p0_offset,
            background + OFFSET_UPPER_NSIGMA * max(
                background_scatter, np.sqrt(max(background, 1.0))
            ),
        ) + 1.0
        if offset_upper <= offset_lower:
            offset_upper = offset_lower + 1.0

        amplitude_lower = min(0.0, float(np.max(img_opt)))
        amplitude_upper = AMPLITUDE_MAX_FACTOR * max(
            float(np.max(img_opt)), p0_amplitude, 1.0
        )
        if amplitude_upper <= amplitude_lower:
            amplitude_upper = amplitude_lower + 1.0

        x_span = float(self.x_bin_edges[-1] - self.x_bin_edges[0])
        y_span = float(self.y_bin_edges[-1] - self.y_bin_edges[0])
        sigx_lower = SIGMA_MIN_BINS * self.x_bin_size
        sigy_lower = SIGMA_MIN_BINS * self.y_bin_size
        sigx_upper = max(x_span, 4.0 * abs(sigx0), sigx_lower * 2.0)
        sigy_upper = max(y_span, 4.0 * abs(sigy0), sigy_lower * 2.0)

        lower = np.array([
            offset_lower,
            amplitude_lower,
            float(self.x_bin_edges[0]),
            float(self.y_bin_edges[0]),
            sigx_lower,
            sigy_lower,
            -RHO_MAX,
        ])
        upper = np.array([
            offset_upper,
            amplitude_upper,
            float(self.x_bin_edges[-1]),
            float(self.y_bin_edges[-1]),
            sigx_upper,
            sigy_upper,
            RHO_MAX,
        ])

        # Keep p0 strictly inside the bounds.
        eps = 1e-9 * np.maximum(1.0, np.abs(upper - lower))
        p0 = np.clip(p0, lower + eps, upper - eps)

        # --- Poisson-weighted fit. Bins with zero counts get sigma=1 to
        # --- keep the weight finite.
        weights = np.sqrt(np.maximum(img_opt, 1.0))
        weighted = True
        try:
            popt, pcov = curve_fit(
                self._gaussian_2d,
                (x_opt, y_opt),
                img_opt,
                p0=p0,
                sigma=weights,
                absolute_sigma=True,
                bounds=(lower, upper),
                maxfev=FIT_MAXFEV,
                method="trf",
            )
        except (RuntimeError, ValueError):
            # Fall back to an unweighted fit. This loses the absolute
            # Poisson interpretation of pcov but still yields a result for
            # unusual or nearly empty ROIs.
            weighted = False
            popt, pcov = curve_fit(
                self._gaussian_2d,
                (x_opt, y_opt),
                img_opt,
                p0=p0,
                bounds=(lower, upper),
                maxfev=FIT_MAXFEV,
                method="trf",
            )

        offset, amplitude, mux, muy, sigx, sigy, rho = popt
        covxy = rho * sigx * sigy
        covmat = np.array([[sigx**2, covxy], [covxy, sigy**2]])
        (sig_major, sig_minor,
         theta, evecs) = self._ellipse_params_from_cov(covmat)

        # --- Parameter uncertainties from pcov.
        if pcov is None or not np.all(np.isfinite(pcov)):
            perr = np.full(popt.shape, np.nan)
        else:
            perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

        # Derived quantities are propagated numerically through pcov. The
        # angle and principal sigmas are written with a continuous formula
        # for a 2x2 covariance so that finite differences do not suffer
        # from eigenvector sign flips.
        def derived(values):
            (_, amp, _, _, sx, sy, r) = values
            cxy = r * sx * sy
            trace = sx**2 + sy**2
            disc = np.sqrt(
                max((sx**2 - sy**2) ** 2 + 4.0 * cxy**2, 0.0)
            )
            major = np.sqrt(max(0.5 * (trace + disc), 0.0))
            minor = np.sqrt(max(0.5 * (trace - disc), 0.0))
            angle = 0.5 * np.arctan2(2.0 * cxy, sx**2 - sy**2)
            return np.array([
                amp / self.bin_area,       # 0 amplitude
                values[0] / self.bin_area,  # 1 offset
                self.sigma_to_fwhm(sx),    # 2 fwhmx
                self.sigma_to_fwhm(sy),    # 3 fwhmy
                sx**2,                     # 4 cov_xx
                cxy,                       # 5 cov_xy
                sy**2,                     # 6 cov_yy
                major,                     # 7 sig_major
                minor,                     # 8 sig_minor
                angle,                     # 9 theta
            ])

        _, derived_err = self._propagate_uncertainties(derived, popt, pcov)

        errors = {
            "offset"    : float(perr[0]),
            "amplitude" : float(perr[1]),
            "mux"       : float(perr[2]),
            "muy"       : float(perr[3]),
            "sigx"      : float(perr[4]),
            "sigy"      : float(perr[5]),
            "rho"       : float(perr[6]),
            "fwhmx"     : float(derived_err[2]),
            "fwhmy"     : float(derived_err[3]),
            "cov_xx"    : float(derived_err[4]),
            "cov_xy"    : float(derived_err[5]),
            "cov_yy"    : float(derived_err[6]),
            "sig_major" : float(derived_err[7]),
            "sig_minor" : float(derived_err[8]),
            "theta"     : float(derived_err[9]),
        }

        # --- Goodness of fit. With Poisson weights and absolute_sigma=True,
        # --- the reduced chi-square is meaningful for counting data.
        chi2 = np.nan
        dof = int(img_opt.size - popt.size)
        reduced_chi2 = np.nan
        if weighted:
            model = self._gaussian_2d((x_opt, y_opt), *popt)
            chi2 = float(np.sum(
                ((img_opt - model) / np.sqrt(np.maximum(img_opt, 1.0))) ** 2
            ))
            if dof > 0:
                reduced_chi2 = chi2 / dof

        self.hprm_fitting = {
            "mux"           : mux,
            "muy"           : muy,
            "sigx"          : sigx,
            "sigy"          : sigy,
            "fwhmx"         : self.sigma_to_fwhm(sigx),
            "fwhmy"         : self.sigma_to_fwhm(sigy),
            "cov"           : covmat,
            "sig_major"     : sig_major,
            "sig_minor"     : sig_minor,
            "theta"         : self.adjust_angle(theta),
            "evecs"         : evecs,
            "rho"           : rho,
            "amplitude"     : amplitude / self.bin_area,
            "offset"        : offset / self.bin_area,
            "errors"        : errors,
            "pcov"          : pcov,
            "chi2"          : chi2,
            "dof"           : dof,
            "reduced_chi2"  : reduced_chi2,
            "weighted"      : weighted,
            "x_bin_centers" : self.x_bin_centers,
            "y_bin_centers" : self.y_bin_centers,
        }
        return self.hprm_fitting

    def compute_threshold(self, useroi=True):
        """Run Kapur entropy thresholding on self.img.

        Stores the optimal threshold in self.optimal_threshold and the
        thresholded image in self.img_thresholded. When useroi is True,
        the entropy histogram is built from ROI values only and the
        thresholded image is zero outside the ROI.

        Arguments:
            useroi: if True, restrict thresholding to self.roi_mask.
                Defaults to True. Falls back to the full image with a
                warning when no ROI is available.

        Returns:
            entropies: entropy value for each split index.
            bin_edges: bin edges of the internal 1D histogram.
            optimal_threshold: threshold that maximises cross-class entropy.
            nbins: number of bins used.
        """
        if useroi and self.roi_mask is None:
            import warnings
            warnings.warn(
                "ROI is not available; computing threshold on full image.",
                stacklevel=2,
            )
            useroi = False

        values = self.img[self.roi_mask] if useroi else self.img
        nbins = self._number_of_bins(values)
        freq, bin_edges = np.histogram(values, bins=nbins)
        thrs = bin_edges[:-1]

        entropies = np.array(
            [self._kapur_entropy(freq, k) for k in range(1, len(freq))]
        )
        thr = thrs[np.argmax(entropies)]

        self.optimal_threshold = thr
        if useroi:
            self.img_thresholded = np.where(
                self.roi_mask & (self.img > thr),
                self.img - self.optimal_threshold,
                0.0,
            )
        else:
            self.img_thresholded = np.where(self.img > thr,
                                            self.img - self.optimal_threshold,
                                            0.0)
        return entropies, bin_edges, thr, nbins

    def print_stats(self, hprm=None):
        """Print parameters from any hprm dict.

        Amplitude and offset are printed per unit bin area, matching the
        values stored in all parameter dicts. When the dict was produced by
        :meth:`fit`, each parameter is shown as ``value +/- error`` using
        the uncertainties propagated from the fit covariance matrix, and
        the reduced chi-square is reported.

        Arguments:
            hprm: optional dict to print; auto-selects from
                  hprm_fitting -> hprm_momenta -> hprm_project.
        """
        if hprm is None:
            hprm = (self.hprm_fitting or self.hprm_momenta
                    or self.hprm_project)
        if hprm is None:
            print("No parameters computed.")
            return

        errors = hprm.get("errors") or {}

        def fmt(key, spec=".4e"):
            """Format a value, appending its uncertainty when available."""
            if key in ("cov_xx", "cov_xy", "cov_yy") and "cov" in hprm:
                row, col = {"cov_xx": (0, 0), "cov_xy": (0, 1),
                            "cov_yy": (1, 1)}[key]
                value = float(np.asarray(hprm["cov"])[row, col])
            else:
                value = hprm.get(key)
            if value is None:
                return "n/a"
            text = format(value, spec)
            err = errors.get(key)
            if err is not None and np.isfinite(err):
                text += f" +/- {format(err, spec)}"
            return text

        print(f"bin size   = ({self.x_bin_size:.4e}, "
              f"{self.y_bin_size:.4e})")
        print(f"bin area   = {self.bin_area:.4e}")

        if "offset" in hprm:
            print(f"offset     = {fmt('offset')}  [counts / bin area]")
        if "amplitude" in hprm:
            print(f"amplitude  = {fmt('amplitude')}  [counts / bin area]")
        if "rho" in hprm:
            print(f"rho        = {fmt('rho')}")
        print()

        print(f"centroid   = ({fmt('mux')}, {fmt('muy')})")
        print(f"sigma x    = {fmt('sigx')}")
        print(f"sigma y    = {fmt('sigy')}")
        if "fwhmx" in hprm:
            print(f"fwhm x     = {fmt('fwhmx')}")
        if "fwhmy" in hprm:
            print(f"fwhm y     = {fmt('fwhmy')}")

        if "cov" not in hprm:
            return

        print(f"xy cov     = {fmt('cov_xy', '.4e')}")
        print()
        print(
            "principal sigmas:\n"
            f"    sigma minor = {fmt('sig_minor')}\n"
            f"    sigma major = {fmt('sig_major')}"
        )
        thetadeg = np.degrees(hprm["theta"])
        theta_err = errors.get("theta")
        theta_text = f"{thetadeg:.4e} deg"
        if theta_err is not None and np.isfinite(theta_err):
            theta_text += f" +/- {np.degrees(theta_err):.4e} deg"
        print(f"theta = {fmt('theta')} rad = {theta_text}")
        print(f"cov matrix:\n{hprm['cov']}")

        if "reduced_chi2" in hprm and np.isfinite(hprm["reduced_chi2"]):
            print(
                f"\nreduced chi2 = {hprm['reduced_chi2']:.4g} "
                f"(dof = {hprm['dof']})"
            )

    def _render_histogram(self, fig, ax, colorbar=True):
        """Render self.img over its supplied or default bin-edge coordinates.

        Returns:
            m: the pcolormesh artist.
        """
        m = ax.pcolormesh(
            self.x_bin_edges, self.y_bin_edges, self.img.T,
            shading="auto", cmap="viridis",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.x_bin_edges[0], self.x_bin_edges[-1])
        ax.set_ylim(self.y_bin_edges[0], self.y_bin_edges[-1])
        if colorbar:
            fig.colorbar(m, ax=ax, label="count")
        return m

    def plot(
        self,
        hprm=None,
        fig=None,
        ax=None,
        title=ELLIPSES_TITLE,
        ellipses=ELLIPSES,
        show_ellipse_axes=True,
        center_label="centroid",
        colorbar=True,
        show_roi=True,
        roi_zoom=1.2,
        roi_color="cyan",
        roi_ls="--",
    ):
        """Plot the histogram with sigma ellipses and principal directions.

        Arguments:
            hprm         : moments/fit dict; coordinates and widths use the
                           bin-edge coordinate units.
            fig          : optional figure to plot on; created if None.
            ax           : optional axis to plot on; created if None.
            title        : axes title.
            ellipses     : iterable of (n_sigma, color) pairs.
            show_ellipse_axes : draw principal-axis lines when True.
            center_label : legend label for the centroid marker.
            colorbar     : whether to show a colorbar for the histogram.
            show_roi     : draw the projection-derived ROI and zoom to it.
            roi_zoom     : positive zoom factor applied to the ROI bounds;
                           set to None to draw the ROI without zooming.
            roi_color    : ROI outline color.
            roi_ls       : ROI outline line style.

        Returns:
            fig, ax
        """
        if roi_zoom is not None:
            try:
                roi_zoom = float(roi_zoom)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    "roi_zoom must be a positive finite number or None."
                ) from err
            if not np.isfinite(roi_zoom) or roi_zoom <= 0:
                raise ValueError(
                    "roi_zoom must be a positive finite number or None."
                )

        if hprm is None:
            if self.hprm_momenta is None:
                self.compute_momenta()
            hprm = self.hprm_momenta

        # This is a workaround for Pylance, since the clauses above
        # guarantee hprm is calculated.
        if hprm is None:
            raise ValueError("No initial parameters available for fitting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        self._render_histogram(fig, ax, colorbar=colorbar)

        ax.plot(
            hprm["mux"], hprm["muy"],
            marker="+", color="red",
            label=center_label,
            # ms=10, mew=2,
        )

        for n, color in ellipses:
            ell = Ellipse(
                (hprm["mux"], hprm["muy"]),
                width=2 * n * hprm["sig_major"],
                height=2 * n * hprm["sig_minor"],
                angle=np.rad2deg(hprm["theta"]),
                fill=False, lw=2, color=color, label=f"{n}sigma",
            )
            ax.add_patch(ell)

        if show_ellipse_axes:
            v_major = hprm["evecs"][:, 0]
            v_minor = hprm["evecs"][:, 1]
            l_major = 3 * hprm["sig_major"]
            l_minor = 3 * hprm["sig_minor"]
            ax.plot(
                [hprm["mux"] - l_major * v_major[0],
                 hprm["mux"] + l_major * v_major[0]],
                [hprm["muy"] - l_major * v_major[1],
                 hprm["muy"] + l_major * v_major[1]],
                color="white", ls="--", lw=1,
            )
            ax.plot(
                [hprm["mux"] - l_minor * v_minor[0],
                 hprm["mux"] + l_minor * v_minor[0]],
                [hprm["muy"] - l_minor * v_minor[1],
                 hprm["muy"] + l_minor * v_minor[1]],
                color="white", ls=":", lw=1,
            )

        if show_roi and self.roi_bounds is not None:
            x0, x1, y0, y1 = self.roi_bounds
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, lw=1.5, color=roi_color, ls=roi_ls,
                label="ROI", zorder=5,
            ))

            if roi_zoom is not None:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                half_w = 0.5 * (x1 - x0) * roi_zoom
                half_h = 0.5 * (y1 - y0) * roi_zoom
                if half_w > 0:
                    xlim = (
                        max(float(self.x_bin_edges[0]), cx - half_w),
                        min(float(self.x_bin_edges[-1]), cx + half_w),
                    )
                    if xlim[1] > xlim[0]:
                        ax.set_xlim(*xlim)
                if half_h > 0:
                    ylim = (
                        max(float(self.y_bin_edges[0]), cy - half_h),
                        min(float(self.y_bin_edges[-1]), cy + half_h),
                    )
                    if ylim[1] > ylim[0]:
                        ax.set_ylim(*ylim)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        return fig, ax

    def plot_entropy(self, entropies, bin_edges, optimal_threshold):
        """Plot the entropy vs. threshold curve.

        Arguments:
            entropies: array returned by compute_threshold().
            bin_edges: bin edges returned by compute_threshold().
            optimal_threshold: threshold value to mark on the plot.

        Returns:
            fig
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bin_edges[1:-1], entropies, label="Entropy")
        ax.axvline(
            optimal_threshold, color="red", linestyle="--",
            label=f"Optimal Threshold: {optimal_threshold:.4g}",
        )
        ax.set_title("Entropy vs. Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Entropy")
        ax.legend()
        ax.grid()
        return fig
