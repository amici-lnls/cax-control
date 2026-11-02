# Statistical and Numerical Methods in `image_statistics.py`

This document is a mathematical and numerical guide to the analysis pipeline in
[`image_statistics.py`](../image_statistics.py). It explains what each stage
assumes, the equations it solves, the meaning of the returned quantities, and
the main limitations of the methods.

The central class is `Histogram2DAnalyzer`. It is designed for one 2D histogram
at a time, typically a beam image produced by a ray-tracing simulation or a
detector.

---

## Contents

1. [Data model and coordinate conventions](#1-data-model-and-coordinate-conventions)
2. [Binning and density normalization](#2-binning-and-density-normalization)
3. [Pipeline overview](#3-pipeline-overview)
4. [Beam detection and projection estimates](#4-beam-detection-and-projection-estimates)
5. [Region of interest](#5-region-of-interest)
6. [Moment analysis](#6-moment-analysis)
7. [Matched-filter amplitude and offset estimate](#7-matched-filter-amplitude-and-offset-estimate)
8. [The 2D Gaussian model](#8-the-2d-gaussian-model)
9. [Nonlinear fit](#9-nonlinear-fit)
10. [Parameter uncertainties](#10-parameter-uncertainties)
11. [Goodness of fit](#11-goodness-of-fit)
12. [The standalone thresholding path](#12-the-standalone-thresholding-path)
13. [Output dictionaries and attributes](#13-output-dictionaries-and-attributes)
14. [Display and diagnostics](#14-display-and-diagnostics)
15. [Numerical constants](#15-numerical-constants)
16. [Assumptions, limitations and best practices](#16-assumptions-limitations-and-best-practices)
17. [Testing and validation](#17-testing-and-validation)

---

## 1. Data model and coordinate conventions

### 1.1 The histogram

`Histogram2DAnalyzer.__init__` stores the input image as a floating-point array:

```python
self.img = np.asarray(img, dtype=float)
```

The array has shape

```text
(nx, ny)
```

and each entry is a **count per bin**, not a density. The first axis is `x`,
the second axis is `y`.

This convention is important because NumPy's `histogram2d` also returns
`hist[x_index, y_index]`. All internal mesh grids are built with

```python
xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                     indexing="ij")
```

so `xg[i, j] = x_bin_centers[i]` and `yg[i, j] = y_bin_centers[j]`.

The plotting helper `_render_histogram` transposes the image before drawing:

```python
ax.pcolormesh(self.x_bin_edges, self.y_bin_edges, self.img.T, ...)
```

because Matplotlib expects array rows to correspond to `y` and columns to `x`.

### 1.2 Bin edges and bin centres

`_bin_centers` validates and stores:

| Attribute | Meaning |
|---|---|
| `x_bin_edges`, `y_bin_edges` | Strictly increasing, uniformly spaced bin edges |
| `x_bin_centers`, `y_bin_centers` | Arithmetic bin centres |
| `x_bin_size`, `y_bin_size` | Physical bin widths |
| `bin_area` | `x_bin_size * y_bin_size` |
| `x_bin_width`, `y_bin_width` | Backwards-compatible aliases for the bin sizes |

If an axis' edges are not supplied, integer bin-index edges `0, 1, ..., n` are
used. Supplied edges must be finite, one-dimensional, strictly increasing, and
uniformly spaced. Uniform spacing is required because several methods assume a
single bin width per axis.

`_nearest_bin_index(coordinate, axis)` returns the index of the bin centre
closest to a given coordinate. It is used by the visibility test and by the
legacy local ROI in `qck_intensity`.

### 1.3 Coordinate units

All coordinates are expressed in the units of the supplied bin edges. For the
CAX notebooks those are micrometres, because the Shadow edges are converted
from millimetres to micrometres before constructing the analyzer. The module
itself is unit-agnostic.

Consequently:

- `mux`, `muy`, `sigx`, `sigy`, `fwhmx`, `fwhmy`, `sig_major`,
  `sig_minor`, and `roi_bounds` all have coordinate units;
- `cov` has coordinate units squared;
- `theta` is in radians;
- raw fit amplitude and offset are in **counts per bin**;
- exposed `amplitude` and `offset` are in **counts per bin area**.

---

## 2. Binning and density normalization

### 2.1 Why amplitude is divided by `bin_area`

A histogram bin with width $\Delta x$ and height $\Delta y$ collects counts
over an area

$$
a = \Delta x\,\Delta y = \texttt{bin\_area}.
$$

If the underlying density at the bin centre is $d(x,y)$ in counts per unit
area, the expected count in that bin is approximately

$$
c(x,y) \approx d(x,y)\,a.
$$

If the image is rebinned by merging $k \times k$ neighbouring bins, counts
add and the area grows by $k^2$:

$$
c' = k^2 c, \qquad a' = k^2 a,
\qquad \frac{c'}{a'} = \frac{c}{a}.
$$

Therefore a peak height expressed as

$$
\frac{c}{a}
$$

is approximately invariant under rebinning, while a raw `counts per bin` peak
is not. This is why every public parameter dictionary stores

```python
"amplitude": amplitude_counts / self.bin_area
"offset":    offset_counts    / self.bin_area
```

and why `print_stats` labels these values `[counts / bin area]`.

The nonlinear fit itself is performed in **counts per bin**, because the
observed data and the Poisson weights are counts. Initial estimates are
converted from density back to counts per bin by multiplying by `bin_area`:

$$
o_0 = \texttt{hprm["offset"]}\,a,
\qquad
A_0 = \texttt{hprm["amplitude"]}\,a.
$$

After fitting, the values are divided by `bin_area` again.

### 2.2 Relationship to the total number of counts

For the model used here, $A$ is the peak height above the offset in counts
per bin at the centre. The continuous density peak is $A/a$. The integral of
a 2D Gaussian is

$$
\iint \exp\!\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right]dx\,dy
= 2\pi\sqrt{\det\Sigma}
= 2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}.
$$

Hence the total number of counts in the fitted beam is approximately

$$
N_{\text{beam}}
\approx
2\pi\,\frac{A}{a}\,\sigma_x\sigma_y\sqrt{1-\rho^2}.
$$

This is useful when comparing the fitted peak with the total beam flux.

---

## 3. Pipeline overview

The class has three analysis stages and one optional segmentation stage:

```text
construction
    __init__
        _bin_centers
        parameters_from_projections
            _centroid_from_projection
            _compute_beam_visibility
            _fwhm_from_projection / _profile_fwhm
            compute_roi / _roi_from_params
            _refine_roi_from_moments
            _gaussian_linear_estimate
        -> hprm_project, roi_mask, roi_bounds

moment analysis
    compute_momenta
        _covariance_from_moments
        _ellipse_params_from_cov
        _gaussian_linear_estimate
        -> hprm_momenta

nonlinear fit
    fit
        _robust_background
        bounds and p0
        curve_fit with Poisson weights
        _propagate_uncertainties
        -> hprm_fitting

optional thresholding
    compute_threshold
        _number_of_bins
        _kapur_entropy
        -> optimal_threshold, img_thresholded
```

`analyze` orchestrates the stages. At the time of writing, the implementation
checks `mode in ('quick', 'momenta', 'fitting', 'all')`. The module-level
docstring uses `'moments'` and `'fit'`; those labels do not match the current
implementation. Use `'momenta'` and `'fitting'`.

The `qck_intensity` method is retained for compatibility but is obsolete. It is
not used by the main pipeline and its `droi` peak estimate has been superseded
by the matched-filter estimate in `_gaussian_linear_estimate`.

---

## 4. Beam detection and projection estimates

### 4.1 Peak-projection centroid: `_centroid_from_projection`

The centroid is not an intensity-weighted first moment. It is the location of
the maximum of a smoothed marginal projection.

For an image $I_{ij}$, define the marginal projections

$$
P_x(x_i) = \sum_j I_{ij}, \qquad
P_y(y_j) = \sum_i I_{ij}.
$$

Each projection is smoothed with a Savitzky–Golay filter
(`scipy.signal.savgol_filter`) of polynomial order 2 and window length

$$
w = \min(21,\; \text{largest odd integer} \le N),
$$

where $N$ is the projection length. If `w < 3`, no smoothing is applied. The
centroid is

$$
\mu_x = x_{\arg\max \tilde P_x}, \qquad
\mu_y = y_{\arg\max \tilde P_y}.
$$

Only the bin centre is returned, so the centroid resolution is one bin. The
Savitzky–Golay step reduces the chance that a single noisy bin becomes the
reported centre.

### 4.2 Visibility test: `_compute_beam_visibility`

The beam is considered visible when the local peak region is sufficiently
brighter than the whole image. Around the centroid bin $(i_x,i_y)$, the code
takes a square of radius `droi` bins and computes its mean:

$$
\bar c_{\text{local}}
=
\frac{1}{(2d+1)^2}
\sum_{i=i_x-d}^{i_x+d}
\sum_{j=i_y-d}^{i_y+d} I_{ij}.
$$

The global mean is

$$
\bar c_{\text{image}} = \frac{1}{N_xN_y}\sum_{i,j} I_{ij}.
$$

The visibility ratio is

$$
R = \frac{\bar c_{\text{local}}}{\bar c_{\text{image}}},
$$

and the beam is visible when

$$
R \ge \texttt{PEAK2AVG\_THRESHOLD} = 5.
$$

This is a heuristic contrast test, not a statistical detection test. A weak,
very broad beam that fills the image can fail even when it is physically
present.

When `beam_visible` is `False`, `parameters_from_projections` sets
`hprm_project = None`, `compute_roi()` produces no ROI, and a warning is
emitted. `compute_momenta` and `fit` return `None`.

### 4.3 Marginal-projection FWHM: `_fwhm_from_projection`

The width estimate uses the **marginal projections**, not a single central
slice:

$$
P_x(x_i) = \sum_j I_{ij}, \qquad
P_y(y_j) = \sum_i I_{ij}.
$$

A central slice is badly biased for tilted beams. For a rotated Gaussian, a
slice through the centre underestimates the projected width because it samples
only one cut through the ellipse. Summing over the other coordinate correctly
projects the full two-dimensional distribution onto each axis.

`_fwhm_from_projection` passes the two projections to `_profile_fwhm`.

### 4.4 One-dimensional FWHM: `_profile_fwhm`

For a profile $p_k$ sampled at centres $c_k$:

1. Estimate the baseline from the outer tenth of each side:

   $$
   m = \max\left(1,\left\lfloor \frac{N}{10}\right\rfloor\right),
   $$

   $$
   b = \frac12\left[
   \operatorname{median}(p_1,\ldots,p_m)
   +
   \operatorname{median}(p_{N-m+1},\ldots,p_N)
   \right].
   $$

2. Subtract the baseline and clip negative values to zero:

   $$
   q_k = \max(p_k-b, 0).
   $$

3. Find the peak index $k_p = \arg\max q_k$ and the half-maximum level

   $$
   \ell = \frac12 q_{k_p}.
   $$

4. Find the last index left of the peak with $q \le \ell$ and the first
   index right of the peak with $q \le \ell$, then linearly interpolate the
   crossing points between adjacent bin centres.

5. Return the distance between the two crossings. If a crossing is not found,
   the corresponding array edge is used.

The baseline subtraction is important when the projection sits on a nonzero
background. The interpolation gives a width estimate that is not quantized to
whole bins.

### 4.5 FWHM and sigma conversion

For a one-dimensional Gaussian,

$$
\mathrm{FWHM} = 2\sqrt{2\ln 2}\,\sigma.
$$

The module defines

```python
FWHM2SIG = 2 * np.sqrt(2 * np.log(2))  # approximately 2.35482
```

and provides:

- `sigma_to_fwhm(sig) -> sig * FWHM2SIG`
- `fwhm_to_sigma(fwhm) -> fwhm / FWHM2SIG`

The projection estimate uses

$$
\sigma_x = \frac{\mathrm{FWHM}_x}{\texttt{FWHM2SIG}},
\qquad
\sigma_y = \frac{\mathrm{FWHM}_y}{\texttt{FWHM2SIG}}.
$$

If this produces a non-finite or non-positive sigma, the corresponding bin
size is used as a fallback.

### 4.6 The provisional projection dictionary

`parameters_from_projections` stores `hprm_project` with:

- `mux`, `muy`: peak-projection centroid;
- `sigx`, `sigy`: FWHM-derived sigmas;
- `fwhmx`, `fwhmy`: projected FWHMs;
- `cov = diag(sigx**2, sigy**2)`: a diagonal covariance, because the
  projection method does not estimate correlation;
- `sig_major = max(sigx, sigy)`, `sig_minor = min(sigx, sigy)`;
- `theta = 0` if `sigx >= sigy`, otherwise $\pi/2$;
- `evecs` set to the identity matrix;
- `amplitude`, `offset`: initially zero, then filled by the matched filter;
- `x_bin_centers`, `y_bin_centers`.

A subtlety: because the projection covariance is diagonal by construction,
`evecs` is the identity. The ellipse patch correctly uses `theta` to rotate
when `sigy > sigx`, but the principal-axis lines drawn by `plot` use the
identity `evecs`, so they are only geometrically consistent when `sigx >=
sigy`. This does not affect the numerical analysis.

---

## 5. Region of interest

### 5.1 Rectangular ROI construction: `_roi_from_params`

Given a parameter dictionary with `mux`, `muy`, `sigx`, `sigy`, the half-widths
are

$$
w_x = \texttt{roi\_sigma}\,\sigma_x
      + \texttt{roi\_pad}\,\Delta x,
$$

$$
w_y = \texttt{roi\_sigma}\,\sigma_y
      + \texttt{roi\_pad}\,\Delta y.
$$

The nominal bounds are

$$
[x_0,x_1] = [\mu_x-w_x,\;\mu_x+w_x],
\qquad
[y_0,y_1] = [\mu_y-w_y,\;\mu_y+w_y].
$$

They are clipped to the image edges. The boolean mask is built on bin centres:

$$
\text{mask}_{ij} =
(x_i \ge x_0)\land(x_i \le x_1)\land
(y_j \ge y_0)\land(y_j \le y_1).
$$

If a span collapses to zero, the full axis span is used. If the mask ends up
empty, the full image is used as a fallback.

The default `roi_sigma = 5.0` and `roi_pad = 2.0` bins. A 5-sigma box already
contains almost all of a Gaussian, and the padding protects against slightly
underestimated projection widths and finite-bin effects. Both are configurable
through `__init__`.

`compute_roi(hprm=None)` is the public wrapper. If `hprm` is omitted it uses
`self.hprm_project`. It stores `roi_mask` and `roi_bounds = (x0, x1, y0, y1)`.

### 5.2 Moment-based refinement: `_refine_roi_from_moments`

After the first projection ROI is built, `parameters_from_projections` calls
`_refine_roi_from_moments`. The motivation is that the projection FWHM is still
a one-dimensional summary; for tilted or mildly non-Gaussian beams it can be
slightly narrow.

At each iteration:

1. Compute weighted moments **inside the current ROI** using the raw image
   counts as weights (see [Section 6](#6-moment-analysis)).
2. Extract $\sigma_x=\sqrt{\mathrm{Cov}_{xx}}$ and
   $\sigma_y=\sqrt{\mathrm{Cov}_{yy}}$.
3. Rebuild the rectangular ROI with `_roi_from_params`.
4. Stop when the largest change in any of the four bounds is at most

   $$
   0.05\,\max(\Delta x,\Delta y).
   $$

The loop runs at most `ROI_REFINE_ITERATIONS = 3` times.

Only the ROI mask and bounds are updated; the projection sigmas in
`hprm_project` are not overwritten. The refined ROI is then used for the
matched-filter estimate and later for the fit.

A limitation: the moment calculation uses raw counts without subtracting a
background. A large flat background inflates the second moments and therefore
the refined ROI. This is conservative (the ROI grows rather than truncating the
beam), but it can reduce the signal-to-background ratio in the fit region.

---

## 6. Moment analysis

### 6.1 Weighted moments: `_covariance_from_moments`

Given a weight array $w_{ij}$ and optional ROI mask, define

$$
S_0 = \sum_{ij} w_{ij}.
$$

If $S_0 \le 0$, the function reports a warning and returns `None`. Otherwise
the weighted centroid is

$$
\mu_x = \frac{1}{S_0}\sum_{ij} w_{ij}x_i,
\qquad
\mu_y = \frac{1}{S_0}\sum_{ij} w_{ij}y_j,
$$

and the weighted covariance is

$$
\mathrm{Cov}_{xx}
= \frac{1}{S_0}\sum_{ij} w_{ij}(x_i-\mu_x)^2,
$$

$$
\mathrm{Cov}_{yy}
= \frac{1}{S_0}\sum_{ij} w_{ij}(y_j-\mu_y)^2,
$$

$$
\mathrm{Cov}_{xy}
= \frac{1}{S_0}\sum_{ij} w_{ij}(x_i-\mu_x)(y_j-\mu_y).
$$

The covariance matrix is

$$
C =
\begin{pmatrix}
\mathrm{Cov}_{xx} & \mathrm{Cov}_{xy}\\
\mathrm{Cov}_{xy} & \mathrm{Cov}_{yy}
\end{pmatrix}.
$$

This is the second central moment of the **weighted intensity distribution**.
It is not corrected for the finite bin width and it does not subtract a
background. For a Gaussian beam with a small constant background, the widths
are therefore slightly overestimated.

The function can operate on the full image or inside `roi_mask`. It is used in
two places:

- `compute_momenta`, where it produces the public moment dictionary;
- `_refine_roi_from_moments`, where it updates the ROI bounds.

### 6.2 Principal axes: `_ellipse_params_from_cov`

The covariance matrix is symmetric positive semidefinite. Its eigendecomposition
is

$$
C = V\Lambda V^T,
\qquad
\Lambda = \operatorname{diag}(\lambda_1,\lambda_2),
\qquad
\lambda_1 \ge \lambda_2 \ge 0.
$$

The principal standard deviations are

$$
\sigma_{\text{major}} = \sqrt{\lambda_1},
\qquad
\sigma_{\text{minor}} = \sqrt{\lambda_2}.
$$

The major-axis direction is the first column of $V$. The angle returned by
`_ellipse_params_from_cov` is

$$
\theta = \operatorname{atan2}(v_{y,\text{major}}, v_{x,\text{major}}).
$$

`adjust_angle` returns a representative obtained by adding $\pi$ and wrapping
to $(-\pi,\pi]$. Because an ellipse has orientation modulo $\pi$, this
describes the same ellipse even though the numeric value can differ from the
eigenvector angle by $\pi$. The stored `theta` should therefore be interpreted
as an ellipse orientation, not as an oriented vector.

The output dictionary also stores `evecs`, whose columns are the principal
directions.

### 6.3 `compute_momenta`

`compute_momenta(img=None, useroi=True)`:

1. Returns `None` with a warning if `beam_visible` is false.
2. Validates the image shape and bin-edge lengths.
3. Calls `_covariance_from_moments`.
4. Computes `sigx`, `sigy`, `fwhmx`, `fwhmy`, `sig_major`, `sig_minor`,
   `theta`, and `evecs`.
5. Builds a temporary shape dictionary from the moments and calls
   `_gaussian_linear_estimate` to obtain a peak amplitude and offset.
6. Stores everything in `hprm_momenta`.

The moment dictionary's `amplitude` and `offset` are already density-normalized
by `bin_area`.

An important distinction: `compute_momenta` uses the **current** `roi_mask`.
It does not rebuild the ROI from the moments unless `_refine_roi_from_moments`
was called explicitly. The constructor has already refined the ROI once.

---

## 7. Matched-filter amplitude and offset estimate

### 7.1 Purpose

`_gaussian_linear_estimate` provides the initial amplitude and offset for the
nonlinear fit, and the amplitude/offset reported in `hprm_project` and
`hprm_momenta`.

A naive estimate such as `max(img)` is strongly biased upward by noise, and a
local box mean underestimates the peak of a narrow beam. The matched filter uses
all ROI pixels while fixing the Gaussian shape.

### 7.2 Linear model

Suppose the centroid and covariance are already known from `hprm`:

$$
g_{ij}
=
\exp\!\left[
-\frac12 (r_{ij}-\mu)^T\Sigma^{-1}(r_{ij}-\mu)
\right],
$$

where $r_{ij}=(x_i,y_j)$ and $\Sigma$ is obtained from `hprm["cov"]` or
from `diag(sigx**2, sigy**2)`. The model is linear in the two remaining
parameters:

$$
f_{ij} = o + A\,g_{ij}.
$$

For all ROI pixels, form the design matrix

$$
X =
\begin{pmatrix}
1 & g_1\\
1 & g_2\\
\vdots & \vdots\\
1 & g_N
\end{pmatrix},
\qquad
y =
\begin{pmatrix}
c_1\\ c_2\\ \vdots\\ c_N
\end{pmatrix}.
$$

The least-squares problem is

$$
\min_{o,A}\|X\beta-y\|_2^2,
\qquad
\beta = \begin{pmatrix}o\\ A\end{pmatrix}.
$$

The normal equations are

$$
X^TX\beta = X^Ty,
$$

solved with `np.linalg.lstsq` for numerical stability. The result is the
constant offset and the peak amplitude in counts per bin.

If the image is nonnegative, the returned offset and amplitude are clipped to
be nonnegative. If the covariance is invalid (not finite, non-positive
diagonal, or non-positive determinant), the code falls back to a diagonal
covariance with sigmas at least one bin wide. If no parameter dictionary is
available at all, it falls back to `median(img)` for the offset and
`max(img) - offset` for the amplitude.

This estimate is unweighted. It is only a starting point and a quick diagnostic;
the final statistical uncertainties come from the nonlinear fit.

### 7.3 Compatibility wrapper

`_amplitude_estimate(img=None, droi=4)` now delegates to
`_gaussian_linear_estimate` and returns the amplitude divided by `bin_area`.
The `droi` argument is retained for backwards compatibility but is ignored.

---

## 8. The 2D Gaussian model

### 8.1 Model equation

The fit model is implemented in `_gaussian_2d`:

$$
f(x,y;\;o,A,\mu_x,\mu_y,\sigma_x,\sigma_y,\rho)
=
o + A\exp\left(-\frac12 Q\right),
$$

with

$$
Q =
(\mathbf{r}-\boldsymbol{\mu})^T
\Sigma^{-1}
(\mathbf{r}-\boldsymbol{\mu}),
\qquad
\mathbf{r} = \begin{pmatrix}x\\ y\end{pmatrix},
\qquad
\boldsymbol{\mu} = \begin{pmatrix}\mu_x\\ \mu_y\end{pmatrix}.
$$

The covariance matrix is parameterized as

$$
\Sigma =
\begin{pmatrix}
\sigma_x^2 & \rho\sigma_x\sigma_y\\
\rho\sigma_x\sigma_y & \sigma_y^2
\end{pmatrix}.
$$

Equivalently,

$$
Q =
\frac{1}{1-\rho^2}
\left[
\frac{(x-\mu_x)^2}{\sigma_x^2}
-
\frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}
+
\frac{(y-\mu_y)^2}{\sigma_y^2}
\right].
$$

The model is evaluated in **counts per bin**. Therefore:

- $o$ is the constant background in counts per bin;
- $A$ is the peak height above the background in counts per bin;
- the peak value at the centre is $o+A$;
- the public `offset` and `amplitude` are $o/a$ and $A/a$.

### 8.2 Why `rho` instead of `covxy`

The older code fitted the raw covariance element $\mathrm{Cov}_{xy}$. That
parameterization has a problem: a finite $\mathrm{Cov}_{xy}$ combined with
arbitrary $\sigma_x,\sigma_y$ can produce a non-positive-definite covariance
matrix. The parameterization

$$
\mathrm{Cov}_{xy} = \rho\,\sigma_x\sigma_y
$$

makes positive definiteness equivalent to

$$
\sigma_x > 0,\qquad \sigma_y > 0,\qquad |\rho| < 1.
$$

The code enforces

$$
-\texttt{RHO\_MAX} \le \rho \le \texttt{RHO\_MAX},
\qquad
\texttt{RHO\_MAX} = 1 - 10^{-6}.
$$

This keeps $\Sigma$ strictly positive definite.

### 8.3 Derived quantities

From the fitted parameters:

$$
\mathrm{Cov}_{xx} = \sigma_x^2,
\qquad
\mathrm{Cov}_{xy} = \rho\sigma_x\sigma_y,
\qquad
\mathrm{Cov}_{yy} = \sigma_y^2,
$$

$$
\sigma_{\text{major/minor}}
=
\sqrt{
\frac{\sigma_x^2+\sigma_y^2}{2}
\pm
\frac12\sqrt{(\sigma_x^2-\sigma_y^2)^2
+4\rho^2\sigma_x^2\sigma_y^2}
},
$$

and

$$
\theta =
\frac12\operatorname{atan2}
\left(
2\rho\sigma_x\sigma_y,\;
\sigma_x^2-\sigma_y^2
\right).
$$

The continuous formulas above are used for uncertainty propagation because
they avoid eigenvector sign flips. The reported `evecs` and principal sigmas
come from `_ellipse_params_from_cov`, which uses `numpy.linalg.eigh`.

---

## 9. Nonlinear fit

### 9.1 Parameter vector and initial guess

`fit(hprm=None, img=None, useroi=True)` fits the seven parameters

$$
p = [o,\;A,\;\mu_x,\;\mu_y,\;\sigma_x,\;\sigma_y,\;\rho].
$$

If `hprm` is omitted it uses `self.hprm_project`. The initial guess is:

$$
o_0 = \texttt{hprm["offset"]}\cdot a,
\qquad
A_0 = \texttt{hprm["amplitude"]}\cdot a,
$$

$$
\mu_{x,0} = \texttt{hprm["mux"]},
\qquad
\mu_{y,0} = \texttt{hprm["muy"]},
$$

$$
\sigma_{x,0} = \texttt{hprm["sigx"]},
\qquad
\sigma_{y,0} = \texttt{hprm["sigy"]},
$$

$$
\rho_0 =
\frac{\texttt{hprm["cov"]}_{xy}}
{\sigma_{x,0}\sigma_{y,0}}.
$$

If the amplitude estimate is non-positive, it is replaced by

$$
A_0 = \max(0,\; \max_{i\in\text{ROI}} y_i - o_0).
$$

The initial vector is clipped to lie strictly inside the bounds.

### 9.2 Bounds

All seven parameters are bounded.

Let $y_{\min}$ and $y_{\max}$ be the minimum and maximum counts in the ROI.
Let $b$ and $s_{\text{MAD}}$ be the robust background and scatter returned
by `_robust_background`.

| Parameter | Lower bound | Upper bound |
|---|---|---|
| `offset` $o$ | $\min(0,y_{\min})$ | $\max\left(o_0,\; b + 5\max(s_{\text{MAD}},\sqrt{\max(b,1)})\right)+1$ |
| `amplitude` $A$ | $\min(0,y_{\max})$ | $2\max(y_{\max},A_0,1)$ |
| `mux` | `x_bin_edges[0]` | `x_bin_edges[-1]` |
| `muy` | `y_bin_edges[0]` | `y_bin_edges[-1]` |
| `sigx` | $0.5\,\Delta x$ | $\max(x_{\text{span}},4\sigma_{x,0},2\cdot 0.5\Delta x)$ |
| `sigy` | $0.5\,\Delta y$ | $\max(y_{\text{span}},4\sigma_{y,0},2\cdot 0.5\Delta y)$ |
| `rho` | $-\texttt{RHO\_MAX}$ | $+\texttt{RHO\_MAX}$ |

The purpose of each bound:

- **Offset:** must not be negative for nonnegative count images, and must not
  rise so high that it absorbs the beam. The upper bound is tied to the robust
  background, the MAD scatter, and the Poisson noise scale. See
  [Section 9.3](#93-robust-background).
- **Amplitude:** cannot be negative for nonnegative images and cannot exceed
  twice the brightest ROI bin.
- **Centroid:** must remain inside the image.
- **Sigma:** cannot collapse below half a bin, which would represent a spike
  narrower than the histogram can resolve; cannot exceed the image span or four
  times the initial estimate.
- **Rho:** keeps the covariance positive definite.

If a supplied upper bound is not strictly above its lower bound, the code
expands it by one unit. The `p0` vector is then clipped with a small relative
margin.

### 9.3 Robust background: `_robust_background`

The fit needs a data-driven upper bound for the constant offset. The function
computes:

$$
b = \operatorname{median}_{i}(y_i),
$$

$$
s_{\text{MAD}}
=
1.4826 \cdot
\operatorname{median}_{i}
\left(
|y_i-b|
\right).
$$

The scaling constant is the Gaussian consistency factor:

$$
\frac{1}{\Phi^{-1}(3/4)} \approx 1.4826.
$$

For Gaussian noise with standard deviation $\sigma$,

$$
\mathrm{MAD} \approx 0.6745\,\sigma,
\qquad
1.4826\,\mathrm{MAD} \approx \sigma.
$$

Why median and MAD:

- The mean and standard deviation are pulled upward by the beam and by hot
  pixels.
- The median and MAD are robust: they describe the typical background pixel and
  its typical fluctuation while ignoring a minority of bright beam pixels.

For a Poisson background with mean $b$, the expected standard deviation is
$\sqrt{b}$. The bound uses

$$
s_b = \max\left(s_{\text{MAD}},\sqrt{\max(b,1)}\right)
$$

so that a zero-MAD, zero-background image still gets a finite Poisson-scale
allowance. The offset upper bound is then

$$
b_{\max}
=
\max\left(o_0,\; b+5s_b\right)+1.
$$

The `+1` is an absolute cushion in counts per bin. `OFFSET_UPPER_NSIGMA = 5`
sets the number of robust sigmas allowed.

Limitations: this is a global background estimate. A strong background gradient
or a beam covering more than half the image can bias the median. The function is
used for a bound, not for subtracting a background model.

### 9.4 Objective function and Poisson weights

`fit` minimizes the weighted least-squares objective

$$
\chi^2(p)
=
\sum_{i\in\text{ROI}}
\left(
\frac{y_i - f(x_i,y_i;p)}{s_i}
\right)^2,
$$

with

$$
s_i = \sqrt{\max(y_i,1)}.
$$

The corresponding weights are

$$
w_i = \frac{1}{s_i^2}
= \frac{1}{\max(y_i,1)}.
$$

This is **Neyman chi-square** weighting. It is motivated by Poisson counting
statistics:

$$
y_i \sim \operatorname{Poisson}(\mu_i),
\qquad
\mathbb{E}[y_i]=\mu_i,
\qquad
\operatorname{Var}(y_i)=\mu_i.
$$

Since $\mu_i$ is unknown, the observed count is used as its estimate:

$$
\operatorname{Var}(y_i) \approx y_i,
\qquad
s_i \approx \sqrt{y_i}.
$$

The floor $\max(y_i,1)$ prevents zero or one-count bins from receiving
infinite weight. A zero-count bin is treated as a one-count fluctuation.

This is not the exact Poisson likelihood. The exact negative log-likelihood is

$$
-2\ln L
=
2\sum_i\left(f_i-y_i\ln f_i\right)
+\text{constant},
$$

which would require a dedicated optimizer rather than `curve_fit`. In the
high-count limit, inverse-variance weighting is a good approximation, and for
the CAX images used here the bin counts are large enough that the difference is
small.

Passing `sigma=s_i` and `absolute_sigma=True` to `curve_fit` means:

- `sigma` is a per-data-point standard deviation in counts per bin;
- `pcov` is not rescaled by the residual variance;
- the resulting parameter errors are intended to be absolute Poisson errors.

The optimizer is `scipy.optimize.curve_fit` with `method="trf"` (trust-region
reflective), `maxfev=FIT_MAXFEV = 20000`, and the bounds above.

If the weighted fit raises `RuntimeError` or `ValueError`, the code retries an
unweighted fit with the same bounds. In that case:

- `weighted = False`;
- `chi2`, `reduced_chi2` are `nan`;
- `pcov` no longer has an absolute Poisson interpretation.

### 9.5 Finite-data filtering

Before fitting, non-finite values are removed from the ROI:

$$
\text{keep}_i =
\operatorname{isfinite}(x_i)\land
\operatorname{isfinite}(y_i)\land
\operatorname{isfinite}(y_i).
$$

At least eight finite bins are required to constrain seven parameters. A single
`NaN` would otherwise contaminate the entire fit.

---

## 10. Parameter uncertainties

### 10.1 What `pcov` is

`curve_fit` returns `pcov`, the estimated covariance matrix of the fitted
parameters. If

$$
J_{ij} = \frac{\partial f(x_i;p)}{\partial p_j},
\qquad
W = \operatorname{diag}(1/s_i^2),
$$

then near the optimum

$$
\operatorname{Cov}(\hat p)
\approx
\left(J^T W J\right)^{-1}.
$$

The diagonal entries are parameter variances:

$$
\operatorname{Var}(\hat p_j) = \texttt{pcov}_{jj},
\qquad
\sigma_{\hat p_j} = \sqrt{\texttt{pcov}_{jj}}.
$$

The off-diagonal entries describe correlations. For example, amplitude and
offset are often correlated because increasing the background and the peak can
partially compensate.

`fit` stores `pcov` in `hprm_fitting["pcov"]` and the square roots of its
diagonal in the `errors` dictionary for the seven direct parameters:

```text
offset, amplitude, mux, muy, sigx, sigy, rho
```

These are 1σ marginal errors. They do not include systematic errors.

### 10.2 The role of `absolute_sigma`

With `absolute_sigma=True`, SciPy does **not** multiply `pcov` by the observed
reduced chi-square. The errors therefore represent the absolute statistical
uncertainty implied by the supplied Poisson sigmas.

With `absolute_sigma=False`, the covariance would be scaled by approximately

$$
\frac{\chi^2}{N-P},
$$

which inflates the errors when the model fits poorly but destroys the strict
Poisson interpretation. The reduced chi-square is reported separately here as a
diagnostic.

### 10.3 Derived quantities and the delta method

Suppose a quantity is a function of the fitted parameters:

$$
q = g(p).
$$

A first-order Taylor expansion gives

$$
q(\hat p+\delta p) \approx g(\hat p) + J_g\,\delta p,
$$

where

$$
(J_g)_{ij} = \frac{\partial g_i}{\partial p_j}.
$$

Therefore,

$$
\operatorname{Cov}(q)
\approx
J_g\,\operatorname{Cov}(\hat p)\,J_g^T.
$$

The uncertainty on each derived quantity is

$$
\sigma_{q_i}
=
\sqrt{
\left(J_g\,\texttt{pcov}\,J_g^T\right)_{ii}
}.
$$

This is the **delta method**.

`_propagate_uncertainties(func, params, pcov, rel_step=1e-6)` computes the
Jacobian numerically with central differences:

$$
(J_g)_{ij}
\approx
\frac{
g_i(p+h_j e_j)-g_i(p-h_j e_j)
}{
2h_j
},
$$

with

$$
h_j = 10^{-6}\max(|p_j|,1).
$$

The `max(...,1)` ensures an absolute step of at least $10^{-6}$ when a
parameter is near zero. The derived covariance is then

```python
derived_cov = jacobian @ pcov @ jacobian.T
errors = np.sqrt(np.clip(np.diag(derived_cov), 0.0, None))
```

If `pcov` is `None` or contains non-finite entries, all derived errors are set
to `NaN`.

### 10.4 Which derived quantities are propagated

The `derived` closure inside `fit` returns:

1. amplitude divided by `bin_area`:

   $$
   q_0 = \frac{A}{a},
   \qquad
   \sigma_{q_0} = \frac{\sigma_A}{a};
   $$

2. offset divided by `bin_area`:

   $$
   q_1 = \frac{o}{a};
   $$

3. FWHM values:

   $$
   q_2 = \texttt{FWHM2SIG}\,\sigma_x,
   \qquad
   q_3 = \texttt{FWHM2SIG}\,\sigma_y;
   $$

4. covariance elements:

   $$
   q_4 = \sigma_x^2,\qquad
   q_5 = \rho\sigma_x\sigma_y,\qquad
   q_6 = \sigma_y^2;
   $$

5. principal sigmas:

   $$
   q_7 = \sigma_{\text{major}},\qquad
   q_8 = \sigma_{\text{minor}};
   $$

6. ellipse angle:

   $$
   q_9 = \frac12\operatorname{atan2}
   \left(2\rho\sigma_x\sigma_y,\sigma_x^2-\sigma_y^2\right).
   $$

The continuous expressions for the principal sigmas and angle are used
specifically to avoid the arbitrary sign of eigenvectors. A direct numerical
derivative of `np.linalg.eigh` could jump when the eigenvector flips sign.

### 10.5 Caveats for uncertainties

- `pcov` is a **local, linearized** covariance. It assumes the model is
  approximately linear in the parameters near the optimum.
- If a parameter is at a bound, the quadratic approximation is invalid and the
  reported error may be misleading. The current implementation does not
  explicitly flag parameters sitting on bounds.
- Correlated data points violate the diagonal-weight assumption. Bins produced
  by summing independent Poisson bins remain independent, but smoothing,
  interpolation, or overlapping bins introduce correlations.
- The angle `theta` is ill-conditioned for nearly circular beams or when
  $\rho\approx 0$. In that case the orientation is barely defined, and a large
  reported angle error is physically meaningful rather than a bug. Angle is
  also periodic, so linear propagation near a branch cut is approximate.
- The values are statistical only. Calibration, background-model, and
  beam-non-Gaussianity systematics are not included.

---

## 11. Goodness of fit

For the weighted fit, the code computes

$$
\chi^2
=
\sum_{i\in\text{ROI}}
\frac{(y_i-f_i)^2}{\max(y_i,1)},
$$

with

$$
\text{dof} = N_{\text{bins}} - N_{\text{parameters}}
= N_{\text{bins}} - 7,
$$

and

$$
\chi^2_\nu = \frac{\chi^2}{\text{dof}}.
$$

These are stored as:

- `chi2`
- `dof`
- `reduced_chi2`

and printed by `print_stats` when finite.

Interpretation:

- $\chi^2_\nu \approx 1$: the Poisson noise model and the Gaussian model are
  mutually consistent.
- $\chi^2_\nu \gg 1$: the model is incomplete, the errors are underestimated,
  or the bins are correlated.
- $\chi^2_\nu \ll 1$: the errors are overestimated, the data are smoother than
  the assumed Poisson model, or Neyman-weighting bias is present.

Neyman weighting uses the observed counts in the denominator, which introduces
a small negative bias in the chi-square even when the model is correct. Values
below 1 are therefore common. In the local CAX and rebinning checks,
$\chi^2_\nu$ was typically in the range 0.3–0.7.

For the unweighted fallback, `weighted=False` and both `chi2` and
`reduced_chi2` are `NaN`.

---

## 12. The standalone thresholding path

Thresholding is independent of the Gaussian fit. It is used for segmentation
and for producing `img_thresholded`; it does not set or constrain the fitted
`offset`.

### 12.1 Choosing the histogram bin count: `_number_of_bins`

The function flattens the values and uses:

1. Freedman–Diaconis (`bins="fd"`) bin edges, capped at 1024 bins;
2. Sturges' rule

   $$
   n_{\text{Sturges}} = \left\lfloor 1+\log_2 N\right\rfloor;
   $$

3. the maximum of the two, and at least 2.

The final count is

$$
n_{\text{bins}}
=
\max\left(
\min(n_{\text{FD}},1024),\;
n_{\text{Sturges}},\;
2
\right).
$$

### 12.2 Kapur cross-class entropy: `_kapur_entropy`

For a histogram frequency array $f_k$ and a split index $K$, define the
background and foreground distributions

$$
p^{(b)}_k =
\frac{f_k}{\sum_{j<K}f_j}, \quad k<K,
\qquad
p^{(f)}_k =
\frac{f_k}{\sum_{j\ge K}f_j}, \quad k\ge K.
$$

The Shannon entropies, in bits, are

$$
H_b = -\sum_{k<K} p^{(b)}_k \log_2 p^{(b)}_k,
\qquad
H_f = -\sum_{k\ge K} p^{(f)}_k \log_2 p^{(f)}_k.
$$

The Kapur objective is

$$
H(K) = H_b + H_f.
$$

A small constant $10^{-10}$ is added inside the logarithm to avoid
$\log 0$.

### 12.3 `compute_threshold`

The method:

1. Uses ROI values if `useroi=True` and a mask exists; otherwise warns and uses
   the full image.
2. Builds a 1D histogram with `_number_of_bins`.
3. Computes $H(K)$ for every split index $K=1,\dots,n_{\text{bins}}-1$.
4. Selects

   $$
   K^* = \arg\max_K H(K),
   \qquad
   \texttt{optimal\_threshold} = \text{left edge of bin }K^*.
   $$

5. Builds

   $$
   \texttt{img\_thresholded}_{ij}
   =
   \begin{cases}
   I_{ij}-\texttt{optimal\_threshold}, & I_{ij}>\texttt{optimal\_threshold}
   \text{ and inside the ROI},\\
   0, & \text{otherwise}.
   \end{cases}
   $$

It returns `(entropies, bin_edges, threshold, nbins)`.

`plot_entropy` plots the entropy curve against the interior bin edges and marks
the selected threshold.

This threshold is a histogram split value in counts per bin. It is unrelated to
the Gaussian baseline `offset`, which is a fitted model parameter.

---

## 13. Output dictionaries and attributes

### 13.1 Instance attributes

| Attribute | Meaning |
|---|---|
| `img` | 2D histogram of counts per bin |
| `x_bin_edges`, `y_bin_edges` | Validated uniform bin edges |
| `x_bin_centers`, `y_bin_centers` | Bin centres |
| `x_bin_size`, `y_bin_size` | Bin widths |
| `bin_area` | Bin area |
| `x_bin_width`, `y_bin_width` | Aliases for the bin widths |
| `droi` | Radius in bins for the local visibility / legacy ROI |
| `roi_sigma` | ROI half-width in projected sigmas |
| `roi_pad` | Extra ROI padding in bins |
| `roi_mask` | Boolean mask of the current rectangular ROI |
| `roi_bounds` | `(xmin, xmax, ymin, ymax)` for plotting |
| `beam_visible` | Result of the peak-to-average visibility test |
| `hprm_project` | Projection-stage parameter dictionary |
| `hprm_momenta` | Moment-stage parameter dictionary |
| `hprm_fitting` | Nonlinear-fit parameter dictionary |
| `hprm` | Legacy attribute, set to `None`; not used by the pipeline |
| `optimal_threshold` | Threshold selected by `compute_threshold` |
| `img_thresholded` | Thresholded, background-subtracted image |
| `data`, `pixel_size` | Legacy placeholders, not used by the pipeline |

### 13.2 Common keys in every `hprm` dictionary

| Key | Units | Meaning |
|---|---|---|
| `mux`, `muy` | coordinate | Centroid |
| `sigx`, `sigy` | coordinate | Standard deviations along x and y |
| `fwhmx`, `fwhmy` | coordinate | FWHM values |
| `cov` | coordinate² | 2×2 covariance matrix |
| `sig_major`, `sig_minor` | coordinate | Principal standard deviations |
| `theta` | rad | Ellipse orientation |
| `evecs` | dimensionless | Principal-axis eigenvectors |
| `amplitude` | counts / bin area | Peak height above offset |
| `offset` | counts / bin area | Constant background |
| `x_bin_centers`, `y_bin_centers` | coordinate | Bin centres |

### 13.3 Additional keys in `hprm_fitting`

| Key | Units | Meaning |
|---|---|---|
| `rho` | dimensionless | Correlation coefficient |
| `errors` | same as each parameter | 1σ uncertainties |
| `pcov` | parameter² | Full 7×7 parameter covariance matrix |
| `chi2` | dimensionless | Weighted chi-square |
| `dof` | dimensionless | Degrees of freedom, $N-7$ |
| `reduced_chi2` | dimensionless | $\chi^2/\text{dof}$ |
| `weighted` | bool | Whether the Poisson-weighted fit succeeded |

The `errors` dictionary contains:

```text
offset, amplitude, mux, muy, sigx, sigy, rho,
fwhmx, fwhmy, cov_xx, cov_xy, cov_yy,
sig_major, sig_minor, theta
```

The direct parameters come from `sqrt(diag(pcov))`; the derived quantities come
from the delta-method propagation described in
[Section 10](#10-parameter-uncertainties).

### 13.4 Stage-specific differences

- `hprm_project` has a diagonal covariance and zero correlation by construction.
  Its `evecs` is the identity.
- `hprm_momenta` has the full moment covariance, including correlation.
- `hprm_fitting` has the full fitted covariance, `rho`, uncertainties, and
  goodness-of-fit diagnostics.
- `hprm_project` and `hprm_momenta` do not contain an `errors` dictionary.

---

## 14. Display and diagnostics

### 14.1 `print_stats(hprm=None)`

If no dictionary is supplied, it selects the most advanced available result in
the order

```text
hprm_fitting -> hprm_momenta -> hprm_project
```

It prints:

- bin size and bin area;
- offset and amplitude in counts per bin area;
- `rho` when present;
- centroid, sigmas, FWHM values;
- covariance and principal sigmas;
- angle in radians and degrees;
- the covariance matrix;
- reduced chi-square and degrees of freedom when available.

When an `errors` dictionary is present, each value is printed as
`value +/- error`.

### 14.2 `plot(hprm=None, ...)`

`plot` renders the histogram with `_render_histogram`, draws the centroid,
draws `n`-sigma ellipses, optionally draws the principal-axis lines, and draws
the current ROI rectangle. The default `roi_zoom=1.2` zooms the axes to the ROI
plus 20%.

Ellipse dimensions are

$$
\text{width} = 2n\sigma_{\text{major}},
\qquad
\text{height} = 2n\sigma_{\text{minor}},
$$

with the ellipse rotated by `theta` in degrees.

### 14.3 `plot_entropy`

Plots the Kapur entropy curve from `compute_threshold` and marks the selected
threshold with a vertical line.

---

## 15. Numerical constants

| Constant | Value | Role |
|---|---|---|
| `FWHM2SIG` | $2\sqrt{2\ln 2}\approx2.35482$ | Gaussian sigma ↔ FWHM |
| `PEAK2AVG_THRESHOLD` | 5 | Minimum local/global mean ratio for visibility |
| `SIGMA_MIN_BINS` | 0.5 | Minimum fitted sigma in bins |
| `RHO_MAX` | $1-10^{-6}$ | Maximum absolute correlation |
| `DEFAULT_ROI_PAD` | 2.0 | Default ROI padding in bins |
| `ROI_REFINE_ITERATIONS` | 3 | Maximum moment-based ROI refinement passes |
| `FIT_MAXFEV` | 20000 | Maximum function evaluations for `curve_fit` |
| `OFFSET_UPPER_NSIGMA` | 5.0 | Offset upper-bound allowance in robust sigmas |
| `AMPLITUDE_MAX_FACTOR` | 2.0 | Amplitude upper bound relative to brightest bin |

Also relevant:

- default `roi_sigma = 5.0`;
- default `droi = 4` bins;
- default `DEFAULT_MEAN = [0, 0]`;
- default `DEFAULT_COV = [[1, 0], [0, 1]]`;
- default `DEFAULT_HIST_RANGE = [[-8, 8], [-8, 8]]`.

---

## 16. Assumptions, limitations and best practices

### 16.1 Binning and data

- Bins must be uniform on each axis.
- Counts are assumed to be independent Poisson variables for the purpose of
  weighting and uncertainty estimation.
- Rebinning by summing disjoint bins preserves the Poisson property. Smoothing,
  interpolation, or overlapping bins introduce correlations that invalidate the
  diagonal covariance assumption.
- `from_gaussian` and `add_noise` are test utilities. `add_noise` adds a second
  Poisson draw to an existing histogram, so the final array is not itself a
  pure Poisson sample of a single mean. Its effective variance is smaller than
  the observed counts; the Poisson-weighted uncertainties and chi-square are
  therefore approximate for such synthetic data.
- Density-normalized amplitude and offset are stable under rebinning only when
  the physical beam is resolved. Very coarse bins bias any width estimator.

### 16.2 Beam geometry and model

- The model is a single elliptical Gaussian with a constant offset. Non-Gaussian
  tails, halos, multiple peaks, or a background gradient will bias the fit and
  inflate the reduced chi-square.
- The projection stage assumes the beam is the dominant feature in each marginal
  projection.
- The visibility test is a heuristic contrast test.
- The ROI refinement uses raw counts, so a strong flat background inflates the
  moment-derived ROI.
- The matched filter fixes the shape from the current centroid and covariance.
  It is a good initializer, not a final estimator.
- The full fit assumes the beam is contained in the ROI and that the ROI is not
  dominated by background.

### 16.3 Uncertainties and fits

- `pcov` is local and linearized. It is unreliable when a parameter is at a
  bound, when the model is badly misspecified, or when parameters are strongly
  degenerate.
- `absolute_sigma=True` means the errors are not rescaled by the observed
  scatter. A reduced chi-square far from 1 indicates that the absolute errors
  should be interpreted cautiously.
- The angle uncertainty is ill-conditioned for circular beams and should not be
  over-interpreted.
- Neyman weights using observed counts are a high-count approximation to the
  Poisson likelihood. For very low-count data, a full Poisson-likelihood fit or
  model-based Pearson residuals would be preferable.
- The thresholding path is separate from the Gaussian fit. `optimal_threshold`
  should not be confused with the fitted baseline `offset`.

### 16.4 Practical recommendations

- Inspect `print_stats()` and `plot()` after a fit. A good fit should have a
  visible Gaussian shape, a finite reduced chi-square, and parameters that do
  not sit on a bound.
- Use `fit(hprm=...)` only with a dictionary produced by this module, because
  amplitude and offset are expected in density units.
- For rebinned images, confirm that `bin_area` changed as expected and that
  `amplitude` is approximately stable.
- If the beam is very broad or very close to an image edge, check `roi_bounds`
  and `roi_mask`; increase `roi_pad` or `roi_sigma` if the ROI truncates the
  beam.
- If the beam is nearly circular, treat the fitted angle as essentially
  undefined.

---

## 17. Testing and validation

Fast regression tests live in
[`tests/test_image_statistics.py`](../tests/test_image_statistics.py). They
cover:

- bin attributes and the legacy width aliases;
- amplitude normalization by bin area;
- marginal-projection widths for a 45° tilted beam;
- Gaussian parameter recovery at several orientations;
- nonnegative fitted offsets on zero-background images;
- finite parameter uncertainties and finite reduced chi-square;
- consistency of amplitude and sigmas under rebinning;
- deterministic `from_gaussian` output with a fixed seed.

Run them with:

```bash
MPLBACKEND=Agg python -m pytest tests/test_image_statistics.py -q
```

The slower CAXSim/Shadow validation is intentionally not part of the committed
test suite. It can be run locally with `CAXSim(total_rays=500000,
image_size=(150, 150))`, followed by analysis of `cax.dvf_B1.image`. In local
checks, the fit results were stable at the few-percent level when the CAX image
was progressively rebinned with successive 2×2 bin sums, and the offset stayed
nonnegative.

---

## Appendix A: End-to-end equation summary

For a single fit, the complete calculation is:

1. **Projection widths**

   $$
   \mathrm{FWHM}_x = \text{half-max width of }\sum_j I_{ij},
   \qquad
   \sigma_x = \frac{\mathrm{FWHM}_x}{2\sqrt{2\ln2}}.
   $$

2. **ROI**

   $$
   w_x = \texttt{roi\_sigma}\,\sigma_x+\texttt{roi\_pad}\,\Delta x,
   \qquad
   \text{mask on } |x-\mu_x|\le w_x,\; |y-\mu_y|\le w_y.
   $$

3. **Matched filter**

   $$
   \min_{o,A}\sum_{i\in\text{ROI}}
   (c_i-o-A g_i)^2.
   $$

4. **Full model in counts per bin**

   $$
   f_i = o+A\exp\left(-\frac12
   (\mathbf{r}_i-\boldsymbol{\mu})^T\Sigma^{-1}
   (\mathbf{r}_i-\boldsymbol{\mu})\right).
   $$

5. **Weighted fit**

   $$
   \min_p\sum_{i\in\text{ROI}}
   \frac{(c_i-f_i(p))^2}{\max(c_i,1)}.
   $$

6. **Parameter covariance**

   $$
   \operatorname{Cov}(\hat p)\approx
   \left(J^TWJ\right)^{-1},
   \qquad
   W_{ii}=\frac{1}{\max(c_i,1)}.
   $$

7. **Derived uncertainty**

   $$
   \operatorname{Cov}(g(\hat p))\approx
   J_g\,\operatorname{Cov}(\hat p)\,J_g^T.
   $$

8. **Public density values**

   $$
   \texttt{amplitude}=\frac{A}{\texttt{bin\_area}},
   \qquad
   \texttt{offset}=\frac{o}{\texttt{bin\_area}}.
   $$
