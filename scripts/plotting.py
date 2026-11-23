"""Plotting helpers shared by the knob-space and optimization notebooks.

This module centralises the uncertainty-aware highlight bands, the objective
maps, the optimizer-history panels and the knob-space path overlays that were
previously copy-pasted between notebooks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from scripts.knobspace_data import REPO_ROOT, to_grid
from scripts.objective_functions import (
    OBJECTIVE_LABELS,
    OBJECTIVE_NAMES,
    OBSERVABLES,
    ORDER,
    objective_grids,
)

FIGURES_DIR = REPO_ROOT / "figures"


def set_plot_style(*, use_latex: bool = False, font_size: int = 12) -> None:
    """Apply a reproducible base style without importing LaTeX by default."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": font_size,
            "text.usetex": use_latex,
        }
    )
    if use_latex:
        plt.rcParams["font.serif"] = ["Computer Modern"]


def save_figure(
    fig,
    name: str,
    output_dir: str | Path | None = None,
    *,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in several formats into a relative output directory."""
    directory = Path(output_dir) if output_dir is not None else FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def _legend_proxy(label: str) -> Patch:
    return Patch(facecolor="red", alpha=0.3, label=label)


def _finish_legend(ax, handles, labels) -> None:
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )


def add_minimum_error_band(ax, x, y, values, errors, *, legend: bool = True):
    """Highlight values within the measurement error of the global minimum."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    finite = np.where(np.isfinite(values), values, np.inf)
    idx = np.unravel_index(np.argmin(finite), values.shape)
    minimum = values[idx]
    error = errors[idx]
    upper = minimum + error

    ax.contourf(x, y, values, levels=[minimum, upper], colors=["red"], alpha=0.3)
    if legend:
        _finish_legend(
            ax,
            [_legend_proxy(rf"$\leq$ min + $\sigma$ ({error:.3g})")],
            [rf"$\leq$ min + $\sigma$ ({error:.3g})"],
        )
    return idx, minimum, error


def add_minimum_percent_band(
    ax, x, y, values, errors, percent: float, *, k: int = 1, legend: bool = True
):
    """Highlight values within a relative tolerance of the minimum."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = np.isfinite(values) & np.isfinite(errors)
    if not np.any(valid):
        return None

    minimum = np.min(values[valid])
    idx_min = np.unravel_index(
        np.nanargmin(np.where(valid, values, np.nan)), values.shape
    )
    sigma_min = errors[idx_min]

    if minimum < 0:
        upper = minimum - (percent / 100) * (minimum + k * sigma_min)
    else:
        upper = minimum + (percent / 100) * (minimum + k * sigma_min)

    ax.contourf(
        x,
        y,
        np.ma.masked_invalid(values),
        levels=[minimum, upper],
        colors=["red"],
        alpha=0.3,
    )

    if legend:
        label = (
            rf"$<$ min $+ ~{percent:.1f}\%$ of min $+ {k} \sigma ~({sigma_min:.3g})$"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(_legend_proxy(label))
        labels.append(label)
        _finish_legend(ax, handles, labels)
    return idx_min, minimum, sigma_min, upper


def add_zero_error_band(ax, x, y, values, errors, *, legend: bool = True):
    """Highlight values statistically consistent with zero."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = np.abs(values) <= np.abs(errors)

    if mask.any():
        ax.contourf(
            x, y, mask.astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.3
        )

    if legend:
        label = r"$|c| \leq \sigma$"
        _finish_legend(ax, [_legend_proxy(label)], [label])
    return mask


def _observable_values(frame, column, error_column, mode):
    values = to_grid(frame, column)
    errors = to_grid(frame, error_column)
    if mode == "neg_min":
        values = -values
    elif mode == "abs_min":
        values = np.abs(values)
    return values, errors


def _panel_contours(ax, x, y, values, upper, mode):
    if mode == "zero":
        ax.contour(x, y, values, levels=10, colors="white", linewidths=0.2)
    else:
        levels = np.linspace(upper, np.nanmax(values), 15)
        ax.contour(x, y, values, levels=levels, colors="white", linewidths=0.2)


def plot_observables(frame: pd.DataFrame, tx, ry, title: str = "Observables"):
    """Six-panel observable map with uncertainty-aware highlight bands."""
    fig, axs = plt.subplots(3, 2, figsize=(17, 16))
    fig.suptitle(title, fontsize=16)

    for ax, (column, err_column, name, clabel, mode) in zip(axs.flat, OBSERVABLES):
        values, errors = _observable_values(frame, column, err_column, mode)
        mesh = ax.pcolormesh(tx, ry, values, shading="auto", rasterized=True)

        upper = None
        if mode == "zero":
            add_zero_error_band(ax, tx, ry, values, errors)
            idx = np.unravel_index(np.argmin(np.abs(values)), values.shape)
            tx_opt, ry_opt = tx[idx], ry[idx]
            inside = bool(np.abs(values[idx]) <= errors[idx])
            stats = (
                f"closest to 0 at ({tx_opt:+.3f}, {ry_opt:+.3f}) = "
                f"{values[idx]:+.4f}  ({'inside' if inside else 'outside'} $\\sigma$)"
            )
        else:
            band = add_minimum_percent_band(
                ax, tx, ry, values, errors, percent=1
            )
            if band is None:
                continue
            idx, minimum, error, upper = band
            tx_opt, ry_opt = tx[idx], ry[idx]
            if mode == "neg_min":
                stats = (
                    f"max at ({tx_opt:+.3f}, {ry_opt:+.3f}) = "
                    f"{-values[idx]:.3f} $\\pm$ {error:.3f}"
                )
            else:
                stats = (
                    f"min at ({tx_opt:+.3f}, {ry_opt:+.3f}) = "
                    f"{minimum:.4f} $\\pm$ {error:.4f}"
                )

        _panel_contours(ax, tx, ry, values, upper, mode)
        ax.scatter(tx_opt, ry_opt, c="white", marker="d", s=30, zorder=5)
        ax.set_xlabel("TX [mm]")
        ax.set_ylabel("RY [mrad]")
        ax.set_title(f"{name}: {stats}")
        fig.colorbar(mesh, ax=ax, label=clabel)
        ax.set_xlim(-2, 2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_objectives(frame: pd.DataFrame, tx, ry, title: str = "Objectives"):
    """Plot all eight objective maps."""
    objectives, errors = objective_grids(frame)
    fig, axs = plt.subplots(4, 2, figsize=(17, 24))
    fig.suptitle(title, fontsize=16)

    for ax, key in zip(axs.flat, ORDER):
        values = objectives[key]
        err = errors[key]

        mesh = ax.pcolormesh(tx, ry, values, shading="auto")
        band = add_minimum_percent_band(
            ax, tx, ry, values, err, percent=1, k=1, legend=(key == "f1")
        )
        if band is None:
            continue
        idx, minimum, uncertainty, upper = band
        tx_opt, ry_opt = tx[idx], ry[idx]

        levels = np.linspace(upper, np.nanmax(values), 15)
        ax.contour(tx, ry, values, levels=levels, colors="white", linewidths=0.2)
        ax.scatter(tx_opt, ry_opt, c="white", marker="d", s=30, zorder=5)
        ax.set_xlabel("TX [mm]")
        ax.set_ylabel("RY [mrad]")
        ax.set_title(
            f"{OBJECTIVE_NAMES[key]} (${key}$): min at "
            f"({tx_opt:+.3f}, {ry_opt:+.3f}) = {minimum:.4f} $\\pm$ {uncertainty:.4f}"
        )
        fig.colorbar(mesh, ax=ax, label=OBJECTIVE_LABELS[key])
        ax.set_xlim(-2, 2)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_single_objective(
    frame: pd.DataFrame,
    objective_key: str,
    tx,
    ry,
    title: str | None = None,
    *,
    fig=None,
    ax=None,
    percent: float = 5,
):
    """Plot one objective map, optionally drawing on an existing axis."""
    objectives, errors = objective_grids(frame)
    values = objectives[objective_key]
    err = errors[objective_key]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(tx, ry, values, shading="auto")
    band = add_minimum_percent_band(ax, tx, ry, values, err, percent=percent, k=1)
    if band is not None:
        idx, minimum, uncertainty, upper = band
        tx_opt, ry_opt = tx[idx], ry[idx]
        levels = np.linspace(upper, np.nanmax(values), 15)
        ax.contour(tx, ry, values, levels=levels, colors="white", linewidths=0.2)
        ax.scatter(tx_opt, ry_opt, c="white", marker="d", s=30, zorder=5)
        stats = (
            f"min at ({tx_opt:+.3f}, {ry_opt:+.3f}) = "
            f"{minimum:.4f} $\\pm$ {uncertainty:.4f}"
        )
    else:
        stats = "no finite objective values"

    ax.set_xlabel("TX [mm]")
    ax.set_ylabel("RY [mrad]")
    ax.set_title(title or f"{OBJECTIVE_NAMES[objective_key]} (${objective_key}$): {stats}")
    fig.colorbar(mesh, ax=ax, label=OBJECTIVE_LABELS[objective_key])
    ax.set_xlim(-2, 2)
    fig.tight_layout()
    return fig, ax


def plot_optimization_data(opt, k: int = 1):
    """Four-panel history of the beam observables recorded during RCDS."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.1)

    x = np.arange(len(opt.observables["f1"]))[::k]
    sigx = np.asarray(opt.observables["sig_major"])[::k]
    sigy = np.asarray(opt.observables["sig_minor"])[::k]
    sigx_err = np.asarray(opt.observable_errors["sig_major"])[::k]
    sigy_err = np.asarray(opt.observable_errors["sig_minor"])[::k]

    ax00r = axs[0, 0].twinx()
    axs[0, 0].errorbar(x, sigx, yerr=sigx_err, marker="o", c="blue")
    ax00r.errorbar(x, sigy, yerr=sigy_err, marker="s", c="red")
    axs[0, 0].set_ylabel(r"$\sigma_x ~[\mu m]$", c="blue")
    ax00r.set_ylabel(r"$\sigma_y ~[\mu m]$", c="red")
    axs[0, 0].tick_params(axis="y", colors="blue")
    ax00r.tick_params(axis="y", colors="red")
    axs[0, 0].set_xlim(x[0], x[-1])
    axs[0, 0].grid(which="both", color="blue")
    ax00r.grid(which="both", color="red")

    peak = np.asarray(opt.observables["peak"])[::k]
    peak_err = np.asarray(opt.observable_errors["peak"])[::k]
    axs[0, 1].errorbar(x, peak, yerr=peak_err, marker="o")
    axs[0, 1].grid(which="both")
    axs[0, 1].set_ylabel(r"$I ~[\text{count}/\mu m ^2]$")

    sig_xy = sigx * sigy
    sig_xy_err = np.sqrt((sigx_err / sigx) ** 2 + (sigy_err / sigy) ** 2)
    axs[1, 0].errorbar(x, sig_xy, yerr=sig_xy_err * sig_xy, marker="o")
    axs[1, 0].grid(which="both")
    axs[1, 0].set_ylabel(r"$\sigma_x\sigma_y ~[\mu m ^2]$")

    f1 = np.asarray(opt.observables["f1"])[::k]
    f1_err = np.asarray(opt.observable_errors["f1"])[::k]
    axs[1, 1].errorbar(x, f1, yerr=f1_err, marker="o")
    axs[1, 1].grid(which="both")
    axs[1, 1].set_ylabel(r"$f_1(\sigma_x, \sigma_y, I) ~[\mu m ^4/\text{count}]$")

    axs[1, 0].set_xlabel("Evaluation")
    axs[1, 1].set_xlabel("Evaluation")
    return fig


def plot_knobspace_slice(optimizer, dir_idcs=(1, 2), obj_idx=None, *, labels=None):
    """Plot a 2D slice of the optimizer's parameter space and its path."""
    idx1, idx2 = np.array(dir_idcs) - 1

    pos_eval = np.asarray(optimizer.positions_evaluated)
    pos_best = np.asarray(optimizer.positions_best)
    objfuncs_eval = np.asarray(optimizer.objfuncs_evaluated)
    if obj_idx is not None:
        objfuncs_eval = objfuncs_eval[:, obj_idx]

    _, pos_cum_opt, _ = optimizer.get_cumul_optima(obj_idx)

    x_eval, y_eval = pos_eval[:, idx1], pos_eval[:, idx2]
    x_best, y_best = pos_best[:, idx1], pos_best[:, idx2]
    x_cum, y_cum = pos_cum_opt[:, idx1], pos_cum_opt[:, idx2]

    if labels is None:
        config = getattr(optimizer, "config", None)
        knobs = getattr(config, "knobs", None)
        labels = (
            (knobs[idx1], knobs[idx2])
            if knobs is not None and len(knobs) > max(idx1, idx2)
            else (f"dir {idx1 + 1}", f"dir {idx2 + 1}")
        )

    fig, ax = plt.subplots()

    ax.scatter(
        x_eval,
        y_eval,
        c=objfuncs_eval,
        cmap="viridis",
        label="positions evaluated",
        zorder=1,
    )
    ax.plot(
        x_best, y_best, "x", color="red", label="end-of-iteration optima", zorder=3
    )

    for i, (x, y) in enumerate(zip(x_best, y_best)):
        if i % 2 == 0:
            ax.annotate(
                str(i),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="white",
                path_effects=[
                    patheffects.withStroke(linewidth=1.5, foreground="black")
                ],
                zorder=5,
            )

    for i in range(len(x_best) - 1):
        ax.annotate(
            "",
            xy=(x_best[i + 1], y_best[i + 1]),
            xytext=(x_best[i], y_best[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": "white",
                "lw": 0.8,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=4,
        )

    ax.plot(
        x_cum,
        y_cum,
        "o",
        color="red",
        mfc="none",
        label="cumulative optima",
        zorder=4,
    )
    ax.plot(
        x_best[-1],
        y_best[-1],
        "d",
        color="magenta",
        markersize=10,
        label="final optimum",
        zorder=6,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("parameter space slice")
    return fig, ax
