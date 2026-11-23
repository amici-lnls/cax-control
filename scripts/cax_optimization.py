"""RCDS optimization helpers for the CAX mirror knobs.

One generic optimizer replaces the four near-identical classes that were
defined inside the original notebook.  The knob order is part of the
configuration, so TXRY, TYRX, TYRXRZ and 5D studies share the same objective,
history recording and plotting code.

External dependencies (apsuite/shadow_interface) are imported defensively so
this module can be imported for documentation and unit tests without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.caustic import fit_beam_image
from scripts.knobspace_data import DEFAULT_DATA_DIR, Z_FOCUS
from scripts.objective_functions import f1_from_measurements

try:  # pragma: no cover - depends on the runtime environment
    from apsuite.optimization.rcds import RCDS, RCDSParams
except Exception:  # pragma: no cover
    RCDS = None

    class RCDSParams:  # minimal stand-in for import-only use
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __str__(self):
            return f"<RCDSParams {self.__dict__}>"


try:  # pragma: no cover
    from shadow_interface.si_beamlines import CAXSim
except Exception:  # pragma: no cover
    try:
        from shadowpy.si_beamlines import CAXSim
    except Exception:
        CAXSim = None

try:  # pragma: no cover
    from shadow_interface.utils import image_from_beam
except Exception:  # pragma: no cover
    try:
        from shadowpy.utils import image_from_beam
    except Exception:
        image_from_beam = None

RUNS_DIR = DEFAULT_DATA_DIR / "rcds_runs"


class _FallbackRCDS:  # pragma: no cover - only used when apsuite is absent
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("apsuite.optimization.rcds.RCDS is not installed")


_RCDSBase = RCDS if RCDS is not None else _FallbackRCDS


@dataclass
class CAXRCDSConfig:
    """All settings that distinguish one knob-space optimization study."""

    name: str
    knobs: tuple[str, ...]
    limit_lower: tuple[float, ...]
    limit_upper: tuple[float, ...]
    z_pos: float = Z_FOCUS
    total_rays: int = 100_000
    image_size: tuple[int, int] = (210, 280)
    max_number_evals: int = 500
    max_number_iters: int = 13
    initial_stepsize: float = 0.5
    noise_level: float | None = None
    tolerance: float = 1e-8
    update_search_directions: bool = True
    initial_position: tuple[float, ...] | None = None
    initial_search_directions: np.ndarray | None = None
    fixed_mirror: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        n = len(self.knobs)
        for field_name in ("limit_lower", "limit_upper"):
            values = getattr(self, field_name)
            if len(values) != n:
                raise ValueError(f"{field_name} must have {n} entries for {self.knobs}")
        if any(lo >= hi for lo, hi in zip(self.limit_lower, self.limit_upper)):
            raise ValueError("Every lower limit must be smaller than its upper limit")
        if self.initial_position is not None and len(self.initial_position) != n:
            raise ValueError(f"initial_position must have {n} entries for {self.knobs}")
        if self.initial_search_directions is not None:
            directions = np.asarray(self.initial_search_directions, dtype=float)
            if directions.shape != (n, n):
                raise ValueError(
                    f"initial_search_directions must have shape ({n}, {n})"
                )


def sample_initial_position(config: CAXRCDSConfig) -> np.ndarray:
    """Return the configured start point, or a seeded uniform random one."""
    if config.initial_position is not None:
        return np.asarray(config.initial_position, dtype=float)
    rng = np.random.default_rng(config.seed)
    return rng.uniform(config.limit_lower, config.limit_upper)


def _apply_fixed_mirror(caxsim, fixed_mirror: dict[str, float]) -> None:
    for knob, value in fixed_mirror.items():
        setattr(caxsim.mirror, knob, float(value))


def build_params(config: CAXRCDSConfig):
    """Build the ``RCDSParams`` object consumed by apsuite's RCDS."""
    if CAXSim is None:
        raise RuntimeError(
            "CAXSim is unavailable; install shadow_interface or shadowpy to run "
            "an optimization."
        )

    params = RCDSParams()
    params.limit_lower = list(config.limit_lower)
    params.limit_upper = list(config.limit_upper)
    params.caxsim = CAXSim(
        total_rays=config.total_rays, image_size=config.image_size
    )
    _apply_fixed_mirror(params.caxsim, config.fixed_mirror)
    params.z_pos = config.z_pos
    params.max_number_evals = config.max_number_evals
    params.max_number_iters = config.max_number_iters
    params.initial_stepsize = config.initial_stepsize
    params.noise_level = config.noise_level if config.noise_level is not None else 1e-3
    params.tolerance = config.tolerance
    params.update_search_directions = config.update_search_directions
    params.initial_position = sample_initial_position(config)
    if config.initial_search_directions is None:
        params.initial_search_directions = np.eye(len(config.knobs))
    else:
        params.initial_search_directions = np.asarray(
            config.initial_search_directions, dtype=float
        )
    params.config = config
    return params


def measure_f1(cax, z_pos: float, *, image_size=(210, 280)):
    """Measure f1 and its monitored observables at one mirror position.

    Returns ``(observables, errors)`` dictionaries with keys ``f1``, ``peak``,
    ``sig_major`` and ``sig_minor``.  Unvisible beams produce all-NaN entries.
    """
    if image_from_beam is None:
        raise RuntimeError("image_from_beam is unavailable in this environment")

    observables = dict.fromkeys(("f1", "peak", "sig_major", "sig_minor"), np.nan)
    observable_errors = observables.copy()

    beamr = cax.dvf_B1.beam.duplicate()
    beamr.retrace(z_pos)
    image = image_from_beam(beamr, cax.dvf_B1.frame, image_size=image_size)
    ana = fit_beam_image(image)

    if not getattr(ana, "beam_visible", False) or getattr(ana, "hprm_fitting", None) is None:
        return observables, observable_errors

    fit = ana.hprm_fitting
    errors = fit.get("errors", {})
    for name, key in (
        ("peak", "amplitude"),
        ("sig_major", "sig_major"),
        ("sig_minor", "sig_minor"),
    ):
        observables[name] = fit[key]
        observable_errors[name] = errors.get(key, np.nan)

    observables["f1"], observable_errors["f1"] = f1_from_measurements(
        observables["peak"],
        observables["sig_major"],
        observables["sig_minor"],
        observable_errors["peak"],
        observable_errors["sig_major"],
        observable_errors["sig_minor"],
    )
    return observables, observable_errors


def compare_conditions(offsets, tilts, z_pos: float, *, image_size=(210, 280)):
    """
    Compare two distinct CAX conditions, e.g. before and after the optimization
    process.
    """
    beam_props = {}
    cax = CAXSim(image_size=image_size)
    for cond, (off, rot) in enumerate(zip(offsets, tilts)):
        
        cax.mirror.offset = off
        cax.mirror.tilt   = rot

        cax.trace()

        beamr = cax.dvf_B1.beam.duplicate()
        beamr.retrace(z_pos)
        image = image_from_beam(beamr, cax.dvf_B1.frame, image_size=image_size)
        ana = fit_beam_image(image)



class CAXRCDS(_RCDSBase):
    """RCDS optimizer that records every measured observable and its error."""

    def __init__(
        self,
        config: CAXRCDSConfig | None = None,
        params=None,
        *,
        use_thread: bool = False,
        isonline: bool = False,
    ):
        super().__init__(use_thread=use_thread, isonline=isonline)
        if params is None:
            if config is None:
                raise ValueError("Provide config or params")
            params = build_params(config)
        self.params = params
        self.config = config if config is not None else getattr(params, "config", None)
        self.observables: dict[str, list[float]] = {}
        self.observable_errors: dict[str, list[float]] = {}

    def _apply_position(self, pos) -> None:
        if self.config is None:
            raise ValueError("This optimizer has no configuration")
        pos = np.asarray(pos, dtype=float).ravel()
        if pos.size != len(self.config.knobs):
            raise ValueError(
                f"Expected {len(self.config.knobs)} knob values "
                f"({self.config.knobs}), got {pos.size}"
            )
        caxsim = self.params.caxsim
        _apply_fixed_mirror(caxsim, self.config.fixed_mirror)
        for knob, value in zip(self.config.knobs, pos):
            setattr(caxsim.mirror, knob, float(value))

    def objective_function(self, pos):
        self._apply_position(pos)
        self.params.caxsim.trace()
        return self._measure_objective()

    def _measure_objective(self):
        """Append one sample per measurement and return the scalar objective."""
        values, errors = measure_f1(
            self.params.caxsim,
            self.params.z_pos,
            image_size=self.config.image_size if self.config else (210, 280),
        )
        for name, value in values.items():
            self.observables.setdefault(name, []).append(value)
            self.observable_errors.setdefault(name, []).append(errors[name])

        # TXRY used the measured f1 uncertainty as RCDS noise level; the other
        # studies can pin it in their config instead.
        if self.config is None or self.config.noise_level is None:
            error = errors["f1"]
            if np.isfinite(error) and error > 0:
                self.params.noise_level = float(error)
        return values["f1"]

    def history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.observables)

    def error_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.observable_errors)


def run(config: CAXRCDSConfig, **optimizer_kwargs) -> CAXRCDS:
    """Create an optimizer for ``config`` and run it to completion."""
    opt = CAXRCDS(config=config, **optimizer_kwargs)
    opt.start()
    return opt


def save_histories(opt: CAXRCDS, output_dir: str | Path | None = None) -> list[Path]:
    """Write the observable and error histories to CSV files."""
    directory = Path(output_dir) if output_dir is not None else RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    name = getattr(getattr(opt, "config", None), "name", "optimization")
    values_path = directory / f"{name}_observables.csv"
    errors_path = directory / f"{name}_errors.csv"
    opt.history_dataframe().to_csv(values_path, index=False)
    opt.error_dataframe().to_csv(errors_path, index=False)
    return [values_path, errors_path]


def optimization_summary(opt, f1_grid=None) -> dict[str, float | int | str]:
    """Compact before/after summary, optionally including the grid minimum."""
    evaluated = np.asarray(opt.objfuncs_evaluated, dtype=float)
    best      = np.asarray(opt.objfuncs_best, dtype=float)
    positions = np.asarray(opt.positions_evaluated, dtype=float)
    summary: dict[str, float | int | str] = {
        "study": getattr(getattr(opt, "config", None), "name", "optimization"),
        "evaluations": int(getattr(opt, "num_objective_evals", evaluated.size)),
        "initial objective": float(evaluated[0]),
        "best objective": float(best[-1]),
        "initial position": positions[0],
        "best position": np.asarray(opt.positions_best[-1], dtype=float),
    }
    if f1_grid is not None:
        summary["f1 grid minimum"] = float(np.nanmin(f1_grid))
    return summary


def make_txry_config(*, seed: int | None = None, **kwargs) -> CAXRCDSConfig:
    """TX/RY optimization at the retraced DVF position (original study)."""
    defaults = {
        "name": "TXRY",
        "knobs": ("tx", "ry"),
        "limit_lower": (-2.0, -0.1),
        "limit_upper": (2.0, 0.1),
        "max_number_evals": 500,
        "max_number_iters": 13,
        "noise_level": None,  # use the measured f1 uncertainty
        "seed": seed,
        "description": "Optimise TX/RY on the retraced z=focus image.",
    }
    defaults.update(kwargs)
    return CAXRCDSConfig(**defaults)


def make_tyrx_config(*, seed: int | None = None, **kwargs) -> CAXRCDSConfig:
    """TY/RX optimization with TX and RY prealigned (original study)."""
    defaults = {
        "name": "TYRX",
        "knobs": ("ty", "rx"),
        "limit_lower": (-0.5, -0.1),
        "limit_upper": (0.5, 0.1),
        "fixed_mirror": {"tx": 0.0, "ry": 25.255e-3},
        "max_number_evals": 500,
        "max_number_iters": 13,
        "noise_level": 1e-3,
        "seed": seed,
        "description": "Optimise TY/RX with TX=0 and RY=25.255e-3 mrad fixed.",
    }
    defaults.update(kwargs)
    return CAXRCDSConfig(**defaults)


def make_tyrxrz_config(*, seed: int | None = None, **kwargs) -> CAXRCDSConfig:
    """TY/RX/RZ study derived from the prealigned TY/RX setup."""
    defaults = {
        "name": "TYRXRZ",
        "knobs": ("ty", "rx", "rz"),
        "limit_lower": (-0.5, -0.1, -0.1),
        "limit_upper": (0.5, 0.1, 0.1),
        "fixed_mirror": {"tx": 0.0, "ry": 25.255e-3},
        "max_number_evals": 600,
        "max_number_iters": 20,
        "noise_level": 1e-3,
        "seed": seed,
        "description": (
            "Optimise TY/RX/RZ with TX=0 and RY=25.255e-3 mrad fixed; "
            "RZ uses the TY/RX tilt range from the prealigned study."
        ),
    }
    defaults.update(kwargs)
    return CAXRCDSConfig(**defaults)


def make_5d_config(*, seed: int | None = None, **kwargs) -> CAXRCDSConfig:
    """Full five-knob TX/TY/RX/RY/RZ optimization (original study)."""
    defaults = {
        "name": "5D",
        "knobs": ("tx", "ty", "rx", "ry", "rz"),
        "limit_lower": (-2.0, -1.0, -0.5, -0.1, -0.5),
        "limit_upper": (2.0, 1.0, 0.5, 0.1, 0.5),
        "max_number_evals": 500,
        "max_number_iters": 13,
        "noise_level": 1e-3,
        "seed": seed,
        "description": "Optimise TX, TY, RX, RY and RZ simultaneously.",
    }
    defaults.update(kwargs)
    return CAXRCDSConfig(**defaults)
