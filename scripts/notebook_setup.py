"""Small bootstrap used at the top of the CAX analysis notebooks.

It makes the repository importable regardless of the directory Jupyter was
started from and applies the shared plotting style.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.knobspace_data import DEFAULT_DATA_DIR, REPO_ROOT
from scripts.plotting import FIGURES_DIR, set_plot_style

RUNS_DIR = DEFAULT_DATA_DIR / "rcds_runs"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the checkout containing ``scripts/caustic.py``."""
    base = Path(start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "scripts" / "caustic.py").exists():
            return candidate
    return REPO_ROOT


def ensure_repo_root(start: str | Path | None = None) -> Path:
    """Insert the repository root into ``sys.path`` exactly once."""
    root = find_repo_root(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def configure_notebook(*, use_latex: bool = False, start: str | Path | None = None) -> Path:
    """Prepare imports and plotting for a notebook cell."""
    root = ensure_repo_root(start)
    set_plot_style(use_latex=use_latex)
    return root
