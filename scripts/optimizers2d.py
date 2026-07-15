import numpy as np

# ---------------- test function: rotated, elongated quadratic bowl ----------------
# f(x,y) = 0.5 * v^T H v ,  v = point - center
# H built from a steep eigenvalue and a shallow (valley) eigenvalue, rotated so the
# valley is NOT aligned with either axis -> naive coordinate descent zig-zags, and
# Powell's method needs to actually build the conjugate ("valley") direction.

CENTER = np.array([1.0, 0.5])
THETA_DEG = 35.0
LAM_STEEP = 1.0
LAM_VALLEY = 0.03

_theta = np.deg2rad(THETA_DEG)
_R = np.array([[np.cos(_theta), np.sin(_theta)],
               [-np.sin(_theta), np.cos(_theta)]])
H = _R.T @ np.diag([LAM_STEEP, LAM_VALLEY]) @ _R

_w, _v = np.linalg.eigh(H)
VALLEY_DIR = _v[:, np.argmin(_w)]  # unit vector along the shallow direction


def f(x, y):
    vx, vy = x - CENTER[0], y - CENTER[1]
    return 0.5 * (H[0, 0] * vx * vx + 2 * H[0, 1] * vx * vy + H[1, 1] * vy * vy)


def f_vec(p):
    v = p - CENTER
    return 0.5 * v @ H @ v


def grad_vec(p):
    v = p - CENTER
    return H @ v


def exact_line_search(x, d):
    """Exact 1D minimizer of f(x + t*d) along direction d (f is quadratic)."""
    d = np.asarray(d, dtype=float)
    denom = d @ H @ d
    if abs(denom) < 1e-14:
        return x.copy(), 0.0
    t_star = -(d @ H @ (x - CENTER)) / denom
    x_new = x + t_star * d
    return x_new, t_star


# ---------------- naive alternating coordinate descent ----------------
def coordinate_descent_history(start, n_cycles=14):
    """
    Each entry = one 1D line search (alternating e1, e2 directions).
    Returns list of dicts: start point, end point, direction, cycle index, axis label.
    """
    x = np.array(start, dtype=float)
    hist = []
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    labels = ["x-direction", "y-direction"]
    for cyc in range(n_cycles):
        for d, lab in zip(axes, labels):
            x_new, t = exact_line_search(x, d)
            hist.append(dict(x_start=x.copy(), x_end=x_new.copy(), direction=d.copy(),
                              label=lab, cycle=cyc, f_end=f_vec(x_new)))
            x = x_new
    return hist


# ---------------- Powell's conjugate-direction method ----------------
def powell_history(start, n_cycles=4):
    """
    Each "cycle" = a full pass through the current direction set (here always 2
    directions to start), followed by construction of the net-displacement
    direction, an extra line search along it, and (if it improves things) that
    direction replaces the one responsible for the largest single decrease.

    Returns a list of step dicts (one per individual line search) each tagged
    with a 'kind' so the animation can distinguish plain line searches from the
    "new conjugate direction" step, plus a 'directions_after' snapshot for
    displaying the current direction basis.
    """
    x = np.array(start, dtype=float)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    hist = []

    for cyc in range(n_cycles):
        x_cycle_start = x.copy()
        deltas = []
        for i, d in enumerate(directions):
            f_before = f_vec(x)
            x_new, t = exact_line_search(x, d)
            f_after = f_vec(x_new)
            deltas.append(f_before - f_after)
            hist.append(dict(x_start=x.copy(), x_end=x_new.copy(), direction=d.copy(),
                              label=f"d{i+1}", cycle=cyc, kind="basis",
                              directions_snapshot=[dd.copy() for dd in directions]))
            x = x_new

        new_dir = x - x_cycle_start
        norm = np.linalg.norm(new_dir)
        if norm > 1e-12:
            new_dir_unit = new_dir / norm
        else:
            new_dir_unit = new_dir

        x_before_ext = x.copy()
        x_new, t = exact_line_search(x, new_dir_unit if norm > 1e-12 else directions[0])
        hist.append(dict(x_start=x_before_ext.copy(), x_end=x_new.copy(),
                          direction=new_dir_unit.copy(), label="new direction",
                          cycle=cyc, kind="new_direction",
                          directions_snapshot=[dd.copy() for dd in directions]))
        x = x_new

        # standard heuristic: drop the direction that contributed the largest
        # single decrease, replace it with the newly built direction
        if norm > 1e-12:
            drop_idx = int(np.argmax(deltas))
            directions[drop_idx] = new_dir_unit

    return hist


if __name__ == "__main__":
    start = (-4.0, 4.2)
    hc = coordinate_descent_history(start, n_cycles=20)
    print("coord descent final f:", hc[-1]["f_end"], "after", len(hc), "line searches")
    for h in hc[:6]:
        print(h["cycle"], h["label"], h["x_end"], h["f_end"])

    hp = powell_history(start, n_cycles=4)
    print()
    print("powell final f:", hp[-1]["f_end"] if "f_end" in hp[-1] else f_vec(hp[-1]["x_end"]),
          "after", len(hp), "line searches")
    for h in hp:
        print(h["cycle"], h["kind"], h["label"], np.round(h["x_end"], 4), round(f_vec(h["x_end"]), 6))