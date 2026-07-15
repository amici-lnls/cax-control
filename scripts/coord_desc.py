import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# import style
from optimizers2d import f, CENTER, coordinate_descent_history, f_vec

START = (-4.0, 4.2)
N_CYCLES = 25
hist = coordinate_descent_history(START, n_cycles=N_CYCLES)
F_TRUE = 0.0

DOM = (-6, 6)
xx = np.linspace(*DOM, 400)
yy = np.linspace(*DOM, 400)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)

HOLD_FRAMES = 20
n_frames = len(hist) + HOLD_FRAMES

fig, ax = plt.subplots(figsize=(7.6, 7.2))
fig.subplots_adjust(top=0.90, bottom=0.10, left=0.11, right=0.98)

levels = np.linspace(0, 10, 30)
ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="max")
ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.12, linewidths=0.5)
ax.plot(*CENTER, marker="*", color="white", ms=18, mec="black", mew=0.8, zorder=6)
ax.set_xlim(*DOM)
ax.set_ylim(*DOM)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title("", fontsize=13, pad=8)

COLOR_X = "red"#style.CURRENT
COLOR_Y = "blue"#style.C_COLOR

path_lines = []  # persistent Line2D segments already drawn
# current_marker = ax.scatter([], [], s=90, color=style.BEST, edgecolor="black",
current_marker = ax.scatter([], [], s=90, color="orange", edgecolor="black",
                             linewidth=0.8, zorder=7)

legend_handles = [
    plt.Line2D([0], [0], color=COLOR_X, lw=2.5, label="line search along x"),
    plt.Line2D([0], [0], color=COLOR_Y, lw=2.5, label="line search along y"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85, fontsize=8.5)

info_text = ax.text(0.015, 0.02, "", transform=ax.transAxes, fontsize=9,
                     va="bottom", ha="left", color="white",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#1b1e29cc", ec="#3a3f4b"))


def update(frame):
    i = min(frame, len(hist) - 1)
    step = hist[i]

    if len(path_lines) <= i:
        color = COLOR_X if step["label"] == "x-direction" else COLOR_Y
        (ln,) = ax.plot([step["x_start"][0], step["x_end"][0]],
                         [step["x_start"][1], step["x_end"][1]],
                         color=color, lw=2.2, zorder=5, solid_capstyle="round")
        path_lines.append(ln)

    current_marker.set_offsets([step["x_end"]])

    err = abs(step["f_end"] - F_TRUE)
    title.set_text(f"Coordinate Descent — cycle {step['cycle']+1}, {step['label']}   "
                    f"(line search {i+1} / {len(hist)})")
    info_text.set_text(
        f"point: ({step['x_end'][0]:+.3f}, {step['x_end'][1]:+.3f})\n"
        f"f = {step['f_end']:.5f}   |f - f*| = {err:.2e}"
    )
    return path_lines + [current_marker, title, info_text]


anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=180)
# writer = FFMpegWriter(fps=7, bitrate=3000)
anim.save("coordinate_descent_2d.gif", writer="pillow", dpi=140)
print("saved coordinate_descent_2d.mp4")