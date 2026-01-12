import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np


def plot_coverage(layout, cov, sensors, out_png, show_blind_spots=True,
                  *,
                  title=None,
                  cmap="inferno",
                  log_scale=True,
                  percentile_clip=(1.0, 99.5),
                  show_grid=False,
                  origin="upper",
                  obstacle_alpha=0.30,
                  coverage_alpha=0.92):
    """Render a coverage heat map with obstacles, sensors, and blind spots.

    Backwards-compatible with the original signature:
      plot_coverage(layout, cov, sensors, out_png, show_blind_spots=True)

    Extra keyword-only options let you tune appearance without changing callers.
    """

    cov = np.asarray(cov)

    # Build obstacle mask and keep obstacle rectangles for crisp rendering
    obs = np.zeros((layout.height, layout.width), dtype=np.uint8)
    obstacle_rects = []
    for r in getattr(layout, "obstacles", []) or []:
        x0, x1 = int(r.x0), int(r.x1)
        y0, y1 = int(r.y0), int(r.y1)
        if x1 <= x0 or y1 <= y0:
            continue
        obs[y0:y1, x0:x1] = 1
        obstacle_rects.append((x0, y0, x1 - x0, y1 - y0))

    # Determine robust normalization for coverage
    cov_free = np.where(obs == 0, cov, np.nan)
    finite = np.isfinite(cov_free)

    norm = None
    if finite.any():
        lo_p, hi_p = percentile_clip
        lo = float(np.nanpercentile(cov_free, lo_p))
        hi = float(np.nanpercentile(cov_free, hi_p))
        if not np.isfinite(lo):
            lo = float(np.nanmin(cov_free))
        if not np.isfinite(hi):
            hi = float(np.nanmax(cov_free))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

        if log_scale and (np.nanmin(cov_free) > 0):
            vmin = max(lo, np.nanmin(cov_free[cov_free > 0]))
            vmax = hi
            if vmax <= vmin:
                vmax = vmin * 10
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=lo, vmax=hi)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Obstacles: draw as rectangles for fidelity
    if obstacle_rects:
        for (x, y, w, h) in obstacle_rects:
            ax.add_patch(
                Rectangle(
                    (x, y), w, h,
                    facecolor=(0, 0, 0, obstacle_alpha),
                    edgecolor=(0, 0, 0, min(1.0, obstacle_alpha + 0.35)),
                    linewidth=0.8,
                    zorder=2,
                )
            )

    # Coverage heatmap (masked where obstacles exist)
    cov_img = np.where(obs == 0, cov, np.nan)
    im = ax.imshow(
        cov_img,
        cmap=cmap,
        norm=norm,
        origin=origin,
        interpolation="nearest",
        alpha=coverage_alpha,
        zorder=1,
    )

    # Sensors
    xs = [s.x for s in sensors] if sensors else []
    ys = [s.y for s in sensors] if sensors else []
    sensor_sc = None
    if xs:
        sensor_sc = ax.scatter(
            xs, ys,
            s=70,
            marker="o",
            facecolors="cyan",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    # Blind spots (free-space cells with non-positive coverage)
    blind_sc = None
    blind_count = 0
    if show_blind_spots:
        by, bx = np.where((obs == 0) & (cov <= 0))
        blind_count = int(len(bx))
        if blind_count:
            blind_sc = ax.scatter(
                bx, by,
                s=10,
                marker="x",
                linewidths=0.8,
                c="white",
                alpha=0.55,
                zorder=3,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title is None:
        title = "Coverage" + (" (log scale)" if isinstance(norm, mcolors.LogNorm) else "")
    ax.set_title(title)

    if show_grid:
        ax.set_xticks(np.arange(-0.5, layout.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, layout.height, 1), minor=True)
        ax.grid(which="minor", linewidth=0.25, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    if finite.any():
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Coverage intensity")

    handles = []
    labels = []
    if obstacle_rects:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=(0, 0, 0, obstacle_alpha), edgecolor="black"))
        labels.append("Obstacle")
    if sensor_sc is not None:
        handles.append(sensor_sc)
        labels.append("Sensor")
    if blind_sc is not None:
        handles.append(blind_sc)
        labels.append("Blind spot")
    if handles:
        ax.legend(handles, labels, loc="upper right", framealpha=0.90)

    free_cells = int(np.sum(obs == 0))
    covered_cells = int(np.sum((obs == 0) & (cov > 0)))
    covered_pct = (covered_cells / free_cells * 100.0) if free_cells else 0.0
    stats = (
        f"Free cells: {free_cells:,}\n"
        f"Covered: {covered_cells:,} ({covered_pct:.1f}%)\n"
        f"Blind spots: {blind_count:,}\n"
        f"Sensors: {len(xs):,}"
    )
    ax.text(
        0.01, 0.01, stats,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.45, edgecolor="none"),
        color="white",
        zorder=5,
    )

    fig.savefig(out_png, dpi=240)
    plt.close(fig)
