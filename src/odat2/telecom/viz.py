import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


def plot_coverage(
    layout,
    cov,
    sensors,
    out_png,
    show_blind_spots=True,
    title="Coverage Heat Map"
):
    """
    Produce a polished heat-map style coverage visualization.
    """

    # Obstacle mask
    obs = np.zeros((layout.height, layout.width))
    for r in getattr(layout, "obstacles", []) or []:
        obs[int(r.y0):int(r.y1), int(r.x0):int(r.x1)] = 1

    fig, ax = plt.subplots(figsize=(9, 6))

    # Obstacles (background)
    ax.imshow(
        obs,
        cmap="gray",
        alpha=0.25,
        origin="lower"
    )

    # Coverage heat map
    cov_safe = np.where(cov > 0, cov, np.nan)

    im = ax.imshow(
        cov_safe,
        origin="lower",
        cmap="inferno",
        alpha=0.9,
        norm=colors.LogNorm(
            vmin=max(1e-2, np.nanmin(cov_safe)),
            vmax=np.nanmax(cov_safe)
        )
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coverage Intensity", rotation=90)

    # Sensors
    if sensors:
        xs = [s.x for s in sensors]
        ys = [s.y for s in sensors]
        ax.scatter(
            xs,
            ys,
            s=80,
            c="cyan",
            edgecolors="black",
            linewidths=0.8,
            label="Sensors",
            zorder=3
        )

    # Blind spots
    if show_blind_spots:
        by, bx = np.where((cov <= 0) & (obs == 0))
        if len(bx):
            ax.scatter(
                bx,
                by,
                s=6,
                c="white",
                alpha=0.25,
                marker="x",
                label="Blind spots",
                zorder=2
            )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
