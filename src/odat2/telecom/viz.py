# src/odat2/telecom/viz.py
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _layout_to_grid(layout) -> np.ndarray:
    """
    Convert Layout(width,height,obstacles[Rect...]) into a grid:
    0 = free, 1 = obstacle
    Assumes obstacle Rect has x0,y0,x1,y1 and is [x0,x1) [y0,y1).
    """
    grid = np.zeros((layout.height, layout.width), dtype=int)

    for rect in getattr(layout, "obstacles", []) or []:
        x0, y0, x1, y1 = int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)
        x0 = max(0, min(layout.width, x0))
        x1 = max(0, min(layout.width, x1))
        y0 = max(0, min(layout.height, y0))
        y1 = max(0, min(layout.height, y1))
        grid[y0:y1, x0:x1] = 1

    return grid


def plot_routes_preview(
    layout,
    routes: List[Tuple[str, List[Tuple[int, int]]]],  # (route_id, path)
    out_png: str,
    show_steps: bool = False,
) -> None:
    grid = _layout_to_grid(layout)

    plt.figure(figsize=(7, 5))
    plt.imshow(grid, cmap="gray_r", origin="upper")

    for route_id, path in routes:
        if not path:
            continue
        ys, xs = zip(*[(p[1], p[0]) for p in path])  # convert (x,y) -> (row=y, col=x)
        plt.plot(xs, ys, "-o", linewidth=2, markersize=2)
        if show_steps:
            for i, (x, y) in enumerate(path):
                plt.text(x, y, f"{i}", fontsize=6)

        # label route at start
        sx, sy = path[0]
        plt.text(sx, sy, route_id, fontsize=8)

    plt.title("A* Routes Preview (obstacles + paths)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
