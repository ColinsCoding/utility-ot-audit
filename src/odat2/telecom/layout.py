from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json

from .route_optimizer import Rect, CostZone, Layout


def load_layout(path: str) -> Layout:
    """
    Layout JSON schema (minimal):

    {
      "width": 40,
      "height": 25,
      "turn_penalty": 0.25,
      "obstacles": [{"x0": 10, "y0": 5, "x1": 15, "y1": 12}],
      "cost_zones": [{"x0": 0, "y0": 0, "x1": 40, "y1": 1, "factor": 1.5}]
    }
    """
    data = json.loads(open(path, "r", encoding="utf-8").read())
    width = int(data["width"])
    height = int(data["height"])
    turn_penalty = float(data.get("turn_penalty", 0.25))

    obstacles: List[Rect] = []
    for r in data.get("obstacles", []):
        obstacles.append(Rect(int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])))

    cost_zones: List[CostZone] = []
    for z in data.get("cost_zones", []):
        cost_zones.append(CostZone(
            int(z["x0"]), int(z["y0"]), int(z["x1"]), int(z["y1"]), factor=float(z.get("factor", 1.0))
        ))

    return Layout(width=width, height=height, obstacles=obstacles, cost_zones=cost_zones, turn_penalty=turn_penalty)
