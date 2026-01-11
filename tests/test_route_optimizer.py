import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

from odat2.telecom.route_optimizer import Layout, Rect, CostZone, astar, compute_routes
from odat2.telecom.dxf_writer import DXFPolyline, write_r12_dxf_polylines
from odat2.telecom.layout import load_layout
import json
import os
import math


def test_astar_avoids_obstacle():
    layout = Layout(
        width=10,
        height=6,
        obstacles=[Rect(4, 1, 6, 5)],  # block center
        cost_zones=[CostZone(0, 0, 10, 1, factor=2.0)],
        turn_penalty=0.5,
    )
    path, cost, turns = astar(layout, (1, 1), (8, 4))
    assert path is not None
    # should never enter obstacle
    for p in path:
        assert not layout.obstacles[0].contains(p)
    # basic sanity: starts and ends correct
    assert path[0] == (1, 1)
    assert path[-1] == (8, 4)
    assert cost < math.inf
    assert turns >= 0


def test_compute_routes_and_dxf(tmp_path):
    layout = Layout(width=8, height=5, obstacles=[], cost_zones=[], turn_penalty=0.0)
    endpoints = [{"route_id": "R1", "cable_type": "fiber", "from_x": 0, "from_y": 0, "to_x": 7, "to_y": 4}]
    res = compute_routes(layout, endpoints, grid_scale_m=1.0)
    assert len(res) == 1
    assert res[0].status == "ok"
    assert res[0].path is not None
    assert res[0].length > 0

    dxf_path = tmp_path / "routes.dxf"
    pts = [(float(x), float(y)) for (x, y) in res[0].path]
    write_r12_dxf_polylines([DXFPolyline(layer="ROUTE_FIBER", points=pts)], str(dxf_path))
    txt = dxf_path.read_text(encoding="utf-8")
    assert "LWPOLYLINE" in txt
    assert "ROUTE_FIBER" in txt
