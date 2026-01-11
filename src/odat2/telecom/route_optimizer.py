from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq
import math


Point = Tuple[int, int]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in grid coordinates, inclusive of min and exclusive of max."""
    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, p: Point) -> bool:
        x, y = p
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


@dataclass(frozen=True)
class CostZone(Rect):
    factor: float = 1.0


@dataclass
class Layout:
    width: int
    height: int
    obstacles: List[Rect]
    cost_zones: List[CostZone]
    turn_penalty: float = 0.25

    def in_bounds(self, p: Point) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, p: Point) -> bool:
        return self.in_bounds(p) and not any(r.contains(p) for r in self.obstacles)

    def zone_factor(self, p: Point) -> float:
        # Apply the max factor of all zones containing the point.
        f = 1.0
        for z in self.cost_zones:
            if z.contains(p):
                f = max(f, float(z.factor))
        return f


@dataclass
class RouteResult:
    route_id: str
    cable_type: str
    start: Point
    goal: Point
    path: Optional[List[Point]]
    cost: float
    length: float
    turns: int
    status: str
    message: str = ""


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors_4(p: Point) -> List[Point]:
    x, y = p
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _dir(a: Point, b: Point) -> Point:
    return (b[0] - a[0], b[1] - a[1])


def astar(
    layout: Layout,
    start: Point,
    goal: Point,
    *,
    distance_cost: float = 1.0,
) -> Tuple[Optional[List[Point]], float, int]:
    """
    Grid-based A* (4-neighborhood).
    Cost = distance_cost * zone_factor + turn_penalty when direction changes.
    Returns (path, cost, turns).
    """
    if not layout.passable(start):
        return None, math.inf, 0
    if not layout.passable(goal):
        return None, math.inf, 0
    if start == goal:
        return [start], 0.0, 0

    # State includes previous direction to penalize turns.
    # prev_dir is a (dx, dy) vector; (0,0) for start.
    start_state = (start, (0, 0))

    frontier: List[Tuple[float, int, Point, Point]] = []
    # heap item: (priority, tie, pos, prev_dir)
    tie = 0
    heapq.heappush(frontier, (0.0, tie, start, (0, 0)))

    came_from: Dict[Tuple[Point, Point], Tuple[Point, Point]] = {}
    cost_so_far: Dict[Tuple[Point, Point], float] = {start_state: 0.0}

    best_goal_state: Optional[Tuple[Point, Point]] = None
    best_goal_cost = math.inf

    while frontier:
        _, _, current, prev_dir = heapq.heappop(frontier)
        current_state = (current, prev_dir)

        # Early exit if we popped a goal state that is already optimal
        if current == goal and cost_so_far[current_state] <= best_goal_cost:
            best_goal_state = current_state
            best_goal_cost = cost_so_far[current_state]
            break

        for nxt in _neighbors_4(current):
            if not layout.passable(nxt):
                continue
            step_dir = _dir(current, nxt)
            step_cost = distance_cost * layout.zone_factor(nxt)

            turn_cost = 0.0
            if prev_dir != (0, 0) and step_dir != prev_dir:
                turn_cost = float(layout.turn_penalty)

            new_cost = cost_so_far[current_state] + step_cost + turn_cost
            nxt_state = (nxt, step_dir)

            if new_cost < cost_so_far.get(nxt_state, math.inf):
                cost_so_far[nxt_state] = new_cost
                came_from[nxt_state] = current_state
                priority = new_cost + manhattan(nxt, goal)
                tie += 1
                heapq.heappush(frontier, (priority, tie, nxt, step_dir))

    if best_goal_state is None:
        return None, math.inf, 0

    # Reconstruct path from best_goal_state
    path: List[Point] = []
    s = best_goal_state
    while True:
        path.append(s[0])
        if s == start_state:
            break
        s = came_from[s]
    path.reverse()

    turns = 0
    prev = None
    prev_d = (0, 0)
    for i in range(1, len(path)):
        d = _dir(path[i - 1], path[i])
        if prev is not None and d != prev_d:
            turns += 1
        prev_d = d
        prev = path[i]

    return path, float(best_goal_cost), int(turns)


def polyline_length(path: List[Point], *, scale: float = 1.0) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        dx = (b[0] - a[0]) * scale
        dy = (b[1] - a[1]) * scale
        total += math.hypot(dx, dy)
    return total


def compute_routes(
    layout: Layout,
    endpoints: Iterable[dict],
    *,
    grid_scale_m: float = 1.0,
) -> List[RouteResult]:
    results: List[RouteResult] = []
    for row in endpoints:
        route_id = str(row.get("route_id") or row.get("id") or f"route_{len(results)+1}")
        cable_type = str(row.get("cable_type") or "unknown")

        sx, sy = int(row["from_x"]), int(row["from_y"])
        gx, gy = int(row["to_x"]), int(row["to_y"])
        start = (sx, sy)
        goal = (gx, gy)

        path, cost, turns = astar(layout, start, goal)
        if path is None:
            results.append(RouteResult(
                route_id=route_id,
                cable_type=cable_type,
                start=start,
                goal=goal,
                path=None,
                cost=math.inf,
                length=0.0,
                turns=0,
                status="no_path",
                message="No feasible route found (blocked or disconnected).",
            ))
            continue

        length = polyline_length(path, scale=grid_scale_m)
        results.append(RouteResult(
            route_id=route_id,
            cable_type=cable_type,
            start=start,
            goal=goal,
            path=path,
            cost=float(cost),
            length=float(length),
            turns=int(turns),
            status="ok",
            message="",
        ))
    return results
