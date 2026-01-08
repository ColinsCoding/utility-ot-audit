from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from odat2.models import CableRecord, ValidationIssue


@dataclass
class _Edge:
    u: str
    v: str
    cable_id: str
    length_m: float


class NetworkTopologyValidator:
    """Graph-theoretic checks over the whole dataset.

    Nodes = devices
    Edges = cables (undirected by default; set directed=False for physical cables)

    What it catches:
      - orphaned / disconnected components
      - single points of failure (articulation points)
      - loops (cycles) that may impact protection / comms design assumptions
    """

    def __init__(self):
        self._adj: Dict[str, List[Tuple[str, _Edge]]] = {}
        self._nodes: Set[str] = set()
        self._edges: List[_Edge] = []

    def build(self, records: List[CableRecord]) -> None:
        self._adj.clear()
        self._nodes.clear()
        self._edges.clear()

        for r in records:
            u = (r.from_device or "").strip()
            v = (r.to_device or "").strip()
            if not u or not v:
                continue

            e = _Edge(u=u, v=v, cable_id=r.cable_id, length_m=float(r.cable_length_m))
            self._edges.append(e)
            self._nodes.add(u)
            self._nodes.add(v)
            self._adj.setdefault(u, []).append((v, e))
            self._adj.setdefault(v, []).append((u, e))

    def _connected_components(self) -> List[Set[str]]:
        seen: Set[str] = set()
        comps: List[Set[str]] = []
        for n in self._nodes:
            if n in seen:
                continue
            stack = [n]
            comp = set()
            seen.add(n)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y, _e in self._adj.get(x, []):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            comps.append(comp)
        return comps

    def _find_articulation_points(self) -> Set[str]:
        # Tarjan articulation points on undirected graph
        time = 0
        disc: Dict[str, int] = {}
        low: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        ap: Set[str] = set()

        def dfs(u: str) -> None:
            nonlocal time
            time += 1
            disc[u] = low[u] = time
            children = 0
            for v, _e in self._adj.get(u, []):
                if v not in disc:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # Root with 2+ children
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    # Non-root where subtree can't reach ancestor
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for n in self._nodes:
            if n not in disc:
                parent[n] = None
                dfs(n)

        return ap

    def _count_cycles_upper_bound(self) -> int:
        # For undirected graph: cyclomatic number = E - N + C
        comps = self._connected_components()
        E = len(self._edges)
        N = len(self._nodes)
        C = len(comps)
        return max(0, E - N + C)

    def validate(self, records: List[CableRecord]) -> List[ValidationIssue]:
        self.build(records)
        issues: List[ValidationIssue] = []

        if not self._nodes:
            return issues

        comps = self._connected_components()
        if len(comps) > 1:
            main = max(comps, key=len)
            for comp in comps:
                if comp == main:
                    continue
                for node in sorted(comp):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="orphaned_device",
                            message=f"Device {node} is disconnected from the main network component.",
                            cable_id="N/A",
                            device_tag=node,
                            drawing_id="N/A",
                        )
                    )

        aps = self._find_articulation_points()
        for node in sorted(aps):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="single_point_of_failure",
                    message=f"Device {node} is an articulation point; its failure can disconnect part of the network.",
                    cable_id="N/A",
                    device_tag=node,
                    drawing_id="N/A",
                )
            )

        cycles = self._count_cycles_upper_bound()
        if cycles > 0:
            issues.append(
                ValidationIssue(
                    severity="info",
                    issue_type="cycles_detected",
                    message=f"Topology contains at least {cycles} independent cycle(s). "
                    "Loops can be good (redundancy) but may need explicit design/protection review.",
                    cable_id="N/A",
                    device_tag="N/A",
                    drawing_id="N/A",
                )
            )

        return issues
