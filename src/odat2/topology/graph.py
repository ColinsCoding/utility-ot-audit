from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import heapq


@dataclass(frozen=True)
class Edge:
    """Undirected physical connection between two devices."""
    u: str
    v: str
    cable_id: str
    length_m: float = 0.0


class UnionFind:
    """Disjoint Set Union (Union-Find) for fast connectivity queries.

    Operations are nearly O(1) amortized with path compression + union by rank.
    """
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        # iterative path compression
        if x not in self.parent:
            self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # compress
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class Graph:
    """Adjacency-list graph specialized for OT/telecom topology audits.

    Nodes: device tags
    Edges: cables (undirected)
    """

    def __init__(self) -> None:
        self._adj: Dict[str, List[Tuple[str, Edge]]] = {}
        self._nodes: Set[str] = set()
        self._edges: List[Edge] = []

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    def add_edge(self, edge: Edge) -> None:
        u, v = edge.u.strip(), edge.v.strip()
        if not u or not v:
            return
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.append(edge)
        self._adj.setdefault(u, []).append((v, edge))
        self._adj.setdefault(v, []).append((u, edge))

    @classmethod
    def from_edges(cls, edges: Iterable[Edge]) -> "Graph":
        g = cls()
        for e in edges:
            g.add_edge(e)
        return g

    def connected_components_union_find(self) -> List[Set[str]]:
        """Connected components using Union-Find (DSA highlight)."""
        uf = UnionFind()
        for n in self._nodes:
            uf.add(n)
        for e in self._edges:
            uf.union(e.u, e.v)
        groups: Dict[str, Set[str]] = {}
        for n in self._nodes:
            r = uf.find(n)
            groups.setdefault(r, set()).add(n)
        return list(groups.values())

    def cyclomatic_number(self) -> int:
        """Upper bound on independent cycles: E - N + C."""
        comps = self.connected_components_union_find()
        E = len(self._edges)
        N = len(self._nodes)
        C = len(comps)
        return max(0, E - N + C)

    def articulation_points(self) -> Set[str]:
        """Tarjan articulation points on undirected graphs (single points of failure)."""
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

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for n in self._nodes:
            if n not in disc:
                parent[n] = None
                dfs(n)
        return ap

    def shortest_path_dijkstra(self, src: str, dst: str) -> Optional[Tuple[float, List[str]]]:
        """Weighted shortest path by cable length.

        Returns (distance, path nodes) or None if disconnected.
        """
        src, dst = src.strip(), dst.strip()
        if src not in self._nodes or dst not in self._nodes:
            return None
        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Optional[str]] = {src: None}
        pq: List[Tuple[float, str]] = [(0.0, src)]
        seen: Set[str] = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                break
            for v, e in self._adj.get(u, []):
                w = float(e.length_m) if e.length_m is not None else 0.0
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dst not in dist:
            return None
        # reconstruct
        path = []
        cur: Optional[str] = dst
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return dist[dst], path
