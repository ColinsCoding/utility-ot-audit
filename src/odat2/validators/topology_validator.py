from __future__ import annotations

from typing import List

from odat2.models import CableRecord, ValidationIssue
from odat2.topology import Edge, Graph


class NetworkTopologyValidator:
    """Graph-theoretic checks over the whole dataset.

    DSA highlights:
      - Union-Find (Disjoint Set Union) for connected components
      - Tarjan articulation points for single points of failure
      - Cyclomatic number (E - N + C) for cycle upper bound
    """

    def __init__(self) -> None:
        self._graph = Graph()

    def build(self, records: List[CableRecord]) -> None:
        self._graph = Graph()
        for r in records:
            u = (r.from_device or "").strip()
            v = (r.to_device or "").strip()
            if not u or not v:
                continue
            self._graph.add_edge(
                Edge(u=u, v=v, cable_id=r.cable_id, length_m=float(r.cable_length_m))
            )

    def validate(self, records: List[CableRecord]) -> List[ValidationIssue]:
        self.build(records)
        issues: List[ValidationIssue] = []

        if not self._graph.nodes:
            return issues

        comps = self._graph.connected_components_union_find()
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

        for node in sorted(self._graph.articulation_points()):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="single_point_of_failure",
                    message=(
                        f"Device {node} is an articulation point; its failure can disconnect part of the network."
                    ),
                    cable_id="N/A",
                    device_tag=node,
                    drawing_id="N/A",
                )
            )

        cycles = self._graph.cyclomatic_number()
        if cycles > 0:
            issues.append(
                ValidationIssue(
                    severity="info",
                    issue_type="cycles_detected",
                    message=(
                        f"Topology contains at least {cycles} independent cycle(s). "
                        "Loops can be good (redundancy) but may need explicit design/protection review."
                    ),
                    cable_id="N/A",
                    device_tag="N/A",
                    drawing_id="N/A",
                )
            )

        return issues
