from odat2.topology import Edge, Graph


def test_union_find_components_and_articulation_points():
    # A-B-C chain plus D-E separate component
    edges = [
        Edge("A","B","c1",1.0),
        Edge("B","C","c2",1.0),
        Edge("D","E","c3",1.0),
    ]
    g = Graph.from_edges(edges)
    comps = [set(c) for c in g.connected_components_union_find()]
    assert any(c == {"A","B","C"} for c in comps)
    assert any(c == {"D","E"} for c in comps)

    aps = g.articulation_points()
    assert "B" in aps
    assert "A" not in aps
    assert "C" not in aps

    # cyclomatic number should be 0 (no cycles)
    assert g.cyclomatic_number() == 0


def test_dijkstra_shortest_path():
    edges = [
        Edge("A","B","c1",5.0),
        Edge("B","C","c2",2.0),
        Edge("A","C","c3",10.0),
    ]
    g = Graph.from_edges(edges)
    res = g.shortest_path_dijkstra("A","C")
    assert res is not None
    dist, path = res
    assert dist == 7.0
    assert path == ["A","B","C"]
