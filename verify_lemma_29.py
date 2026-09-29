#!/usr/bin/env python3
"""Exact exhaustive verification of Lemma 2.9 in the Li--Zhang manuscript.

Python 3.10+; standard library only.
Run: python verify_lemma_29.py --out results

The computation tests ALL connected simple unicyclic graphs of orders 3--10.
Both the upper bound and the complete equality classification are checked.
The five expected equality graphs are specified from Figure 2, not inferred
from the computed maximum. No expected graph counts or maxima are hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

Graph = tuple[tuple[int, ...], ...]
Code = tuple[str, ...]


def make_graph(n: int, edges) -> Graph:
    """Build a simple graph with vertex set {0,...,n-1}."""
    if n < 0:
        raise ValueError("The order must be nonnegative.")
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("Invalid edge.")
        adj[u].add(v)
        adj[v].add(u)
    return tuple(tuple(sorted(neighbors)) for neighbors in adj)


def cycle_graph(n: int) -> Graph:
    if n < 3:
        raise ValueError("A simple cycle has at least three vertices.")
    return make_graph(n, ((i, (i + 1) % n) for i in range(n)))


def add_leaf(adj: Graph, v: int) -> Graph:
    n = len(adj)
    if not 0 <= v < n:
        raise ValueError("Invalid attachment vertex.")
    return adj[:v] + (adj[v] + (n,),) + adj[v + 1:] + ((v,),)


def canonical_code(adj: Graph) -> Code:
    """Exact canonical code for a connected simple unicyclic graph.

    The input is an undirected adjacency tuple on {0,...,n-1}.
    No special labeling of the cycle vertices is required.
    """
    n = len(adj)
    if n < 3 or sum(map(len, adj)) != 2 * n:
        raise ValueError("Expected a unicyclic graph.")
    reached, todo = {0}, [0]
    while todo:
        v = todo.pop()
        for w in adj[v]:
            if w not in reached:
                reached.add(w)
                todo.append(w)
    if len(reached) != n:
        raise ValueError("Expected a connected graph.")

    # Repeatedly delete leaves. The remaining vertices form the unique cycle.
    degree = list(map(len, adj))
    queue = deque(v for v in range(n) if degree[v] == 1)
    while queue:
        v = queue.popleft()
        degree[v] = 0
        for w in adj[v]:
            if degree[w] > 0:
                degree[w] -= 1
                if degree[w] == 1:
                    queue.append(w)
    core = {v for v in range(n) if degree[v] > 0}
    if len(core) < 3 or any(degree[v] != 2 for v in core):
        raise ValueError("The remaining core is not a simple cycle.")

    start = min(core)
    cycle = [start]
    previous, current = start, min(w for w in adj[start] if w in core)
    while current != start:
        cycle.append(current)
        nxt = next(w for w in adj[current] if w in core and w != previous)
        previous, current = current, nxt

    def rooted_code(v: int, parent: int) -> str:
        children = [rooted_code(w, v) for w in adj[v]
                    if w != parent and w not in core]
        return "(" + "".join(sorted(children)) + ")"

    word = tuple(rooted_code(v, -1) for v in cycle)
    return min(w[i:] + w[:i]
               for w in (word, word[::-1]) for i in range(len(word)))


def unicyclic_levels(max_n: int = 10):
    """Yield one representative per unicyclic isomorphism class, by order."""
    previous = {}
    for n in range(3, max_n + 1):
        cycle = cycle_graph(n)
        current = {canonical_code(cycle): cycle}
        for adj in previous.values():
            for v in range(n - 1):
                graph = add_leaf(adj, v)
                current.setdefault(canonical_code(graph), graph)
        yield n, current
        previous = current


def maximal_masks(adj: Graph) -> list[int]:
    """List all maximal dissociation sets as vertex bitmasks.

    good[S] means that the selected vertices induce maximum degree <= 1.
    A good subset is maximal iff every one-vertex extension is not good.
    """
    n = len(adj)
    neighborhood = [sum(1 << w for w in adj[v]) for v in range(n)]
    good = bytearray(1 << n)
    for chosen in range(1 << n):
        good[chosen] = all((neighborhood[v] & chosen).bit_count() <= 1
                           for v in range(n) if chosen & (1 << v))
    return [chosen for chosen in range(1 << n)
            if good[chosen] and all(not good[chosen | (1 << v)]
                                   for v in range(n)
                                   if not chosen & (1 << v))]


def phi(adj: Graph) -> int:
    return len(maximal_masks(adj))


def compare_h(value: int, n: int) -> int:
    """Return -1, 0, or 1 according as value is <, =, or > h(n)."""
    if n in (3, 4):
        delta = value - {3: 3, 4: 6}[n]
    elif n >= 5:
        delta = (value - n + 5) ** 3 - 8 * 3 ** (n - 2)
    else:
        raise ValueError("h(n) is defined here only for n >= 3.")
    return (delta > 0) - (delta < 0)


def h_display(n: int) -> float:
    """Display only: this floating-point value never decides a test."""
    return float({3: 3, 4: 6}[n]) if n < 5 else 2 * 3 ** ((n - 2) / 3) + n - 5


def equality_templates(n: int) -> dict[Code, str]:
    """The equality family shown in Figure 2, restricted to 3 <= n <= 10."""
    templates = []
    if n == 3:
        templates = [("C3", cycle_graph(3))]
    elif n == 4:
        templates = [("C4", cycle_graph(4))]
    elif n == 5:
        triangle = add_leaf(add_leaf(cycle_graph(3), 0), 1)
        square = add_leaf(cycle_graph(4), 0)
        templates = [("C3 with leaves at two distinct cycle vertices", triangle),
                     ("C4 with one leaf", square)]
    elif n == 8:
        graph = make_graph(8, [(0, 1), (1, 2), (2, 3), (3, 0),
                               (0, 4), (4, 5), (5, 6), (5, 7)])
        templates = [("G8_star", graph)]
    return {canonical_code(graph): name for name, graph in templates}


def verify(out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, records, equalities, violations = [], [], [], []
    started = perf_counter()
    print("n,graphs,max_phi,maximizers,h_approx,equality_graphs,bound_OK,equality_OK")
    for n, graphs in unicyclic_levels(10):
        best, maximizing = -1, 0
        observed = set()
        bound_ok = True
        expected = equality_templates(n)
        for code, adj in sorted(graphs.items()):
            sets = maximal_masks(adj)
            value = len(sets)
            relation = compare_h(value, n)
            record = dict(n=n, code=code, adjacency=adj, phi=value,
                          comparison_with_h=relation)
            records.append(record)
            if value > best:
                best, maximizing = value, 1
            elif value == best:
                maximizing += 1
            if relation > 0:
                bound_ok = False
                violations.append(dict(kind="upper_bound", **record))
            if relation == 0:
                observed.add(code)
                equalities.append(dict(
                    **record, name=expected.get(code, "UNEXPECTED_EQUALITY"),
                    maximal_dissociation_sets=[
                        [v for v in range(n) if mask & (1 << v)] for mask in sets]))
        equality_ok = observed == set(expected)
        if not equality_ok:
            violations.append(dict(kind="equality_classification", n=n,
                                   missing=sorted(set(expected) - observed),
                                   unexpected=sorted(observed - set(expected))))
        row = dict(n=n, graphs=len(graphs), max_phi=best, maximizers=maximizing,
                   h_approx=h_display(n), equality_graphs=len(observed),
                   bound_OK=bound_ok, equality_OK=equality_ok)
        rows.append(row)
        print(f"{n},{len(graphs)},{best},{maximizing},{h_display(n):.9f},"
              f"{len(observed)},{bound_ok},{equality_ok}", flush=True)

    passed = not violations
    metadata = dict(python=sys.version, platform=platform.platform(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    seconds=round(perf_counter() - started, 6),
                    total_graphs=len(records), total_equality_graphs=len(equalities),
                    passed=passed, rows=rows)
    for name, data in (("summary.json", metadata), ("all_graphs.json", records),
                       ("equality_graphs.json", equalities), ("violations.json", violations)):
        (out_dir / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'PASS' if passed else 'FAIL'}: {len(records)} graphs; "
          f"{len(equalities)} equality graphs; {len(violations)} violations.")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("lemma_29_results"))
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.out) else 1)
