#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""穷举验证 Li--Zhang 稿件的 Lemmas 2.10 和 2.11。

仅使用 Python 标准库。建议 Python 3.10 或更高版本。
默认验证全部原文范围，不采用附录额外的支撑点限制，也不使用归纳假设。

运行：
  python verify_lemmas_210_211.py --self-test
  python verify_lemmas_210_211.py --out results
  python verify_lemmas_210_211.py --max-n 13 --out small_results

输出：summary.csv、results.json、witnesses.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

Graph = tuple[tuple[int, ...], ...]
Word = tuple[str, ...]


def cycle_graph(c: int) -> Graph:
    if c < 3:
        raise ValueError("简单圈的阶数必须至少为 3。")
    return tuple(tuple(sorted(((i - 1) % c, (i + 1) % c)))
                 for i in range(c))


def add_leaf(adj: Graph, v: int) -> Graph:
    """新顶点编号为 len(adj)，只与 v 相邻。"""
    n = len(adj)
    return adj[:v] + (adj[v] + (n,),) + adj[v + 1:] + ((v,),)


def canonical_word(adj: Graph, c: int) -> Word:
    """前 c 个顶点按 0,1,...,c-1 构成唯一圈。

    根树用排序括号编码；圈上编码再对全部旋转和反向取最小值。
    这是精确的同构规范编码，不是可能碰撞的图哈希。
    """
    def tree_code(v: int, parent: int) -> str:
        children = [tree_code(w, v) for w in adj[v]
                    if w != parent and w >= c]
        return "(" + "".join(sorted(children)) + ")"

    word = tuple(tree_code(v, -1) for v in range(c))
    rev = word[::-1]
    return min(w[i:] + w[:i] for w in (word, rev) for i in range(c))


def unicyclic_levels(max_order: int):
    """依次生成各阶全部非同构连通单圈图。

    每阶加入纯圈；其余图由上一阶的图添加一个叶子得到。
    字典的值为 (邻接表, 圈长)。只保留相邻两阶的图。
    """
    previous = {}
    for n in range(3, max_order + 1):
        cycle = cycle_graph(n)
        current = {canonical_word(cycle, n): (cycle, n)}
        for adj, c in previous.values():
            for v in range(n - 1):
                graph = add_leaf(adj, v)
                key = canonical_word(graph, c)
                if key not in current:
                    current[key] = (graph, c)
        yield n, current
        previous = current


def attach_fork(adj: Graph, root: int) -> tuple[Graph, int]:
    """添加 root--x2--x1，其中 x1 邻接两个新叶子 x0,y。

    新顶点依次为 x2=m, x1=m+1, x0=m+2, y=m+3。
    返回 (新图, x0)。
    """
    m = len(adj)
    graph = add_leaf(adj, root)
    graph = add_leaf(graph, m)
    graph = add_leaf(graph, m + 1)
    graph = add_leaf(graph, m + 1)
    return graph, m + 2


def optimal_path(adj: Graph, c: int) -> tuple[int, tuple[int, ...], list[int]]:
    """严格实现原文定义：先最大距离，再最大度数字典序。

    返回 (到圈最大距离, 最优度序列, 从叶子到圈的最优路径)。
    所有并列最优路径的度序列相同，因此任取其中一条即可。
    """
    n = len(adj)
    distance = [-1] * n
    parent = [-1] * n
    queue = deque(range(c))
    for v in range(c):
        distance[v] = 0
    while queue:
        v = queue.popleft()
        for w in adj[v]:
            if distance[w] == -1:
                distance[w] = distance[v] + 1
                parent[w] = v
                queue.append(w)
    if min(distance) < 0:
        raise ValueError("输入图不连通。")
    d = max(distance)
    best = None
    best_path = []
    for v in range(n):
        if distance[v] != d:
            continue
        path = [v]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        degrees = tuple(len(adj[w]) for w in path[1:])
        if best is None or degrees > best:
            best = degrees
            best_path = path
    return d, best if best is not None else (), best_path


def child_codes(code: str):
    """把一个根树的括号编码分解为各个子树的编码。"""
    start = 1
    depth = 0
    for i in range(1, len(code) - 1):
        depth += 1 if code[i] == "(" else -1
        if depth == 0:
            yield code[start:i + 1]
            start = i + 1


@lru_cache(maxsize=None)
def tree_summary(code: str) -> tuple[int, int, int, int, int, int]:
    """根树六个中间计数 (T,A,B,I,P,M)，全部为整数。

    顶点状态：0=未选；1=已选且孤立；2=已选且恰有一个已选邻点。
    根为0时：T=子树所有可行组合；A=所有孩子为0；
             B=恰有一个孩子为1，其余为0。
    根为1时：I=所有孩子为0的组合数。
    根为2时：P=所有孩子为0；M=恰有一个孩子为2，其余为0。
    除根可能还需父点/圈上邻点满足条件外，所有后代均满足条件。
    """
    T = A = I = P = 1
    B = M = 0
    for child in child_codes(code):
        t, a, b, iso, pair0, pair1 = tree_summary(child)
        # 根为0时，孩子状态为0/1/2的权重。
        w0, w1, w2 = t - a - b, iso, pair1
        T *= w0 + w1 + w2
        B, A = B * w0 + A * w1, A * w0
        # 根为1时，孩子必须为0且要已被阻挡。
        I *= t - a
        # 根为2时，孩子为0的权重=t，为2的权重=pair0。
        M, P = M * t + P * pair0, P * t
    return T, A, B, I, P, M


@lru_cache(maxsize=None)
def local_table(code: str) -> tuple[int, ...]:
    """W(left,center,right)：一个圈顶点及其挂树的局部计数。

    表下标为 9*left + 3*center + right。
    """
    T, A, B, I, P, M = tree_summary(code)
    table = []
    for left in range(3):
        for center in range(3):
            for right in range(3):
                if center == 0:
                    if left == 2 or right == 2 or (left == right == 1):
                        value = T
                    elif left == 1 or right == 1:
                        value = T - A
                    else:
                        value = T - A - B
                elif center == 1:
                    value = I if left == right == 0 else 0
                else:
                    if left == 1 or right == 1 or (left == right == 2):
                        value = 0
                    elif left == 2 or right == 2:
                        value = P
                    else:
                        value = M
                table.append(value)
    return tuple(table)


@lru_cache(maxsize=None)
def transitions(code: str):
    table = local_table(code)
    return tuple(tuple((3 * s + t, table[9 * p + 3 * s + t])
                       for t in range(3) if table[9 * p + 3 * s + t])
                 for p in range(3) for s in range(3))


def phi(word: Word) -> int:
    """沿唯一圈做三状态动态规划，精确计算极大 dissociation sets 数量。"""
    c = len(word)
    if c < 3:
        raise ValueError("至少需要三个圈顶点。")
    first, last = local_table(word[0]), local_table(word[-1])
    middle = [transitions(code) for code in word[1:-1]]
    total = 0
    # 相邻两点唯一可能的状态对。
    for a, b in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 2)):
        dp = [0] * 9
        dp[3 * a + b] = 1
        for trans in middle:
            following = [0] * 9
            for pair, count in enumerate(dp):
                if count:
                    for new_pair, weight in trans[pair]:
                        following[new_pair] += count * weight
            dp = following
        for pair, count in enumerate(dp):
            if count:
                p, s = divmod(pair, 3)
                total += (count * last[9 * p + 3 * s + a]
                          * first[9 * s + 3 * a + b])
    return total


def less_than_h(value: int, n: int) -> bool:
    """精确判断 value < 2*3**((n-2)/3) + n-5，不使用浮点数。"""
    if n < 5:
        return value < {3: 3, 4: 6}[n]
    return (value - n + 5) ** 3 < 8 * 3 ** (n - 2)


def h_display(n: int) -> float:
    """仅供输出，不参与 PASS/FAIL 的判断。"""
    return 2 * 3 ** ((n - 2) / 3) + n - 5


def brute_phi(adj: Graph) -> int:
    """独立的 2**n 子集枚举；只用于交叉测试及最大值见证核对。"""
    n = len(adj)
    masks = [sum(1 << w for w in neighbors) for neighbors in adj]
    good = bytearray(1 << n)
    for chosen in range(1 << n):
        good[chosen] = all((masks[v] & chosen).bit_count() <= 1
                           for v in range(n) if chosen & (1 << v))
    return sum(1 for chosen in range(1 << n)
               if good[chosen] and all(not good[chosen | (1 << v)]
                                      for v in range(n)
                                      if not chosen & (1 << v)))


def self_test() -> None:
    """用独立子集枚举交叉核对所有 3--9 阶单圈图的 DP 计数。"""
    tested = 0
    for n, graphs in unicyclic_levels(9):
        for word, (adj, c) in graphs.items():
            exact = brute_phi(adj)
            if phi(word) != exact:
                raise AssertionError((n, word, phi(word), exact))
            tested += 1
        print(f"SELF-TEST n={n}: {len(graphs)} graphs, DP=brute force", flush=True)
    if not less_than_h(59, 11) or less_than_h(60, 11):
        raise AssertionError("严格不等式比较器未通过测试。")
    print(f"SELF-TEST PASS: {tested} graphs", flush=True)


def verify(max_n: int, out_dir: Path) -> list[dict]:
    if not 11 <= max_n <= 18:
        raise ValueError("max_n 必须在 11 到 18 之间。")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, witnesses = [], []
    start = time.perf_counter()
    print("lemma,n,cores,root_candidates,graphs,max_phi,h(n),PASS", flush=True)
    for m, cores in unicyclic_levels(max_n - 4):
        if m < 7:
            continue
        n = m + 4
        active = ("2.10", "2.11") if n <= 16 else ("2.11",)
        seen = {lemma: set() for lemma in active}
        stats = {lemma: dict(root_candidates=0, graphs=0, max_phi=-1,
                            maximizing_graphs=0, violations=0)
                 for lemma in active}
        best_witness = {}
        for core, c in cores.values():
            # x3 不在圈上；加支以后 degree_G(x3)=degree_H(x3)+1。
            for root in range(c, m):
                lemma = "2.10" if len(core[root]) == 1 else "2.11"
                if lemma not in stats:
                    continue
                stats[lemma]["root_candidates"] += 1
                graph, x0 = attach_fork(core, root)
                d, degrees, path = optimal_path(graph, c)
                if d < 4 or degrees[:2] != (3, 2):
                    continue
                if lemma == "2.10" and degrees[2] != 2:
                    continue
                if lemma == "2.11" and degrees[2] <= 2:
                    continue
                word = canonical_word(graph, c)
                if word in seen[lemma]:
                    continue
                seen[lemma].add(word)
                value = phi(word)
                passed = less_than_h(value, n)
                s = stats[lemma]
                s["graphs"] += 1
                if not passed:
                    s["violations"] += 1
                    failure = dict(lemma=lemma, n=n, phi=value, adjacency=graph,
                                   canonical_word=word, optimal_path=path,
                                   optimal_degrees=degrees, distance=d)
                    target = out_dir / f"failure_{lemma}_{n}_{s['violations']}.json"
                    target.write_text(json.dumps(failure, indent=2), encoding="utf-8")
                if value > s["max_phi"]:
                    s["max_phi"] = value
                    s["maximizing_graphs"] = 1
                    best_witness[lemma] = dict(
                        lemma=lemma, n=n, phi=value, cycle_length=c,
                        adjacency=graph, canonical_word=word,
                        optimal_path=path, optimal_degrees=degrees, distance=d,
                        exact_cubed_slack=8 * 3 ** (n - 2) - (value - n + 5) ** 3)
                elif value == s["max_phi"]:
                    s["maximizing_graphs"] += 1
        for lemma in active:
            s = stats[lemma]
            row = dict(lemma=lemma, n=n, core_order=m, cores=len(cores), **s,
                       h_approx=h_display(n),
                       gap_approx=h_display(n) - s["max_phi"],
                       passed=s["violations"] == 0,
                       cumulative_seconds=round(time.perf_counter() - start, 3))
            rows.append(row)
            if lemma in best_witness:
                witnesses.append(best_witness[lemma])
            print(f"{lemma},{n},{len(cores)},{s['root_candidates']},"
                  f"{s['graphs']},{s['max_phi']},{h_display(n):.9f},"
                  f"{row['passed']}", flush=True)
        # 每完成一阶即保存；中断时已完成各阶的记录仍保留。
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = dict(
            python=sys.version, platform=platform.platform(), max_n=max_n,
            scope="Original lemma hypotheses; no additional support restrictions",
            completed_rows=len(rows), results=rows,
            all_completed_checks_passed=all(r["passed"] for r in rows))
        (out_dir / "results.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "witnesses.json").write_text(
            json.dumps(witnesses, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-n", type=int, default=18)
    parser.add_argument("--out", type=Path, default=Path("verification_results"))
    parser.add_argument("--self-test", action="store_true", help="仅运行独立子集枚举交叉测试")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows = verify(args.max_n, args.out)
    failures = sum(row["violations"] for row in rows)
    print(f"Finished: {sum(r['graphs'] for r in rows)} graphs checked; "
          f"{failures} violations.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
