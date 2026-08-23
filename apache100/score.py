"""Reproduce the Apache100 retrieval tables: means, 95% bootstrap CIs, and
paired randomization tests versus a baseline system.

Stdlib only. Deterministic means; CIs and p-values depend on the seeded RNG
draw order (fixed by BOOT_SEED). Reproduces the public-benchmark tables in the
TreeIntel paper to three decimals.

Each system is a JSONL file of per-query records with a precomputed `metrics`
block (target_hit, recall_at_k, mrr, and list_f1 for LISTING queries). Records
are keyed by `id`; systems are compared on the shared, ok-status query IDs.

Usage:
    # score every routes/*.jsonl, treating `flat` as the paired baseline
    python score.py --routes-dir routes --baseline flat

    # or name systems explicitly (overrides / adds to auto-discovery)
    python score.py --study flat=routes/flat.jsonl \
                    --study treeintel=routes/treeintel.jsonl \
                    --baseline flat
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Sequence

BOOT_SEED = 20260801
BOOT_REPS = 10000
RAND_REPS = 50000
RANK_METRICS = ("target_hit", "recall_at_k", "mrr")   # defined for all queries
LIST_METRIC = "list_f1"                                # defined only for LISTING


def load_records(path: str) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                latest[str(r["id"])] = r
    return latest


def metric(rec: dict, name: str):
    if rec.get("status") != "ok":
        return None
    v = (rec.get("metrics") or {}).get(name)
    return float(v) if isinstance(v, (int, float)) else None


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def boot_ci(values: List[float], rng: random.Random, reps: int = BOOT_REPS):
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(reps):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * (reps - 1))]
    hi = means[int(0.975 * (reps - 1))]
    return (lo, hi)


def paired_delta(a: List[float], b: List[float], rng: random.Random):
    """a/b are aligned per-query (a = system, b = baseline)."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    md = mean(diffs)
    boot = []
    for _ in range(BOOT_REPS):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boot.append(s / n)
    boot.sort()
    ci = (boot[int(0.025 * (BOOT_REPS - 1))], boot[int(0.975 * (BOOT_REPS - 1))])
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    obs = abs(md)
    ge = 0
    for _ in range(RAND_REPS):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= obs - 1e-12:
            ge += 1
    p = (ge + 1) / (RAND_REPS + 1)
    return {"n": n, "mean_delta": md, "ci": ci, "wins": wins, "losses": losses, "rand_p": p}


def aligned(sys_recs, base_recs, name, group=None):
    """Per-query aligned value lists over ids present+ok in both, optional group filter."""
    sv, bv = [], []
    for qid, sr in sys_recs.items():
        br = base_recs.get(qid)
        if br is None:
            continue
        if group and str(sr.get("group")) != group:
            continue
        s, b = metric(sr, name), metric(br, name)
        if s is not None and b is not None:
            sv.append(s)
            bv.append(b)
    return sv, bv


def fmt(v: float) -> str:
    return f"{v:+.3f}"


def parse_studies(args) -> Dict[str, str]:
    studies: Dict[str, str] = {}
    if args.routes_dir:
        for path in sorted(glob.glob(os.path.join(args.routes_dir, "*.jsonl"))):
            studies[os.path.splitext(os.path.basename(path))[0]] = path
    for spec in args.study or []:
        if "=" not in spec:
            raise SystemExit(f"--study must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        studies[name] = path
    return studies


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routes-dir", default="routes",
                    help="Directory of <system>.jsonl files (default: routes).")
    ap.add_argument("--study", action="append",
                    help="name=path.jsonl; repeatable; adds to/overrides --routes-dir.")
    ap.add_argument("--baseline", default="flat",
                    help="System name used as the paired baseline (default: flat).")
    ap.add_argument("--out", default=None, help="Write markdown report here.")
    args = ap.parse_args(argv)

    studies = {name: load_records(path) for name, path in parse_studies(args).items()}
    if args.baseline not in studies:
        raise SystemExit(f"baseline '{args.baseline}' not among systems: {sorted(studies)}")
    base = studies[args.baseline]
    others = [n for n in studies if n != args.baseline]

    rng = random.Random(BOOT_SEED)
    lines: List[str] = []
    lines.append("# Apache100 Statistical Analysis\n")
    lines.append(f"Baseline: `{args.baseline}` | systems: {', '.join(sorted(studies))}  ")
    lines.append(f"Bootstrap seed: `{BOOT_SEED}` | bootstrap reps: {BOOT_REPS} | "
                 f"randomization reps: {RAND_REPS}\n")

    lines.append("## Overall Means with 95% Bootstrap CI\n")
    lines.append("| System | Metric | n | Mean | 95% Bootstrap CI |")
    lines.append("|---|---|---:|---:|---|")
    for name in [args.baseline] + sorted(others):
        recs = studies[name]
        for m in RANK_METRICS + (LIST_METRIC,):
            if m == LIST_METRIC:
                vals = [metric(r, m) for r in recs.values()
                        if str(r.get("group")) == "LISTING" and metric(r, m) is not None]
            else:
                vals = [v for v in (metric(r, m) for r in recs.values()) if v is not None]
            lo, hi = boot_ci(vals, rng)
            lines.append(f"| {name} | {m} | {len(vals)} | {mean(vals):.3f} | [{lo:.3f}, {hi:.3f}] |")

    lines.append(f"\n## Paired Deltas vs `{args.baseline}`\n")
    lines.append("| System | Metric | n pairs | Mean delta | 95% paired bootstrap CI | "
                 "Wins/losses | Randomization p |")
    lines.append("|---|---|---:|---:|---|---:|---:|")
    for name in sorted(others):
        recs = studies[name]
        for m in RANK_METRICS:
            sv, bv = aligned(recs, base, m)
            d = paired_delta(sv, bv, rng)
            lines.append(f"| {name} | {m} | {d['n']} | {fmt(d['mean_delta'])} | "
                         f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['wins']}/{d['losses']} | "
                         f"{d['rand_p']:.4f} |")
        sv, bv = aligned(recs, base, LIST_METRIC, group="LISTING")
        if sv:
            d = paired_delta(sv, bv, rng)
            lines.append(f"| {name} | {LIST_METRIC} | {d['n']} | {fmt(d['mean_delta'])} | "
                         f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['wins']}/{d['losses']} | "
                         f"{d['rand_p']:.4f} |")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
