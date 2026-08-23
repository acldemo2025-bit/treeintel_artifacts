"""Query-path leakage diagnostic (paper analysis E10).

Addresses the reviewer concern that tree retrieval only wins because queries
leak the exact breadcrumb/path. For each benchmark item we measure the token
overlap between the query and the expected target's structural metadata (title,
breadcrumb, root title, space key), bucket queries by overlap, and report flat
vs adaptive retrieval quality per bucket. The test of interest: does structural
retrieval still help when overlap is LOW?

The primary score is asymmetric containment (how much of the query is covered
by the target metadata):

    query_target_overlap = |query_tokens & meta_tokens| / |query_tokens|

We also report the symmetric Jaccard overlap. Common English stopwords are
removed before tokenizing.

Input is one flat records.jsonl and one adaptive records.jsonl over the SAME
benchmark (shared query ids, each line carrying `query`, `expected_target`, and
a `metrics` block; adaptive records additionally carry
`result.routing.executed_mode`). No corpus text or query text is emitted; the
report is aggregate buckets only, matching the privacy constraints of the
enterprise benchmark.

Usage:
    python overlap.py \
        --flat path/to/flat/records.jsonl \
        --adaptive path/to/tree-adaptive/records.jsonl \
        --out leakage_analysis.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords removed before overlap so that function words do not
# inflate query-metadata overlap.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do", "does",
    "for", "from", "get", "give", "has", "have", "how", "i", "in", "is", "it",
    "its", "list", "me", "my", "of", "on", "or", "our", "show", "the", "their",
    "to", "under", "up", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}

# Overlap buckets (upper bounds exclusive except the last).
BUCKETS = (("low (<0.25)", 0.0, 0.25),
           ("mid (0.25-0.50)", 0.25, 0.50),
           ("high (>=0.50)", 0.50, 1.01))


def tokens(value) -> List[str]:
    return [t for t in TOKEN_RE.findall(str(value or "").lower()) if t not in STOPWORDS]


def target_meta_text(target: dict) -> str:
    parts = [target.get("page_title"), target.get("breadcrumb"),
             target.get("root_title"), target.get("space_key")]
    return " ".join(p for p in parts if p)


def containment(query: str, meta: str) -> float:
    q, m = set(tokens(query)), set(tokens(meta))
    return len(q & m) / len(q) if q else 0.0


def jaccard(query: str, meta: str) -> float:
    q, m = set(tokens(query)), set(tokens(meta))
    return len(q & m) / len(q | m) if (q or m) else 0.0


def load(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok":
                out[str(r["id"])] = r
    return out


def metric(rec: dict, name: str):
    v = (rec.get("metrics") or {}).get(name)
    return float(v) if isinstance(v, (int, float)) else None


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bucket_of(score: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= score < hi:
            return name
    return BUCKETS[-1][0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat", required=True, type=Path)
    ap.add_argument("--adaptive", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    flat = load(args.flat)
    adaptive = load(args.adaptive)
    ids = [i for i in adaptive if i in flat]

    overlaps, jacc, by_group = {}, {}, {}
    for i in ids:
        r = adaptive[i]
        meta = target_meta_text(r.get("expected_target") or {})
        ov = containment(r.get("query", ""), meta)
        overlaps[i] = ov
        jacc[i] = jaccard(r.get("query", ""), meta)
        by_group.setdefault(r.get("group") or "?", []).append(ov)

    ov_vals = list(overlaps.values())
    jc_vals = list(jacc.values())

    L: List[str] = []
    L.append("# Query-Path Leakage Analysis\n")
    L.append("Aggregate token-overlap buckets only; no query text, titles, "
             "breadcrumbs, URLs, or page ids are emitted.\n")
    L.append("## Overall Overlap Distribution\n")
    L.append("| Measure | Mean | Median | p25 | p75 | p90 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| Query-target overlap | {mean(ov_vals):.3f} | {pct(ov_vals,0.5):.3f} | "
             f"{pct(ov_vals,0.25):.3f} | {pct(ov_vals,0.75):.3f} | {pct(ov_vals,0.90):.3f} |")
    L.append(f"| Query-target Jaccard | {mean(jc_vals):.3f} | {pct(jc_vals,0.5):.3f} | "
             f"{pct(jc_vals,0.25):.3f} | {pct(jc_vals,0.75):.3f} | {pct(jc_vals,0.90):.3f} |")

    L.append("\n## Overlap by Task Group\n")
    L.append("| Group | n | Mean overlap | Median overlap | p75 |")
    L.append("|---|---:|---:|---:|---:|")
    for g in sorted(by_group):
        vs = by_group[g]
        L.append(f"| {g} | {len(vs)} | {mean(vs):.3f} | {pct(vs,0.5):.3f} | {pct(vs,0.75):.3f} |")

    L.append("\n## Results by Overlap Bucket\n")
    L.append("| Overlap bucket | n | Flat R@3 | Adaptive R@3 | Adaptive delta | Flat MRR | Adaptive MRR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    modes_by_bucket: Dict[str, Dict[str, int]] = {}
    for name, lo, hi in BUCKETS:
        bids = [i for i in ids if lo <= overlaps[i] < hi]
        fr = [metric(flat[i], "recall_at_k") for i in bids if metric(flat[i], "recall_at_k") is not None]
        ar = [metric(adaptive[i], "recall_at_k") for i in bids if metric(adaptive[i], "recall_at_k") is not None]
        fm = [metric(flat[i], "mrr") for i in bids if metric(flat[i], "mrr") is not None]
        am = [metric(adaptive[i], "mrr") for i in bids if metric(adaptive[i], "mrr") is not None]
        L.append(f"| {name} | {len(bids)} | {mean(fr):.3f} | {mean(ar):.3f} | "
                 f"{mean(ar)-mean(fr):+.3f} | {mean(fm):.3f} | {mean(am):.3f} |")
        counts: Dict[str, int] = {}
        for i in bids:
            mode = ((adaptive[i].get("result") or {}).get("routing") or {}).get("executed_mode")
            if mode:
                counts[mode] = counts.get(mode, 0) + 1
        modes_by_bucket[name] = counts

    L.append("\n## Adaptive Routing Modes by Overlap Bucket\n")
    L.append("| Overlap bucket | Dominant adaptive modes |")
    L.append("|---|---|")
    for name, _, _ in BUCKETS:
        top = sorted(modes_by_bucket[name].items(), key=lambda kv: -kv[1])
        L.append(f"| {name} | " + ", ".join(f"`{m}` {c}" for m, c in top) + " |")

    report = "\n".join(L)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
