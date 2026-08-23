# Apache100 benchmark

A 100-query reproducibility benchmark over a public Apache wiki corpus
(38,145 pages), used for the external-validity evaluation in the TreeIntel
paper. Queries are split evenly across five task groups (20 each):
Navigation, Content, Listing, Temporal, and Mixed.

This directory reproduces the public-benchmark retrieval tables (Recall@3, MRR,
Listing F1) and the paired significance tests, end to end, from precomputed
per-query records.

## Contents

- `ground_truth.json` — the 100 queries keyed by id, each with its expected
  target page (id, title, breadcrumb, space, last-modified) and, for Listing
  and Mixed queries, the expected child page ids.
- `routes/<system>.jsonl` — one file per compared system. Each line is a
  per-query record with a precomputed `metrics` block. Systems:
  - `flat` — flat hybrid RAG (BM25 + dense + reranking). **Paired baseline.**
  - `flat-path` — flat hybrid RAG with breadcrumb path text appended.
  - `li-vector` — LlamaIndex vector retriever.
  - `li-automerge` — LlamaIndex auto-merging retriever.
  - `raptor` — RAPTOR (collapsed-tree retrieval).
  - `treeintel` — the query-adaptive method from the paper.
- `score.py` — stdlib-only scorer: means, 95% bootstrap CIs, and paired
  randomization tests versus the baseline.

## Reproduce

```bash
python score.py --routes-dir routes --baseline flat
```

Expected TreeIntel means: Recall@3 0.586, MRR 0.669, Listing F1 0.625; paired
deltas versus flat +0.378 / +0.418 / +0.568 (all randomization p < 1e-4).

Means are deterministic. The confidence intervals and p-values depend on the
seeded RNG draw order (`BOOT_SEED = 20260801`, 10k bootstrap / 50k
randomization reps) and reproduce the paper tables to three decimals.

## Record schema (`routes/*.jsonl`)

| field | meaning |
|---|---|
| `id` | query id, matches `ground_truth.json` |
| `group` | task group (NAVIGATION / CONTENT / LISTING / TEMPORAL / MIXED) |
| `query` | the natural-language query |
| `space_key` | Apache wiki space of the expected target |
| `status` | `ok` when the query was scored; other records are skipped |
| `expected_target` | gold page (id, title, breadcrumb, ...) |
| `expected_child_ids` | gold child ids for Listing/Mixed (else empty) |
| `metrics.target_hit` | 1.0 if the gold page is in top-k, else 0.0 |
| `metrics.recall_at_k` | Recall@3 for this query |
| `metrics.mrr` | reciprocal rank of the gold page (0 if absent) |
| `metrics.list_f1` | set F1 over expected children (LISTING/MIXED only) |
| `metrics.topk` / `retrieved_page_ids` | retrieved page ids |
| `metrics.elapsed_ms` | per-query latency |

Generated answers are omitted from these records; only the retrieval metrics
needed to reproduce the tables are included.
