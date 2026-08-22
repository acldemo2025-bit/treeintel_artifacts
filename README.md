# TreeIntel artifacts

Reproducibility artifacts for **TreeIntel: Query-Adaptive Retrieval over Native
Enterprise Wiki Hierarchies**.

TreeIntel decides, per query, whether the query carries usable structural intent
and, when it does, adapts retrieval to the wiki page hierarchy; otherwise it
falls back to ordinary flat retrieval. Two temperature-0 LLM calls (a small
Gemma-4-class model) produce a scope-confidence score that selects one of six
retrieval strategies. See `enterprise/policy.md` for the policy and
`prompts/` for the router prompts.

## What is and is not here

- **Included:** the public **Apache100** benchmark end to end (ground truth,
  per-system retrieval records, and a scorer that reproduces the paper's public
  tables); the router prompts; the analysis code; the aggregate enterprise
  result tables; a synthetic schema demo.
- **Not included:** the confidential 250-query enterprise benchmark and its
  corpus. Those cannot be released. The enterprise numbers are provided as
  aggregate tables only (`enterprise/tables/`) and are not publicly
  reproducible, like all enterprise results in the paper.

## Layout

```
apache100/     Public benchmark: ground_truth.json, routes/<system>.jsonl,
               score.py, README.md
prompts/       Router prompts (tree locator + intent classifier), genericized
enterprise/    Router policy, analysis code (overlap.py, sensitivity.py),
               aggregate enterprise tables, synthetic mini-example
LICENSE        Apache-2.0
NOTICE         Data attribution for the Apache100 corpus
```

## Quickstart

Reproduce the public Apache100 retrieval tables (Recall@3, MRR, Listing F1) with
paired significance tests. Stdlib only, no dependencies:

```bash
cd apache100
python score.py --routes-dir routes --baseline flat
```

Expected TreeIntel means: Recall@3 **0.586**, MRR **0.669**, Listing F1
**0.625**; paired deltas versus flat **+0.378 / +0.418 / +0.568** (all
randomization p < 1e-4). Means are deterministic; CIs and p-values are fixed by
the seed `20260801`.

## Notes

- Means are exact; confidence intervals and p-values depend on the seeded RNG
  draw order and reproduce the paper to three decimals.
- The leakage diagnostic (`enterprise/overlap.py`) and routing-stability
  analysis (`enterprise/sensitivity.py`) both run on the public Apache100 data;
  see `enterprise/README.md`.
