# Router prompts

TreeIntel decides how to use the wiki hierarchy with two LLM calls per query,
both at temperature 0 using a small Gemma-4-class model. The prompts are
released verbatim except that the few-shot examples use neutral placeholder
projects and spaces instead of internal names.

- `locator.txt` — the **tree locator**. Given the query and candidate breadcrumb
  paths (grouped by space, ordered by retrieval relevance), it selects the best
  breadcrumb location(s) to scope to, or returns `none` to fall back to flat
  retrieval. Emits `selection_mode`, `needs_tree`, `selected_paths` (each with a
  per-path confidence), and a rationale.
- `intent.txt` — the **intent classifier**. Given the query and the locator's
  selected candidates, it classifies the query into exactly one of NONE,
  NAVIGATION, TEMPORAL, LISTING, CONTENT, or MIXED, with a confidence and
  rationale.

## From prompt outputs to a retrieval strategy

The two calls feed a deterministic scope-confidence score:

```
s = clamp01(0.35 * path + 0.25 * locator + 0.25 * intent + 0.15 * node
            - breadth - ambiguity)
```

where `path`, `locator`, `intent`, and `node` are signals derived from the
prompt outputs and the matched tree node, and `breadth` / `ambiguity` are
penalties. The score selects the retrieval strategy:

| scope confidence `s` | strategy |
|---|---|
| `s < 0.55` | flat fallback (ordinary hybrid RAG) |
| `0.55 ≤ s < 0.75` | soft path bias |
| `s ≥ 0.75` | hard subtree scope |

CONTENT and MIXED intents are downgraded to soft bias even inside the hard
band. The intent then further specializes the act (direct leaf retrieval,
child listing, or temporal descendant selection).

Over the 250-query enterprise benchmark, the two calls cost roughly 200-400
tokens each, under 0.1 cent per query, with zero parse failures.
