# Learnings

Durable findings from this project. Keep this compact and useful for future work.

## Confirmed

- `final_tables.md` is the current compact summary of the paper-facing half-subject and full-subject cross-dataset evaluation.
- In the current final tables, `ABCD -> E` is the hardest transfer setting across methods; TAR@FAR=0.1% stays at `0.00` for every listed method in both half-subject and full-subject summaries.
- The proposed method is competitive across several splits, but `BCDE -> A` at very low FAR still trails `Chen et al` in the current summary tables.
- `scripts/unenrolled_eval.py` implements pairwise score distributions for the half-subject protocol; `paper.tex` now matches this, but `response_letter.tex` still has one stale max-gallery explanation.
- `scripts/unenrolled_eval.py` no longer duplicates half-protocol impostor scores inside the full-subject impostor distribution; old score-count tables/artifacts remain stale until regenerated.
- The current cross-dataset artifacts are 4-seed results for most methods, not 5-fold/five-run results; FV-USM/LGFIN has only 3 seeds.
- The current intra-database implementation is 80% development IDs plus 20% left-out pairwise evaluation, not a 50/30/20 unknown-rejection protocol.
- The current paper/rebuttal text has remaining implementation/artifact mismatches for graph neighbor `k`, DSConv stem kernels, runtime values, dataset counts, and ablation evidence bases; the CAH-loss formula difference is intentionally simplified and no longer treated as an active inconsistency.

## Likely But Needs Verification

- The strongest journal-paper framing is likely cross-dataset robustness and consistency across splits rather than uniform dominance at the strictest operating point.
- The t-SNE generalization plots can support qualitative interpretation, but should not be framed as confirming data-driven causality without quantitative stability evidence.

## Failed Approaches

- 

## Reusable Ideas

- Keep paper-facing tables in versioned Markdown (`final_tables.md`) backed by machine-readable JSON outputs so claims stay auditable.
