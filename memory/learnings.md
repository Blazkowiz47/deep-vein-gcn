# Learnings

Durable findings from this project. Keep this compact and useful for future work.

## Confirmed

- `final_tables.md` is the current compact summary of the paper-facing half-subject and full-subject cross-dataset evaluation.
- In the current final tables, `ABCD -> E` is the hardest transfer setting across methods; TAR@FAR=0.1% stays at `0.00` for every listed method in both half-subject and full-subject summaries.
- The proposed method is competitive across several splits, but `BCDE -> A` at very low FAR still trails `Chen et al` in the current summary tables.

## Likely But Needs Verification

- The strongest journal-paper framing is likely cross-dataset robustness and consistency across splits rather than uniform dominance at the strictest operating point.

## Failed Approaches

- 

## Reusable Ideas

- Keep paper-facing tables in versioned Markdown (`final_tables.md`) backed by machine-readable JSON outputs so claims stay auditable.
