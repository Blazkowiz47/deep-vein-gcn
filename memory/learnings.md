# Learnings

Durable findings from this project. Keep this compact and useful for future work.

## Confirmed

- `final_tables.md` is the current compact summary of the paper-facing half-subject and full-subject cross-dataset evaluation.
- In the current final tables, `ABCD -> E` is the hardest transfer setting across methods; TAR@FAR=0.1% stays at `0.00` for every listed method in both half-subject and full-subject summaries.
- The proposed method is competitive across several splits, but `BCDE -> A` at very low FAR still trails `Chen et al` in the current summary tables.
- `scripts/unenrolled_eval.py` implements pairwise score distributions for the half-subject protocol; `paper.tex` now matches this, but `response_letter.tex` still has one stale max-gallery explanation.
- `scripts/unenrolled_eval.py` no longer duplicates half-protocol impostor scores inside the full-subject impostor distribution; the remaining score-count issue is the FV-300 table row, which still carries stale/shifted impostor counts.
- The current cross-dataset artifacts are 4-seed results for all learned methods on all datasets, not 5-fold/five-run results.
- The current intra-database implementation is 80% development IDs plus 20% left-out pairwise evaluation, not a 50/30/20 unknown-rejection protocol.
- The current paper/rebuttal text has remaining implementation/artifact mismatches for graph neighbor `k`, DSConv stem kernels, runtime values, dataset counts, and ablation evidence bases; the CAH-loss formula difference is intentionally simplified and no longer treated as an active inconsistency.
- `README.md` now links a `final_runs.zip` checkpoint archive, but the SharePoint URL is access-controlled from unauthenticated requests, so manuscript wording should not imply checkpoints are directly included in the public repository unless access is made public.
- When syncing the paper-facing half/full tables from `final_tables.md`, also check adjacent narrative values; the full-subject prose must track the final proposed-method EERs for `ABDE -> C` (`7.45%`) and `ABCE -> D` (`5.20%`).
- The intra-database supplementary protocol should be described as an 80% model-development identity split plus a fully held-out 20% identity evaluation split with pairwise genuine/impostor scoring, not as a `50/30/20` identity-disjoint train/validation/test split.
- For intra-database runs, the `train` and `test` folders are used within the 80% development identities for optimisation and validation/model selection; the subject-disjoint open-set test set is the remaining 20% identities, evaluated separately.
- Supplementary intra-database split reporting is clearest as a single identity-level table; the old per-identity train/validation table is redundant and easy to misread as the open-set identity split.
- In the response letter, significance-test `n=20` should be explained as five held-out datasets times four predefined statistical seeds, not as five seeds/five-fold evaluation per held-out split.
- Checkpoint wording should not imply trained checkpoints are directly included in the GitHub repository; current accurate phrasing is that README links `final_runs.zip` and users can request password/access.

## Likely But Needs Verification

- The strongest journal-paper framing is likely cross-dataset robustness and consistency across splits rather than uniform dominance at the strictest operating point.
- The t-SNE generalization plots can support qualitative interpretation, but should not be framed as confirming data-driven causality without quantitative stability evidence.

## Failed Approaches

- 

## Reusable Ideas

- Keep paper-facing tables in versioned Markdown (`final_tables.md`) backed by machine-readable JSON outputs so claims stay auditable.
