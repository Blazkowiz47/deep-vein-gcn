# gcn-deep-vein Memory

## Context Card

Status: Active journal-paper work
Domain: phd
Tags: phd, journal-paper, biometrics, gcn, vein-recognition
Project path: /home/ubuntu/sushrut/gcn-deep-vein
Main brain workstream: /home/ubuntu/.sushrut/wiki/workstreams/gcn-deep-vein/index.md
Devices/servers: raghu2 (SALT servers)
Latest useful result: `rebuttal/inconsistencies.md` now has a 2026-05-01 current-state re-check for the updated `paper.tex` and `response_letter.tex`, separating resolved findings from active inconsistencies.
Current blocker: The updated rebuttal still has active mismatches around stale score-count/runtime tables, five-fold/five-run claims, train/validation identity wording, intra-database protocol wording, processed dataset counts, architecture table values, ablation evidence bases, and LaTeX build dependencies.
Next action: Regenerate score-count/result tables from the corrected `scripts/unenrolled_eval.py` run, then patch stale protocol/repetition wording in the paper and response letter.

## Active Threads

- Journal-paper narrative and evidence consolidation from `final_tables.md`, `ablation/`, `final_runs/`, and `rebuttal/`.
- Run-traceability cleanup so paper claims map cleanly to commands, configs, datasets, and output paths.

## Recent Work

- 2026-04-30: Initialized project memory and linked it to the main knowledge base workstream.
- 2026-04-30: Captured current repo state as `main` at `83fe33e311f12be4f33694e13925f8359072d5da` (`2026-04-29`, `chore: updates`).
- 2026-04-30: Reviewed `rebuttal/paper.tex` against the current evaluation/training scripts and inserted red clarification text for the main protocol and narrative mismatches.
- 2026-04-30: Ran critic/replier agent review for each rebuttal file and logged confirmed inconsistencies in `rebuttal/inconsistencies.md`.
- 2026-05-01: Re-reviewed the updated paper/response files with sub-agents and recorded the current active inconsistency set.

## Recent Runs

- See `runs.md`.

## Durable Learnings

- See `learnings.md`.

## Decisions

- See `decisions.md`.

## Links

- Main workstream: `/home/ubuntu/.sushrut/wiki/workstreams/gcn-deep-vein/index.md`
- Final tables: `/home/ubuntu/sushrut/gcn-deep-vein/final_tables.md`
- Ablation outputs: `/home/ubuntu/sushrut/gcn-deep-vein/ablation/`
- Rebuttal materials: `/home/ubuntu/sushrut/gcn-deep-vein/rebuttal/`
