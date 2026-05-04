# gcn-deep-vein Memory

## Context Card

Status: Active journal-paper work
Domain: phd
Tags: phd, journal-paper, biometrics, gcn, vein-recognition
Project path: /home/ubuntu/sushrut/gcn-deep-vein
Main brain workstream: /home/ubuntu/.sushrut/wiki/workstreams/gcn-deep-vein/index.md
Devices/servers: raghu2 (SALT servers)
Latest useful result: `rebuttal/response_letter.tex` now also clarifies that checkpoint archives are README-linked with password/access request, while code/scripts/model definitions are public in GitHub.
Current blocker: The updated rebuttal still has active mismatches around FV-300 score counts, raw-vs-processed dataset counts, architecture/graph-`k` claims, ablation evidence bases, runtime values, stale response-letter protocol wording, intra-database protocol wording, and access-controlled checkpoint wording.
Next action: Patch the remaining items in `rebuttal/inconsistencies.md`, then rerun a final consistency pass against local artifacts.

## Active Threads

- Journal-paper narrative and evidence consolidation from `final_tables.md`, `ablation/`, `final_runs/`, and `rebuttal/`.
- Run-traceability cleanup so paper claims map cleanly to commands, configs, datasets, and output paths.

## Recent Work

- 2026-04-30: Initialized project memory and linked it to the main knowledge base workstream.
- 2026-04-30: Captured current repo state as `main` at `83fe33e311f12be4f33694e13925f8359072d5da` (`2026-04-29`, `chore: updates`).
- 2026-04-30: Reviewed `rebuttal/paper.tex` against the current evaluation/training scripts and inserted red clarification text for the main protocol and narrative mismatches.
- 2026-04-30: Ran critic/replier agent review for each rebuttal file and logged confirmed inconsistencies in `rebuttal/inconsistencies.md`.
- 2026-05-01: Re-reviewed the updated paper/response files with sub-agents and recorded the current active inconsistency set.
- 2026-05-02: Aligned project memory initialization to the current template by refreshing `AGENTS.md`, adding `memory/commands/`, and opening today's project note.
- 2026-05-03: Re-reviewed updated rebuttal files and README with four sub-agents; refreshed `rebuttal/inconsistencies.md` to current-only remaining issues.
- 2026-05-04: Synced `rebuttal/paper.tex` half/full result tables to `final_tables.md`, including the adjacent full-subject EER prose.
- 2026-05-04: Updated supplementary intra-database paragraph to remove stale `50/30/20` wording and describe held-out pairwise scoring on the left-out 20% identities.
- 2026-05-04: Renamed the supplementary split section to intra-database train--test, updated table headers/captions, and removed stale train--validation wording/comments.
- 2026-05-04: Removed the redundant/inconsistent per-identity train/validation table from Supplementary Section 5; the remaining table reports development IDs, development train/test image counts, and held-out test IDs.
- 2026-05-04: Updated `rebuttal/response_letter.tex` to remove stale five-fold/five-run wording, max-gallery half-subject wording, and intra-database `50/30/20` wording.
- 2026-05-04: Softened response-letter checkpoint wording: code/scripts/model definitions are in the public repository; `final_runs.zip` checkpoints are linked from README and require password/access request.
- 2026-05-04: Re-aligned project memory initialization to the current node-aware template: refreshed the Sushrut `AGENTS.md` block, synced six `memory/commands/` specs, and updated `memory/devices.md` node wording.

## Recent Runs

- See `runs.md`.

## Durable Learnings

- See `learnings.md`.

## Decisions

- See `decisions.md`.

## Commands

- See `commands/index.md`.

## Links

- Main workstream: `/home/ubuntu/.sushrut/wiki/workstreams/gcn-deep-vein/index.md`
- Final tables: `/home/ubuntu/sushrut/gcn-deep-vein/final_tables.md`
- Ablation outputs: `/home/ubuntu/sushrut/gcn-deep-vein/ablation/`
- Rebuttal materials: `/home/ubuntu/sushrut/gcn-deep-vein/rebuttal/`
