# gcn-deep-vein Memory

## Context Card

Status: Active journal-paper work
Domain: phd
Tags: phd, journal-paper, biometrics, gcn, vein-recognition
Project path: /home/ubuntu/1Projects/deep-vein-gcn
Main brain workstream: /home/ubuntu/.sushrut/wiki/workstreams/gcn-deep-vein/index.md
Devices/servers: raghu2, mobai
Latest useful result: Older full-FV-USM WLD score artifacts reproduce AUC `83.07`, EER `22.87`, TAR `0.00/0.08/20.58`, but all 5,904 current FV-USM WLD MAT features are zero and came from a different export.
Current blocker: WLD exports are all zero, and the shared EER helper reverses FAR/FRR arrays used for TAR reporting; FV-300 static features also lack 22--35 held-out image IDs for seeds 1--4.
Next action: Regenerate valid WLD features and fix/recompute TAR operating points before treating the new handcrafted intra rows as final paper results.

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
- 2026-07-17: Generated and verified exact intra-dataset test manifests for seeds 0--4; confirmed FV-USM/MMCBNU image sets are seed-invariant while FV-300 image counts vary, and identified missing static-feature coverage for FV-300 seeds 1--4.
- 2026-07-18: Drafted and source-checked the second-round response letter; the Reviewer 3 response retains Sections 3--6 and explains their distinct roles.
- 2026-07-20: Added the FV-300 request-by-email access procedure to the README and aligned the Reviewer 2 response letter wording.
- 2026-07-20: Updated the README and Reviewer 1 response to describe the trained checkpoints as openly available through the published link.

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
