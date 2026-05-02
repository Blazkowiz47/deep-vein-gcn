# Remaining Rebuttal Inconsistencies

Date: 2026-05-01

Scope:
- `rebuttal/paper.tex`
- `rebuttal/response_letter.tex`
- local code/artifacts used to verify claims

Ignored for this checklist:
- CAH-loss formula simplification, per author decision.
- Local figure/build-path issues, because figures are maintained on Overleaf.
- Public checkpoint availability, because checkpoint links will be added later in the GitHub README/release.

## Paper

1. FV-300 score-count row is still stale/shifted.
   Evidence: `rebuttal/paper.tex:808`, `scripts/unenrolled_eval.py:220`, `scripts/unenrolled_eval.py:229`.
   Current expected FV-300 counts are:
   - Half genuine: `598,279`
   - Full genuine: `1,198,364`
   - Half impostor: `181,709,824`
   - Full impostor: `362,208,956`
   The table currently reports FV-300 impostors as `362,208,956` and `543,918,780`.

2. FV-300 and PolyU dataset counts mix raw dataset specs with processed experimental data.
   Evidence: `rebuttal/paper.tex:776`, `rebuttal/paper.tex:782`, `rebuttal/paper.tex:1012`, `rebuttal/paper.tex:1015`, local `data/<dataset>/0`.
   Local processed counts are FV-300 `301` identities / `26,960` images and PolyU `218` identities / `1,274` images, while Table 3 reports FV-300 `300` / `27,600` and PolyU `218` / `872`.

3. Architecture table still mismatches the final config/code.
   Evidence: `rebuttal/paper.tex:582`, `rebuttal/paper.tex:586`, `configs/dscgrapher2.yaml:30`, `configs/dscgrapher2.yaml:45`, `models/dscgrapher.py:139`.
   DSConv stem kernels are `[9, 7, 3]`, not `F=3` for all stem modules. Both Grapher stages use global `kernel_size: 9`; Stage I does not use `K=18`.

4. Repetition wording is improved, but may still overinclude static baselines.
   Evidence: `rebuttal/paper.tex:835`, `scripts/generate_final_tables.py:64`, `scripts/generate_final_tables.py:223`.
   The `stat_seed`/available-runs wording is appropriate for learned methods. If the surrounding results include static baselines, avoid saying all methods are repeated across predefined splits, because static baselines appear single-run in table generation.

5. Runtime table and prose are stale.
   Evidence: `rebuttal/paper.tex:1196`, `rebuttal/paper.tex:1200`, `rebuttal/paper.tex:1213`, `ablation/fv300_runtime_benchmark.json`.
   Current Proposed values are `60.53 ms` CPU / `16.52 img/s` and `13.74 ms` GPU / `72.78 img/s`; the paper reports `55.82`, `17.92`, `13.00`, and `76.94`. ArcVein, LGFIN, and FV-ViT also differ from the JSON.

6. Loss and beta ablation tables mix evidence bases.
   Evidence: `rebuttal/paper.tex:1115`, `rebuttal/paper.tex:1133`, `ablation/ablation_input_loss_eers.jsonl`, `ablation/ablation_beta_eers.jsonl`, `ablation/ablation_results_loss_eers.json`.
   Strong-loss rows are mostly supported by the 5-seed JSONL, but CosFace has a single recorded run and is omitted. Beta `0.1`, `0.5`, and `0.9` are seed-0 ablations, while default `0.3 = 7.44` appears pulled from multi-run final results rather than the beta-sweep artifact.

7. Significance wording is only safe if Chen is excluded.
   Evidence: `rebuttal/paper.tex:1033`, `ablation/significance_tests.json`.
   The claim that all competing deep baselines improve at `p < 10^{-3}` is not supported by the local artifact if Chen is included; Chen has Holm `p = 0.02037`.

## Supplementary Material

1. Statistical significance table has an LGFIN sample-count mismatch.
   Evidence: `rebuttal/supplementary_material.tex:467`, `ablation/significance_tests.json`.
   The supplement reports `n = 20` for Proposed vs LGFIN, but the local significance artifact has `n_pairs = 19`, because FV-USM/LGFIN is missing one matched seed.

2. Statistical-significance scope is ambiguous if Chen remains part of the comparison set.
   Evidence: `rebuttal/supplementary_material.tex:455`, `rebuttal/supplementary_material.tex:466`, `ablation/significance_tests.json`.
   The table excludes Chen, while the local significance artifact includes `Proposed vs Chen et al` with Holm `p = 0.02037`. If Chen is globally excluded from the manuscript, the table is fine; otherwise, avoid wording such as "competing approaches" unless it means only the listed baselines.

3. Intra-database protocol text still says `50%/30%/20%` identity-disjoint training/validation/testing.
   Evidence: `rebuttal/supplementary_material.tex:488`, `cdatasets/intra.py:43`, `cdatasets/intra.py:77`, `scripts/intra_open_set_eval.py:100`, `scripts/intra_open_set_eval.py:252`.
   Local intra code uses an `80%` development-ID / `20%` left-out-ID protocol. Training and validation use image splits for the same development identities, and evaluation uses exhaustive pairwise scores over the left-out identities.

4. Intra result table uses `\multirow{6}` but shows five method rows per dataset.
   Evidence: `rebuttal/supplementary_material.tex:501`, `rebuttal/supplementary_material.tex:508`, `rebuttal/supplementary_material.tex:515`.
   Each dataset currently lists ArcVein, LGFIN, FV-ViT, VeinAttNet, and Proposed Method only. If Chen/resnet rows remain omitted, the multirow count should be `5`.
   If Chen is included from the local intra artifact, the table/prose must also change: Chen has FV-300 mean EER around `0.09%`, lower than Proposed `0.39%`.

5. Intra split-statistics table includes PolyU and VERA even though the intra experiment excludes them.
   Evidence: `rebuttal/supplementary_material.tex:488`, `rebuttal/supplementary_material.tex:650`, `rebuttal/supplementary_material.tex:651`, `scripts/intra_open_set_eval.py:21`, `scripts/intra_open_set_runs.py:26`.
   The text says intra experiments are conducted only on FV-300, FV-USM, and MMCBNU because PolyU/VERA have too few samples. Table `tab:intra_split_stats` is captioned as intra-experiment split statistics but includes PolyU and VERA.

6. Train-validation section still claims subject/identity-disjoint source splits.
   Evidence: `rebuttal/supplementary_material.tex:602`, `rebuttal/supplementary_material.tex:606`, `rebuttal/supplementary_material.tex:634`, `cdatasets/leaveoneout.py:75`, `cdatasets/leaveoneout.py:115`, `train.py:140`.
   Local leave-one-dataset-out training uses the same class IDs across train/test image directories, and `validation` maps to the wrapper's non-train split. The supplement should describe this as a per-identity image split/source-data split unless the code/data are changed.

7. Train-validation statistics table mixes per-identity image counts with identity-disjoint wording.
   Evidence: `rebuttal/supplementary_material.tex:625`, `rebuttal/supplementary_material.tex:634`.
   The table values are images per identity, training images, and validation images, but the caption says the split is at identity level with no identity shared. Those two statements conflict with the local data layout and with the equal train/validation identity counts later shown in `tab:intra_split_stats`.

8. Shared protocol wording says the same statistical seeds were used across all learned methods.
   Evidence: `rebuttal/supplementary_material.tex:548`, `run_name_mappings.py:93`, `ablation/half_subjects_results.json`, `ablation/full_subjects_results.json`.
   This is mostly true, but FV-USM/LGFIN has only seeds `1,2,3`, while the other learned methods have `0,1,2,3`. Use "available matched statistical seeds" if keeping the current artifacts.

9. FV-300 intra split counts are seed-specific but presented as a single general split table.
   Evidence: `rebuttal/supplementary_material.tex:647`, `ablation/intra_open_set_results.jsonl`, local `data/fv300/<stat_seed>`.
   The shown `16,536 / 4,954` train/validation image counts match one seed-specific partition, while FV-300 image counts vary slightly across `stat_seed` splits because removed/available images differ. State the seed or report ranges/averages if this table is meant to summarize all five seeds.

10. FV-300 train-validation table reports `~64 / ~28` images per identity, but local processed seed-0 split is closer to `~69 / ~21`.
    Evidence: `rebuttal/supplementary_material.tex:627`, local `data/fv300/0/train`, local `data/fv300/0/test`.
    Local seed-0 processed counts are `20,728` train images and `6,232` validation/test images over `301` identities, i.e. about `68.86 / 20.70` images per identity. The table should either use processed counts or clearly label the `64 / 28` values as approximate/raw planning values.

## Response Letter

1. Five-fold/five-run wording is still stale.
   Evidence: `rebuttal/response_letter.tex:260`, `rebuttal/response_letter.tex:295`, `rebuttal/response_letter.tex:318`, `rebuttal/response_letter.tex:356`, `rebuttal/paper.tex:836`, `run_name_mappings.py:93`.
   The response letter should match the paper's current `stat_seed` / available matched runs wording. Current artifacts use seeds `0..3` for most learned methods and only `1..3` for FV-USM/LGFIN.

2. Train-validation split is still described as identity-disjoint.
   Evidence: `rebuttal/response_letter.tex:388`, `rebuttal/response_letter.tex:393`, `rebuttal/response_letter.tex:399`, `cdatasets/leaveoneout.py:75`, `cdatasets/leaveoneout.py:115`, `train.py:139`.
   Local leave-one-out training uses train/test image directories for the same class IDs; validation maps to the wrapper's non-train split. The response should avoid claiming disjoint training and validation identities unless the data split is changed.

3. Half-subject protocol still contains max-gallery wording.
   Evidence: `rebuttal/response_letter.tex:495`, `rebuttal/paper.tex:827`, `rebuttal/paper.tex:830`, `scripts/unenrolled_eval.py:220`.
   The response says a probe is accepted by maximum cosine similarity to an enrolled gallery. Current paper/code use exhaustive pairwise enrolled-vs-non-enrolled score distributions.

4. Intra-database protocol wording still says `50%/30%/20%`.
   Evidence: `rebuttal/response_letter.tex:765`, `rebuttal/response_letter.tex:774`, `cdatasets/intra.py:43`, `cdatasets/intra.py:77`, `scripts/intra_open_set_eval.py:60`, `scripts/intra_open_set_eval.py:252`.
   Local intra code uses `80%` development IDs and `20%` left-out IDs, then evaluates exhaustive pairwise scores over the left-out identities.

5. Runtime values and performance wording are stale/overbroad.
   Evidence: `rebuttal/response_letter.tex:522`, `rebuttal/response_letter.tex:581`, `ablation/fv300_runtime_benchmark.json`.
   The response still reports `55.82 ms` CPU / `13.00 ms` GPU and includes overbroad phrases such as "consistently outperforms" and "consistently delivers superior accuracy across all evaluation protocols."
