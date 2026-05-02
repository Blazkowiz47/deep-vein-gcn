# Significance Tests

Two-sided exact paired sign-flip permutation tests on EER only.
Positive mean delta favors Proposed Method.
Holm-adjusted significance threshold: `0.05`.
Structured output: `ablation/significance_tests.json`.

| Split | Comparison | n | Mean Delta EER (pp) | Holm p-value | Significant |
|---|---|---:|---:|---:|---|
| Full | Proposed vs ArcVein (seed-matched) | 20 | 15.17 | 7.63e-06 | yes |
| Full | Proposed vs LGFIN (seed-matched) | 20 | 13.05 | 7.63e-06 | yes |
| Full | Proposed vs FV-ViT (seed-matched) | 20 | 13.66 | 7.63e-06 | yes |
| Full | Proposed vs VeinAttNet (seed-matched) | 20 | 4.44 | 9.54e-05 | yes |
