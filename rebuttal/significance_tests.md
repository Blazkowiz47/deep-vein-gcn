# Significance Tests

Two-sided exact paired sign-flip permutation tests on EER only.
Positive mean delta favors Proposed Method.
Holm-adjusted significance threshold: `0.05`.
Structured output: `ablation/significance_tests.json`.

| Split | Comparison | n | Mean Delta EER (pp) | Holm p-value | Significant |
|---|---|---:|---:|---:|---|
| Full | Proposed vs ArcVein (seed-matched) | 20 | 15.17 | 9.54e-06 | yes |
| Full | Proposed vs LGFIN (seed-matched) | 19 | 13.33 | 1.14e-05 | yes |
| Full | Proposed vs FV-ViT (seed-matched) | 20 | 13.66 | 9.54e-06 | yes |
| Full | Proposed vs Chen et al (seed-matched) | 20 | 2.64 | 0.0204 | yes |
| Full | Proposed vs VeinAttNet (seed-matched) | 20 | 4.44 | 0.0002 | yes |
