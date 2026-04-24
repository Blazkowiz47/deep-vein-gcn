# Quality Metric Analysis

This report summarizes lightweight ROI quality metrics over the cached dataset-level results.
The interpretation is intentionally conservative: the analysis describes score distributions, cross-metric relationships,
and composite rankings, but it does not by itself establish that any metric improves recognition performance.

## Key Observations

- `gradient` and `contrast` are consistently the most correlated pair across datasets, indicating partial signal overlap rather than independent evidence.
- `gradient_c` is typically less correlated with the other metrics and therefore receives more weight in the correlation-aware composite, especially on FVUSM and MMCBNU.
- `laplacian` often receives a high redundancy-aware weight, which should be interpreted as lower overlap with the other metrics within a dataset rather than proof of greater downstream utility.
- The highest- and lowest-ranked images remain broadly stable after redundancy-aware weighting, which suggests that the extreme cases are not artifacts of one arbitrary weighting choice.
- Agreement across all metrics on the same bottom-10% or top-10% images is modest, so the metrics should be treated as complementary heuristics rather than interchangeable surrogates. The low Jaccard values indicate that the metrics often disagree on which exact images are most extreme even when their overall rankings are correlated.
- The lowest-ranked examples frequently cluster within one subject or finger identity rather than being uniformly distributed across the dataset, which suggests that the composite score is sensitive to subject-level acquisition factors as well as per-image nuisance.

## Publication-Safe Interpretation

These results support three restrained conclusions.

1. The four handcrafted metrics capture related but non-identical aspects of ROI quality.
2. A correlation-aware composite reduces the influence of highly redundant metrics compared with equal weighting.
3. The resulting ranking is suitable for exploratory analysis and candidate sample filtering, but it still requires validation against downstream verification performance before stronger claims are made.

## Cross-Dataset Context

A direct comparison of raw metric magnitudes across datasets is not fully reliable because the sensors, preprocessing pipelines, and image statistics differ.
This matters especially for `laplacian`, whose scale can vary strongly with resolution, noise floor, and preprocessing. For that reason, the report avoids a hard cross-dataset quality ranking based on all four metrics.

For a limited descriptive comparison, the dimensionless metrics `gradient_c` and `contrast` provide the most interpretable cross-dataset context. Using dataset medians on those two metrics only, the current descriptive ordering is:

1. `mmcbnu` (`gradient_c` median=0.5384, `contrast` median=0.0379; `gradient_c`=1, `contrast`=1)
2. `polyu` (`gradient_c` median=0.4603, `contrast` median=0.0252; `gradient_c`=2, `contrast`=3)
3. `fv300` (`gradient_c` median=0.3730, `contrast` median=0.0296; `gradient_c`=3, `contrast`=2)
4. `vera` (`gradient_c` median=0.1400, `contrast` median=0.0191; `gradient_c`=5, `contrast`=4)
5. `fvusm` (`gradient_c` median=0.2955, `contrast` median=0.0137; `gradient_c`=4, `contrast`=5)

This ordering should be read as descriptive context rather than a claim that one dataset is intrinsically better than another.
It is included only to summarize how the more comparable handcrafted metrics behave across the present cached datasets.

## Dataset Notes

### vera

- Composite spread: mean=0.500, median=0.510, std=0.246. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.
- Correlation-aware weights: `gradient`=0.18, `gradient_c`=0.32, `laplacian`=0.35, `contrast`=0.15.
- Highest-weight metric under redundancy-aware aggregation: `laplacian`. Lowest-weight metric: `contrast`.
- Extreme-case agreement remains limited: bottom-10% Jaccard=0.19, top-10% Jaccard=0.13. This indicates that the metrics do not flag exactly the same images as extreme cases.
- Lowest-ranked examples are concentrated around: `test/070_R/070_R_1.png`, `train/070_L/070_L_1.png`, `train/087_R/087_R_1.png`.
- Highest-ranked examples are concentrated around: `train/042_L/042_L_1.png`, `train/034_L/034_L_1.png`, `train/075_L/075_L_1.png`.

### polyu

- Composite spread: mean=0.500, median=0.496, std=0.232. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.
- Correlation-aware weights: `gradient`=0.19, `gradient_c`=0.27, `laplacian`=0.36, `contrast`=0.18.
- Highest-weight metric under redundancy-aware aggregation: `laplacian`. Lowest-weight metric: `contrast`.
- Extreme-case agreement remains limited: bottom-10% Jaccard=0.11, top-10% Jaccard=0.04. This indicates that the metrics do not flag exactly the same images as extreme cases.
- Lowest-ranked examples are concentrated around: `test/32/38_6_f2_1.bmp`, `test/32/38_3_f2_1.bmp`, `test/32/38_4_f2_1.bmp`.
- Highest-ranked examples are concentrated around: `test/148/140_6_f2_1.bmp`, `test/67/71_2_f2_1.bmp`, `test/187/12_5_f2_1.bmp`.

### mmcbnu

- Composite spread: mean=0.500, median=0.498, std=0.228. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.
- Correlation-aware weights: `gradient`=0.18, `gradient_c`=0.33, `laplacian`=0.29, `contrast`=0.20.
- Highest-weight metric under redundancy-aware aggregation: `gradient_c`. Lowest-weight metric: `gradient`.
- Extreme-case agreement remains limited: bottom-10% Jaccard=0.03, top-10% Jaccard=0.03. This indicates that the metrics do not flag exactly the same images as extreme cases.
- Lowest-ranked examples are concentrated around: `test/001_R_Ring/02.bmp`, `train/001_R_Ring/03.bmp`, `train/033_R_Middle/02.bmp`.
- Highest-ranked examples are concentrated around: `train/083_L_Middle/01.bmp`, `test/090_R_Fore/04.bmp`, `test/018_L_Ring/09.bmp`.

### fvusm

- Composite spread: mean=0.500, median=0.498, std=0.207. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.
- Correlation-aware weights: `gradient`=0.19, `gradient_c`=0.38, `laplacian`=0.25, `contrast`=0.17.
- Highest-weight metric under redundancy-aware aggregation: `gradient_c`. Lowest-weight metric: `contrast`.
- Extreme-case agreement remains limited: bottom-10% Jaccard=0.02, top-10% Jaccard=0.03. This indicates that the metrics do not flag exactly the same images as extreme cases.
- Lowest-ranked examples are concentrated around: `test/vein060_4/test_06.jpg`, `test/vein060_3/test_04.jpg`, `test/vein060_2/test_03.jpg`.
- Highest-ranked examples are concentrated around: `train/vein007_1/train_03.jpg`, `train/vein007_1/train_02.jpg`, `train/vein007_4/train_02.jpg`.

### fv300

- Composite spread: mean=0.500, median=0.506, std=0.256. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.
- Correlation-aware weights: `gradient`=0.19, `gradient_c`=0.26, `laplacian`=0.35, `contrast`=0.20.
- Highest-weight metric under redundancy-aware aggregation: `laplacian`. Lowest-weight metric: `gradient`.
- Extreme-case agreement remains limited: bottom-10% Jaccard=0.12, top-10% Jaccard=0.04. This indicates that the metrics do not flag exactly the same images as extreme cases.
- Lowest-ranked examples are concentrated around: `train/207/39.bmp`, `test/207/33.bmp`, `train/207/21.bmp`.
- Highest-ranked examples are concentrated around: `train/184/15.bmp`, `train/184/16.bmp`, `train/184/18.bmp`.

## Limitations

- The metrics are unsupervised heuristics and may reward some nuisance factors such as strong non-vein edges or sensor-specific texture.
- The correlation-aware weights are descriptive for these cached datasets; they should not be interpreted as universally optimal parameters.
- Stability of the extreme rankings is encouraging, but qualitative inspection and downstream EER analysis are still necessary to establish practical usefulness.

## Recommended Next Step

Evaluate whether removing the lowest-ranked samples, or using the composite score as a covariate, improves verification performance on the same datasets.
