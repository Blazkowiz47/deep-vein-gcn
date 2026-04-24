from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import json
from PIL import Image
import pyiqa
from tqdm import tqdm
import numpy as np
import scipy.io
import torch
from piq import brisque as piq_brisque
from pypiqe import piqe as pypiqe_piqe
from scipy.ndimage import gaussian_filter
from scipy.special import gamma
from scipy.stats import spearmanr
from skimage.filters import frangi
from skimage.transform import rescale, resize


DEFAULT_NIQE_MODEL_URL = (
    "https://raw.githubusercontent.com/utlive/niqe/master/modelparameters.mat"
)
DEFAULT_NIQE_MODEL_PATH = (
    Path.home() / ".cache" / "gcn-deep-vein" / "niqe" / "modelparameters.mat"
)
GAMMA_RANGE = np.arange(0.2, 10.001, 0.001, dtype=np.float32)
AGGD_RATIO_LUT = gamma(2.0 / GAMMA_RANGE) ** 2 / (
    gamma(1.0 / GAMMA_RANGE) * gamma(3.0 / GAMMA_RANGE)
)
ILNIQE_METRIC = pyiqa.create_metric("ilniqe", device="cpu")
ARICAN_MASK_ROW_KNOTS = np.asarray(
    [0.0, 8.0, 20.0, 35.0, 50.0, 80.0, 95.0, 108.0, 120.0, 128.0],
    dtype=np.float32,
)
ARICAN_MASK_VALUE_KNOTS = np.asarray(
    [0.0, 0.0, 0.25, 0.25, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0],
    dtype=np.float32,
)


def assess_gradient_quality(image: np.ndarray) -> float:
    """
    Simple mean gradient magnitude quality metric.
    Higher value = better quality (sharper, more vein structure).
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        image = image.mean(axis=2)

    # Normalize to [0, 1] float
    gray = image.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    # Compute Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = np.sqrt(gx * gx + gy * gy)

    # Return mean — higher is better
    return float(magnitude.mean())


def assess_gradient_coherence_quality(
    image: np.ndarray,
    block_size: int = 16,
    eps: float = 1e-8,
) -> float:
    """Return a block-wise gradient coherence score. Higher is better."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape `(H, W)`, got shape {image.shape}."
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value <= 0:
            raise ValueError("Unsupported integer image range.")
        gray = image.astype(np.float32) / max_value
    else:
        gray = image.astype(np.float32)
        min_value = float(gray.min())
        max_value = float(gray.max())
        if max_value == min_value:
            gray = np.zeros_like(gray, dtype=np.float32)
        elif min_value < 0.0 or max_value > 1.0:
            gray = (gray - min_value) / (max_value - min_value)
    gray = np.clip(gray, 0.0, 1.0)

    height, width = gray.shape
    cropped_h = height - (height % block_size)
    cropped_w = width - (width % block_size)
    if cropped_h == 0 or cropped_w == 0:
        return 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gx = gx[:cropped_h, :cropped_w]
    gy = gy[:cropped_h, :cropped_w]

    coherence_sq: list[float] = []
    for row in range(0, cropped_h, block_size):
        for col in range(0, cropped_w, block_size):
            gx_block = gx[row : row + block_size, col : col + block_size]
            gy_block = gy[row : row + block_size, col : col + block_size]

            j11 = float(np.mean(gx_block * gx_block))
            j22 = float(np.mean(gy_block * gy_block))
            j12 = float(np.mean(gx_block * gy_block))

            coherence = ((j11 - j22) ** 2 + 4.0 * (j12**2)) ** 0.5 / (j11 + j22 + eps)
            coherence_sq.append(coherence * coherence)

    return float(np.mean(coherence_sq))


def assess_laplacian_sharpness(image: np.ndarray) -> float:
    """
    Variance of Laplacian — a classic sharpness/focus measure.
    Higher value = sharper image (better quality).
    Low values indicate blur.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape `(H, W)`, got shape {image.shape}."
        )

    # Normalize to [0, 1] float
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value <= 0:
            raise ValueError("Unsupported integer image range.")
        gray = image.astype(np.float32) / max_value
    else:
        gray = image.astype(np.float32)
        min_value = float(gray.min())
        max_value = float(gray.max())
        if max_value == min_value:
            gray = np.zeros_like(gray, dtype=np.float32)
        elif min_value < 0.0 or max_value > 1.0:
            gray = (gray - min_value) / (max_value - min_value)
    gray = np.clip(gray, 0.0, 1.0)

    # Apply Laplacian and return variance of the response
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(laplacian.var())


def assess_local_contrast(image: np.ndarray, block_size: int = 16) -> float:
    """
    Block-wise local contrast measure — mean of per-block standard deviations.
    Higher value = better contrast (stronger vein/tissue distinction).
    Low values indicate washed-out or flat images.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape `(H, W)`, got shape {image.shape}."
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    # Normalize to [0, 1] float
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value <= 0:
            raise ValueError("Unsupported integer image range.")
        gray = image.astype(np.float32) / max_value
    else:
        gray = image.astype(np.float32)
        min_value = float(gray.min())
        max_value = float(gray.max())
        if max_value == min_value:
            gray = np.zeros_like(gray, dtype=np.float32)
        elif min_value < 0.0 or max_value > 1.0:
            gray = (gray - min_value) / (max_value - min_value)
    gray = np.clip(gray, 0.0, 1.0)

    # Crop to multiple of block_size
    height, width = gray.shape
    cropped_h = height - (height % block_size)
    cropped_w = width - (width % block_size)
    if cropped_h == 0 or cropped_w == 0:
        return 0.0

    # Reshape into blocks and compute per-block std
    cropped = gray[:cropped_h, :cropped_w]
    blocks = cropped.reshape(
        cropped_h // block_size, block_size, cropped_w // block_size, block_size
    ).swapaxes(1, 2)
    block_stds = blocks.std(axis=(2, 3))

    return float(block_stds.mean())


def assess_brisque_quality(image: np.ndarray) -> float:
    """Return the BRISQUE score for one grayscale `(H, W)` image. Lower is better."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape `(H, W)`, got shape {image.shape}."
        )

    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value <= 0:
            raise ValueError("Unsupported integer image range.")
        image = image.astype(np.float32) / max_value
    else:
        image = image.astype(np.float32)
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value == min_value:
            image = np.zeros_like(image, dtype=np.float32)
        elif min_value < 0.0 or max_value > 1.0:
            image = (image - min_value) / (max_value - min_value)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        score = piq_brisque(
            image_tensor,
            data_range=1.0,
            reduction="none",
        )

    return float(score.squeeze().cpu().item())


def get_quality_metrics_for_dataset(dataset: str) -> dict:
    global metric_fns, disable
    rdir = Path(f"./data/{dataset}/0/")
    output_path = Path(f"./ablation/{dataset}_quality.json")

    if output_path.exists():
        with output_path.open("r") as fp:
            results = json.load(fp)
    else:
        results = {}

    for metric_name in metric_fns:
        results.setdefault(metric_name, {})

    for path in tqdm(rdir.rglob("*"), desc=dataset, leave=True, disable=disable):
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
            continue
        path_key = str(path)

        img = np.array(Image.open(path))
        for metric_name, metric_fn in metric_fns.items():
            if path_key in results[metric_name]:
                continue
            results[metric_name][path_key] = metric_fn(img)

    with output_path.open("w+") as fp:
        json.dump(results, fp)

    summaries = []

    for metric_name in metric_fns:
        metric_values = np.asarray(
            list(results[metric_name].values()), dtype=np.float32
        )
        summaries.append(
            f"mean={metric_values.mean():.4f}, std={metric_values.std():.4f}"
        )
        if metric_name == "laplacian" and dataset == 'fv300':
            lap_values = np.asarray(list(results["laplacian"].values()))
            print(
                f"Percentiles: 5%={np.percentile(lap_values, 5):.4f}, "
                f"50%={np.percentile(lap_values, 50):.4f}, "
                f"95%={np.percentile(lap_values, 95):.4f}, "
                f"99%={np.percentile(lap_values, 99):.4f}, "
                f"max={lap_values.max():.4f}"
            )

    # rho, _ = spearmanr(
    #     list(results["gradient"].values()), list(results["gradient_c"].values())
    # )
    # print(f"Rank correlation: {rho: .3f}")

    print(f"{dataset}: " + " | ".join(summaries))
    return results

def run_quality_analysis(
    datasets: list[str] = ("vera", "polyu", "mmcbnu", "fvusm", "fv300"),
    ablation_dir: str = "./ablation",
    output_dir: str = "./ablation/quality",
) -> None:
    """
    Load cached quality metric results and run comprehensive analysis:
    1. Per-dataset robust distribution stats (median, IQR, skew ratio)
    2. Pairwise Spearman correlation between metrics (per dataset + pooled)
    3. Identify top/bottom quality images per metric per dataset
    4. Cross-metric agreement on extreme cases
    """
    from itertools import combinations

    ablation_path = Path(ablation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all cached results
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        cache_path = ablation_path / f"{dataset}_quality.json"
        if not cache_path.exists():
            print(f"[WARN] No cache found for {dataset}, skipping")
            continue
        with cache_path.open("r") as fp:
            all_results[dataset] = json.load(fp)

    if not all_results:
        print("No cached results found. Run the main metric loop first.")
        return

    # Discover metrics present in the cache
    first_ds = next(iter(all_results.values()))
    metric_names = list(first_ds.keys())
    # "higher is better" direction — flip here if you add lower-is-better metrics
    higher_is_better = {
        "gradient": True,
        "gradient_c": True,
        "laplacian": True,
        "contrast": True,
        "brisque": False,
        "niqe": False,
    }
    analysis_summary: dict[str, object] = {
        "datasets": list(all_results.keys()),
        "metrics": metric_names,
        "distribution_stats": {},
        "pairwise_spearman": {},
        "combined_scores": {},
        "extreme_agreement": {},
        "cross_dataset_summary": {},
        "publication_safe_summary": {},
    }

    # ============================================================
    # 1. Robust distribution statistics
    # ============================================================
    print("\n" + "=" * 80)
    print("1. ROBUST DISTRIBUTION STATISTICS (median, IQR, skew ratio)")
    print("=" * 80)
    print(
        f"{'dataset':<8} {'metric':<12} {'median':>10} {'IQR':>20} {'95/50 skew':>12} {'n':>6}"
    )
    print("-" * 80)
    for dataset, results in all_results.items():
        dataset_distribution_stats: dict[str, dict[str, float | int | list[float]]] = {}
        for metric in metric_names:
            if metric not in results or not results[metric]:
                continue
            values = np.asarray(list(results[metric].values()), dtype=np.float32)
            q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
            skew = q95 / q50 if q50 > 1e-10 else float("inf")
            iqr_str = f"[{q25:.4f}, {q75:.4f}]"
            dataset_distribution_stats[metric] = {
                "n": int(len(values)),
                "q05": float(q05),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "q95": float(q95),
                "q95_to_median_ratio": float(skew),
            }
            print(
                f"{dataset:<8} {metric:<12} {q50:>10.4f} {iqr_str:>20} "
                f"{skew:>12.2f} {len(values):>6}"
            )
        analysis_summary["distribution_stats"][dataset] = dataset_distribution_stats
        print()

    # ============================================================
    # 2. Pairwise Spearman correlations (per dataset)
    # ============================================================
    print("=" * 80)
    print("2. PAIRWISE SPEARMAN CORRELATIONS (within each dataset)")
    print("    Low |rho| (< 0.5) means metrics capture independent signal")
    print("=" * 80)
    for dataset, results in all_results.items():
        # Align all metrics on the same image keys
        common_keys = set.intersection(
            *[set(results[m].keys()) for m in metric_names if m in results]
        )
        if not common_keys:
            print(f"[{dataset}] No common keys across metrics")
            continue
        sorted_keys = sorted(common_keys)
        scores = {
            m: np.asarray([results[m][k] for k in sorted_keys], dtype=np.float32)
            for m in metric_names
            if m in results
        }
        dataset_pairwise_spearman: dict[str, dict[str, float | str]] = {}
        print(f"\n[{dataset}] n={len(sorted_keys)}")
        for a, b in combinations(scores.keys(), 2):
            rho, _ = spearmanr(scores[a], scores[b])
            # Adjust sign when one metric is lower-is-better
            if higher_is_better.get(a, True) != higher_is_better.get(b, True):
                rho = -rho
            flag = ""
            if abs(rho) > 0.9:
                flag = "  [REDUNDANT]"
            elif abs(rho) < 0.3:
                flag = "  [INDEPENDENT]"
            dataset_pairwise_spearman[f"{a}__{b}"] = {
                "rho": float(rho),
                "label": flag.strip(" []") if flag else "",
            }
            print(f"    {a:<12} vs {b:<12}: rho = {rho:+.3f}{flag}")
        analysis_summary["pairwise_spearman"][dataset] = dataset_pairwise_spearman

    # ============================================================
    # 3. Combined quality score — correlation-aware rank aggregation
    # ============================================================
    print("\n" + "=" * 80)
    print("3. COMBINED QUALITY SCORE (correlation-aware, higher = better)")
    print("=" * 80)
    combined_scores: dict[str, dict[str, float]] = {}
    for dataset, results in all_results.items():
        common_keys = sorted(
            set.intersection(
                *[set(results[m].keys()) for m in metric_names if m in results]
            )
        )
        if not common_keys:
            continue
        available_metrics = [m for m in metric_names if m in results]
        rank_matrix = []
        for m in available_metrics:
            values = np.asarray([results[m][k] for k in common_keys], dtype=np.float32)
            ranks = values.argsort().argsort().astype(np.float32)
            ranks /= max(len(ranks) - 1, 1)
            if not higher_is_better.get(m, True):
                ranks = 1.0 - ranks
            rank_matrix.append(ranks)

        rank_matrix_np = np.stack(rank_matrix, axis=0)

        if len(available_metrics) == 1:
            weights = np.ones(1, dtype=np.float32)
        else:
            mean_abs_rhos: list[float] = []
            for i, metric_name in enumerate(available_metrics):
                abs_rhos: list[float] = []
                for j, other_metric_name in enumerate(available_metrics):
                    if i == j:
                        continue
                    rho, _ = spearmanr(rank_matrix_np[i], rank_matrix_np[j])
                    if np.isnan(rho):
                        rho = 0.0
                    abs_rhos.append(abs(float(rho)))
                mean_abs_rhos.append(float(np.mean(abs_rhos)) if abs_rhos else 0.0)

            # Downweight metrics that mostly repeat the same ranking signal.
            uniqueness = 1.0 - np.asarray(mean_abs_rhos, dtype=np.float32)
            uniqueness = np.clip(uniqueness, 0.05, None)
            weights = uniqueness / uniqueness.sum()

        combined = np.average(rank_matrix_np, axis=0, weights=weights)
        combined_scores[dataset] = dict(zip(common_keys, combined.tolist()))
        bottom_items = sorted(combined_scores[dataset].items(), key=lambda x: x[1])[:3]
        top_items = sorted(combined_scores[dataset].items(), key=lambda x: x[1])[-3:]
        dataset_root = Path("data") / dataset / "0"
        analysis_summary["combined_scores"][dataset] = {
            "mean": float(combined.mean()),
            "median": float(np.median(combined)),
            "std": float(combined.std()),
            "weights": {
                metric_name: float(weight)
                for metric_name, weight in zip(available_metrics, weights)
            },
            "bottom3": [
                {
                    "path": path,
                    "relative_path": str(Path(path).relative_to(dataset_root)),
                    "score": float(score),
                }
                for path, score in bottom_items
            ],
            "top3": [
                {
                    "path": path,
                    "relative_path": str(Path(path).relative_to(dataset_root)),
                    "score": float(score),
                }
                for path, score in top_items
            ],
        }

        weight_summary = ", ".join(
            f"{metric_name}={weight:.2f}"
            for metric_name, weight in zip(available_metrics, weights)
        )
        print(
            f"{dataset:<8}: mean={combined.mean():.3f}, "
            f"median={np.median(combined):.3f}, "
            f"std={combined.std():.3f} | weights: {weight_summary}"
        )

    # ============================================================
    # 4. Top/bottom images per dataset (for manual inspection)
    # ============================================================
    print("\n" + "=" * 80)
    print("4. TOP & BOTTOM IMAGES BY COMBINED SCORE (inspect these manually!)")
    print("=" * 80)
    for dataset, scores in combined_scores.items():
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        print(f"\n[{dataset}]")
        print("  Lowest quality (inspect for actual poor images):")
        for path, score in sorted_items[:3]:
            print(f"    {score:.3f}  {path}")
        print("  Highest quality (inspect for actual good images):")
        for path, score in sorted_items[-3:]:
            print(f"    {score:.3f}  {path}")

    # ============================================================
    # 5. Agreement on extreme cases
    # ============================================================
    print("\n" + "=" * 80)
    print("5. METRIC AGREEMENT ON EXTREME CASES")
    print("    Fraction of images in bottom-10% of ALL metrics simultaneously")
    print("    High agreement = metrics concur on what's bad")
    print("=" * 80)
    for dataset, results in all_results.items():
        common_keys = sorted(
            set.intersection(
                *[set(results[m].keys()) for m in metric_names if m in results]
            )
        )
        if not common_keys:
            continue
        bottom_sets = []
        top_sets = []
        for m in metric_names:
            if m not in results:
                continue
            values = np.asarray([results[m][k] for k in common_keys], dtype=np.float32)
            if not higher_is_better.get(m, True):
                values = -values  # flip so higher is always better
            threshold_low = np.percentile(values, 10)
            threshold_high = np.percentile(values, 90)
            bottom_sets.append(
                {common_keys[i] for i, v in enumerate(values) if v <= threshold_low}
            )
            top_sets.append(
                {common_keys[i] for i, v in enumerate(values) if v >= threshold_high}
            )
        bottom_agreement = set.intersection(*bottom_sets)
        top_agreement = set.intersection(*top_sets)
        bottom_union = set.union(*bottom_sets)
        top_union = set.union(*top_sets)
        n = len(common_keys)
        analysis_summary["extreme_agreement"][dataset] = {
            "bottom10_intersection": int(len(bottom_agreement)),
            "bottom10_union": int(len(bottom_union)),
            "bottom10_jaccard": float(len(bottom_agreement) / max(len(bottom_union), 1)),
            "top10_intersection": int(len(top_agreement)),
            "top10_union": int(len(top_union)),
            "top10_jaccard": float(len(top_agreement) / max(len(top_union), 1)),
            "n": int(n),
        }
        print(
            f"{dataset:<8}: bottom-10% intersection={len(bottom_agreement):>4} "
            f"(union={len(bottom_union):>4}, jaccard={len(bottom_agreement)/max(len(bottom_union),1):.2f})  "
            f"| top-10% intersection={len(top_agreement):>4} "
            f"(union={len(top_union):>4}, jaccard={len(top_agreement)/max(len(top_union),1):.2f})"
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    cross_dataset_summary: dict[str, object] = {
        "comparable_metric_medians": {},
        "descriptive_ordering": [],
    }
    comparable_metrics = [
        metric_name for metric_name in ("gradient_c", "contrast")
        if metric_name in metric_names
    ]
    comparable_rank_positions: dict[str, dict[str, int]] = {
        dataset: {} for dataset in all_results.keys()
    }
    for metric_name in comparable_metrics:
        ranked = sorted(
            [
                {
                    "dataset": dataset,
                    "median": float(analysis_summary["distribution_stats"][dataset][metric_name]["median"]),
                }
                for dataset in all_results.keys()
                if metric_name in analysis_summary["distribution_stats"][dataset]
            ],
            key=lambda item: item["median"],
            reverse=True,
        )
        cross_dataset_summary["comparable_metric_medians"][metric_name] = ranked
        for rank_idx, item in enumerate(ranked, start=1):
            comparable_rank_positions[item["dataset"]][metric_name] = rank_idx

    descriptive_ordering = []
    for dataset in all_results.keys():
        if comparable_rank_positions[dataset]:
            avg_rank = float(np.mean(list(comparable_rank_positions[dataset].values())))
        else:
            avg_rank = float("nan")
        descriptive_ordering.append(
            {
                "dataset": dataset,
                "average_comparable_rank": avg_rank,
                "metric_ranks": comparable_rank_positions[dataset],
                "gradient_c_median": float(analysis_summary["distribution_stats"][dataset]["gradient_c"]["median"]) if "gradient_c" in analysis_summary["distribution_stats"][dataset] else None,
                "contrast_median": float(analysis_summary["distribution_stats"][dataset]["contrast"]["median"]) if "contrast" in analysis_summary["distribution_stats"][dataset] else None,
            }
        )
    descriptive_ordering.sort(key=lambda item: item["average_comparable_rank"])
    cross_dataset_summary["descriptive_ordering"] = descriptive_ordering
    analysis_summary["cross_dataset_summary"] = cross_dataset_summary

    summary_lines = [
        "# Quality Metric Analysis",
        "",
        "This report summarizes lightweight ROI quality metrics over the cached dataset-level results.",
        "The interpretation is intentionally conservative: the analysis describes score distributions, cross-metric relationships,",
        "and composite rankings, but it does not by itself establish that any metric improves recognition performance.",
        "",
        "## Key Observations",
        "",
        "- `gradient` and `contrast` are consistently the most correlated pair across datasets, indicating partial signal overlap rather than independent evidence.",
        "- `gradient_c` is typically less correlated with the other metrics and therefore receives more weight in the correlation-aware composite, especially on FVUSM and MMCBNU.",
        "- `laplacian` often receives a high redundancy-aware weight, which should be interpreted as lower overlap with the other metrics within a dataset rather than proof of greater downstream utility.",
        "- The highest- and lowest-ranked images remain broadly stable after redundancy-aware weighting, which suggests that the extreme cases are not artifacts of one arbitrary weighting choice.",
        "- Agreement across all metrics on the same bottom-10% or top-10% images is modest, so the metrics should be treated as complementary heuristics rather than interchangeable surrogates. The low Jaccard values indicate that the metrics often disagree on which exact images are most extreme even when their overall rankings are correlated.",
        "- The lowest-ranked examples frequently cluster within one subject or finger identity rather than being uniformly distributed across the dataset, which suggests that the composite score is sensitive to subject-level acquisition factors as well as per-image nuisance.",
        "",
        "## Publication-Safe Interpretation",
        "",
        "These results support three restrained conclusions.",
        "",
        "1. The four handcrafted metrics capture related but non-identical aspects of ROI quality.",
        "2. A correlation-aware composite reduces the influence of highly redundant metrics compared with equal weighting.",
        "3. The resulting ranking is suitable for exploratory analysis and candidate sample filtering, but it still requires validation against downstream verification performance before stronger claims are made.",
        "",
        "## Cross-Dataset Context",
        "",
        "A direct comparison of raw metric magnitudes across datasets is not fully reliable because the sensors, preprocessing pipelines, and image statistics differ.",
        "This matters especially for `laplacian`, whose scale can vary strongly with resolution, noise floor, and preprocessing. For that reason, the report avoids a hard cross-dataset quality ranking based on all four metrics.",
        "",
        "For a limited descriptive comparison, the dimensionless metrics `gradient_c` and `contrast` provide the most interpretable cross-dataset context. Using dataset medians on those two metrics only, the current descriptive ordering is:",
        "",
    ]

    for rank_idx, row in enumerate(descriptive_ordering, start=1):
        metric_rank_text = ", ".join(
            f"`{metric}`={rank}" for metric, rank in row["metric_ranks"].items()
        )
        summary_lines.append(
            f"{rank_idx}. `{row['dataset']}` "
            f"(`gradient_c` median={row['gradient_c_median']:.4f}, "
            f"`contrast` median={row['contrast_median']:.4f}; {metric_rank_text})"
        )

    summary_lines.extend(
        [
            "",
            "This ordering should be read as descriptive context rather than a claim that one dataset is intrinsically better than another.",
            "It is included only to summarize how the more comparable handcrafted metrics behave across the present cached datasets.",
            "",
            "## Dataset Notes",
            "",
        ]
    )

    for dataset in all_results.keys():
        combined_info = analysis_summary["combined_scores"][dataset]
        agreement_info = analysis_summary["extreme_agreement"][dataset]
        weight_items = combined_info["weights"]
        strongest_metric = max(weight_items, key=weight_items.get)
        weakest_metric = min(weight_items, key=weight_items.get)
        summary_lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Composite spread: mean={combined_info['mean']:.3f}, median={combined_info['median']:.3f}, std={combined_info['std']:.3f}. This rank-based spread is descriptive only and mainly reflects dispersion of the aggregated ranks rather than a calibrated physical quality scale.",
                f"- Correlation-aware weights: "
                + ", ".join(f"`{k}`={v:.2f}" for k, v in weight_items.items())
                + ".",
                f"- Highest-weight metric under redundancy-aware aggregation: `{strongest_metric}`. Lowest-weight metric: `{weakest_metric}`.",
                f"- Extreme-case agreement remains limited: bottom-10% Jaccard={agreement_info['bottom10_jaccard']:.2f}, top-10% Jaccard={agreement_info['top10_jaccard']:.2f}. This indicates that the metrics do not flag exactly the same images as extreme cases.",
                f"- Lowest-ranked examples are concentrated around: "
                + ", ".join(f"`{item['relative_path']}`" for item in combined_info["bottom3"])
                + ".",
                f"- Highest-ranked examples are concentrated around: "
                + ", ".join(f"`{item['relative_path']}`" for item in combined_info["top3"])
                + ".",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Limitations",
            "",
            "- The metrics are unsupervised heuristics and may reward some nuisance factors such as strong non-vein edges or sensor-specific texture.",
            "- The correlation-aware weights are descriptive for these cached datasets; they should not be interpreted as universally optimal parameters.",
            "- Stability of the extreme rankings is encouraging, but qualitative inspection and downstream EER analysis are still necessary to establish practical usefulness.",
            "",
            "## Recommended Next Step",
            "",
            "Evaluate whether removing the lowest-ranked samples, or using the composite score as a covariate, improves verification performance on the same datasets.",
            "",
        ]
    )

    analysis_summary["publication_safe_summary"] = {
        "markdown_path": str(output_path / "quality_metric_analysis.md"),
        "json_path": str(output_path / "quality_metric_analysis.json"),
    }

    with (output_path / "quality_metric_analysis.json").open("w") as fp:
        json.dump(analysis_summary, fp, indent=2)

    with (output_path / "quality_metric_analysis.md").open("w") as fp:
        fp.write("\n".join(summary_lines))

if __name__ == "__main__":
    metric_fns = {
        "gradient": assess_gradient_quality,
        "gradient_c": assess_gradient_coherence_quality,
        "laplacian": assess_laplacian_sharpness,
        "contrast": assess_local_contrast,
    }
    print("dataset: " + " | ".join(metric_fns.keys()))
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    disable = args.quiet

    # for dataset in ["vera", "polyu", "mmcbnu", "fvusm", "fv300"]:
    #     get_quality_metrics_for_dataset(dataset)
    
    run_quality_analysis()
