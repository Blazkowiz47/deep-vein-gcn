import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from scripts.unenrolled_eval import get_metric_summary
from utils import calculate_eer


RESULT_FILES = {
    "half": Path("ablation/half_subjects_results.json"),
    "full": Path("ablation/full_subjects_results.json"),
}

DATASET_ORDER = ["fv300", "mmcbnu", "fvusm", "polyu", "vera"]
DATASET_LABELS = {
    "fv300": "BCDE -> A",
    "mmcbnu": "ACDE -> B",
    "fvusm": "ABDE -> C",
    "polyu": "ABCE -> D",
    "vera": "ABCD -> E",
}

METHOD_LABELS = {
    "mcp": "MCP",
    "rlt": "RLT",
    "wld": "WLD",
    "arcvein": "ArcVein",
    "lgfin": "LGFIN",
    "fv-vit": "FV-ViT",
    "veinAttNet": "VeinAttNet",
    "chen": "Chen et al",
    "snakegraph2": "Proposed Method",
}

PROPOSED_METHOD = "snakegraph2"
VEIN_ATTNET = "veinAttNet"
HANDCRAFTED_METHODS = ("mcp", "rlt", "wld")
LEARNED_BASELINES = ("arcvein", "lgfin", "fv-vit", "veinAttNet", "chen")
FIXED_LEARNED_COMPARATORS = (
    "arcvein",
    "lgfin",
    "fv-vit",
    "chen",
    "veinAttNet",
)
METRICS = [
    ("eer", "EER"),
    ("auc", "AUC"),
    ("tar_far_0.1", "TAR@FAR=0.1%"),
    ("tar_far_1", "TAR@FAR=1%"),
    ("tar_far_10", "TAR@FAR=10%"),
]
COMPACT_ROWS = (
    ("Proposed vs ArcVein", "seed"),
    ("Proposed vs LGFIN", "seed"),
    ("Proposed vs FV-ViT", "seed"),
    ("Proposed vs Chen et al", "seed"),
    ("Proposed vs VeinAttNet", "seed"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired significance tests for leave-one-dataset-out rebuttal results."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["full"],
        choices=sorted(RESULT_FILES.keys()),
        help="Result splits to test.",
    )
    parser.add_argument(
        "--output-json",
        default="ablation/significance_tests.json",
        help="Structured output path.",
    )
    parser.add_argument(
        "--output-md",
        default="rebuttal/significance_tests.md",
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Adjusted significance threshold.",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Include the long per-metric markdown report instead of the compact rebuttal table.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def baseline_rate_paths(method: str, dataset: str) -> Tuple[Path, Path]:
    return (
        Path(f"tmp/{method}/{dataset}/far_scores.npy"),
        Path(f"tmp/{method}/{dataset}/frr_scores.npy"),
    )


def baseline_score_paths(method: str, dataset: str) -> Tuple[Path, Path]:
    return (
        Path(f"tmp/{method}/{dataset}/genuine.txt"),
        Path(f"tmp/{method}/{dataset}/imposter.txt"),
    )


def load_baseline_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASET_ORDER:
        results[dataset] = {}
        for method in HANDCRAFTED_METHODS:
            far_path, frr_path = baseline_rate_paths(method, dataset)
            if far_path.exists() and frr_path.exists():
                far = np.load(far_path)
                frr = np.load(frr_path)
                results[dataset][method] = get_metric_summary(far, frr)
                continue

            genuine_path, imposter_path = baseline_score_paths(method, dataset)
            if not genuine_path.exists() or not imposter_path.exists():
                continue

            genuine = np.loadtxt(genuine_path).reshape(-1)
            imposter = np.loadtxt(imposter_path).reshape(-1)
            _, far, frr, _ = calculate_eer(genuine.tolist(), imposter.tolist())
            results[dataset][method] = get_metric_summary(far, frr)
    return results


def metric_scale(metric_key: str) -> float:
    return 100.0 if metric_key.startswith("tar_") else 1.0


def paired_improvement(proposed_value: float, comparator_value: float, metric_key: str) -> float:
    scale = metric_scale(metric_key)
    if metric_key == "eer":
        return (comparator_value - proposed_value) * scale
    return (proposed_value - comparator_value) * scale


def exact_sign_flip_pvalue(deltas: Iterable[float]) -> float:
    vals = np.asarray(list(deltas), dtype=float)
    if vals.size == 0:
        return float("nan")

    sums = np.array([0.0], dtype=np.float64)
    for delta in vals:
        sums = np.concatenate((sums + delta, sums - delta))

    observed = abs(float(vals.sum()))
    return float(np.mean(np.abs(sums) >= observed - 1e-12))


def holm_adjust(records: List[dict]) -> None:
    if not records:
        return

    order = sorted(range(len(records)), key=lambda idx: records[idx]["p_value"])
    adjusted = [0.0] * len(records)
    running_max = 0.0
    m = len(records)
    for rank, idx in enumerate(order):
        raw = records[idx]["p_value"]
        value = min(1.0, raw * (m - rank))
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    for idx, value in enumerate(adjusted):
        records[idx]["holm_p_value"] = value


def mean_seed_metric(method_seed_metrics: dict, metric_key: str) -> float:
    values = [seed_metrics[metric_key] for seed_metrics in method_seed_metrics.values()]
    return float(np.mean(values))


def select_best_learned_baselines(results: dict) -> Dict[str, str]:
    best_methods: Dict[str, str] = {}
    for dataset in DATASET_ORDER:
        best_method = None
        best_eer = None
        for method in LEARNED_BASELINES:
            method_results = results.get(dataset, {}).get(method, {})
            if not method_results:
                continue
            mean_eer = mean_seed_metric(method_results, "eer")
            if best_eer is None or mean_eer < best_eer:
                best_eer = mean_eer
                best_method = method
        if best_method is None:
            raise ValueError(f"No learned baselines found for dataset={dataset}")
        best_methods[dataset] = best_method
    return best_methods


def select_best_handcrafted_baselines(baselines: dict) -> Dict[str, str]:
    best_methods: Dict[str, str] = {}
    for dataset in DATASET_ORDER:
        best_method = None
        best_eer = None
        for method in HANDCRAFTED_METHODS:
            metrics = baselines.get(dataset, {}).get(method)
            if metrics is None:
                continue
            eer = float(metrics["eer"])
            if best_eer is None or eer < best_eer:
                best_eer = eer
                best_method = method
        if best_method is None:
            raise ValueError(f"No handcrafted baselines found for dataset={dataset}")
        best_methods[dataset] = best_method
    return best_methods


def seed_level_deltas_fixed(results: dict, comparator_method: str, metric_key: str) -> List[float]:
    deltas: List[float] = []
    for dataset in DATASET_ORDER:
        proposed = results.get(dataset, {}).get(PROPOSED_METHOD, {})
        comparator = results.get(dataset, {}).get(comparator_method, {})
        common_seeds = sorted(set(proposed) & set(comparator), key=int)
        for seed in common_seeds:
            deltas.append(
                paired_improvement(
                    proposed[seed][metric_key], comparator[seed][metric_key], metric_key
                )
            )
    return deltas


def seed_level_deltas_dataset_specific(
    results: dict, comparator_by_dataset: Dict[str, str], metric_key: str
) -> List[float]:
    deltas: List[float] = []
    for dataset in DATASET_ORDER:
        comparator_method = comparator_by_dataset[dataset]
        proposed = results.get(dataset, {}).get(PROPOSED_METHOD, {})
        comparator = results.get(dataset, {}).get(comparator_method, {})
        common_seeds = sorted(set(proposed) & set(comparator), key=int)
        for seed in common_seeds:
            deltas.append(
                paired_improvement(
                    proposed[seed][metric_key], comparator[seed][metric_key], metric_key
                )
            )
    return deltas


def split_level_deltas_fixed(results: dict, comparator_method: str, metric_key: str) -> List[float]:
    deltas: List[float] = []
    for dataset in DATASET_ORDER:
        proposed = results.get(dataset, {}).get(PROPOSED_METHOD, {})
        comparator = results.get(dataset, {}).get(comparator_method, {})
        if not proposed or not comparator:
            continue
        deltas.append(
            paired_improvement(
                mean_seed_metric(proposed, metric_key),
                mean_seed_metric(comparator, metric_key),
                metric_key,
            )
        )
    return deltas


def split_level_deltas_dataset_specific(
    results: dict, comparator_by_dataset: Dict[str, str], metric_key: str
) -> List[float]:
    deltas: List[float] = []
    for dataset in DATASET_ORDER:
        comparator_method = comparator_by_dataset[dataset]
        proposed = results.get(dataset, {}).get(PROPOSED_METHOD, {})
        comparator = results.get(dataset, {}).get(comparator_method, {})
        if not proposed or not comparator:
            continue
        deltas.append(
            paired_improvement(
                mean_seed_metric(proposed, metric_key),
                mean_seed_metric(comparator, metric_key),
                metric_key,
            )
        )
    return deltas


def split_level_deltas_handcrafted(
    results: dict, baselines: dict, comparator_by_dataset: Dict[str, str], metric_key: str
) -> List[float]:
    deltas: List[float] = []
    for dataset in DATASET_ORDER:
        comparator_method = comparator_by_dataset[dataset]
        proposed = results.get(dataset, {}).get(PROPOSED_METHOD, {})
        comparator = baselines.get(dataset, {}).get(comparator_method)
        if not proposed or comparator is None:
            continue
        deltas.append(
            paired_improvement(
                mean_seed_metric(proposed, metric_key), comparator[metric_key], metric_key
            )
        )
    return deltas


def build_record(
    split_name: str,
    comparison_key: str,
    comparison_label: str,
    level: str,
    metric_key: str,
    metric_label: str,
    deltas: List[float],
) -> dict:
    p_value = exact_sign_flip_pvalue(deltas)
    return {
        "split": split_name,
        "comparison_key": comparison_key,
        "comparison": comparison_label,
        "level": level,
        "metric_key": metric_key,
        "metric": metric_label,
        "n_pairs": len(deltas),
        "mean_delta_pp": float(np.mean(deltas)) if deltas else float("nan"),
        "median_delta_pp": float(np.median(deltas)) if deltas else float("nan"),
        "p_value": p_value,
    }


def dataset_mapping_lines(title: str, mapping: Dict[str, str]) -> List[str]:
    lines = [f"- {title}"]
    for dataset in DATASET_ORDER:
        method = mapping[dataset]
        lines.append(f"  - `{DATASET_LABELS[dataset]}`: `{METHOD_LABELS[method]}`")
    return lines


def format_p(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_delta(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.2f}"


def compact_baseline_note(title: str, mapping: Dict[str, str]) -> str:
    parts = [f"{DATASET_LABELS[dataset]}={METHOD_LABELS[mapping[dataset]]}" for dataset in DATASET_ORDER]
    return f"- {title} " + ", ".join(parts) + "."


def build_compact_markdown(summary: dict, output_json: str, alpha: float) -> str:
    lines = [
        "# Significance Tests",
        "",
        "Two-sided exact paired sign-flip permutation tests on EER only.",
        "Positive mean delta favors Proposed Method.",
        f"Holm-adjusted significance threshold: `{alpha:.2f}`.",
        f"Structured output: `{output_json}`.",
        "",
        "| Split | Comparison | n | Mean Delta EER (pp) | Holm p-value | Significant |",
        "|---|---|---:|---:|---:|---|",
    ]

    for split_name in summary["splits"]:
        split_payload = summary["splits"][split_name]
        eer_records = [
            record
            for record in split_payload["records"]
            if record["metric_key"] == "eer"
        ]
        for comparison, level in COMPACT_ROWS:
            record = next(
                item
                for item in eer_records
                if item["comparison"] == comparison and item["level"] == level
            )
            significant = "yes" if record["holm_p_value"] < alpha else "no"
            label = f"{comparison} (seed-matched)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name.title(),
                        label,
                        str(record["n_pairs"]),
                        format_delta(record["mean_delta_pp"]),
                        format_p(record["holm_p_value"]),
                        significant,
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_markdown(
    summary: dict, output_json: str, alpha: float
) -> str:
    lines = [
        "# Significance Tests",
        "",
        "Two-sided exact paired sign-flip permutation tests.",
        "Positive mean delta favors Proposed Method.",
        f"Holm-adjusted significance threshold: `{alpha:.2f}`.",
        f"Structured output: `{output_json}`.",
        "",
    ]

    for split_name in summary["splits"]:
        split_payload = summary["splits"][split_name]
        lines.append(f"## {split_name.title()} Subjects")
        lines.append("")
        lines.append("### EER Summary")
        lines.append("")
        lines.append(
            "| Comparison | Level | n | Mean Delta (pp) | p-value | Holm p-value | Significant |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for record in split_payload["records"]:
            if record["metric_key"] != "eer":
                continue
            significant = "yes" if record["holm_p_value"] < alpha else "no"
            lines.append(
                "| "
                + " | ".join(
                    [
                        record["comparison"],
                        record["level"],
                        str(record["n_pairs"]),
                        format_delta(record["mean_delta_pp"]),
                        format_p(record["p_value"]),
                        format_p(record["holm_p_value"]),
                        significant,
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("### All Metrics")
        lines.append("")
        lines.append(
            "| Comparison | Level | Metric | n | Mean Delta (pp) | p-value | Holm p-value | Significant |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---|")
        for record in split_payload["records"]:
            significant = "yes" if record["holm_p_value"] < alpha else "no"
            lines.append(
                "| "
                + " | ".join(
                    [
                        record["comparison"],
                        record["level"],
                        record["metric"],
                        str(record["n_pairs"]),
                        format_delta(record["mean_delta_pp"]),
                        format_p(record["p_value"]),
                        format_p(record["holm_p_value"]),
                        significant,
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    summary = {
        "test": "two-sided exact paired sign-flip permutation test",
        "alpha": args.alpha,
        "splits": {},
    }

    for split_name in args.splits:
        results = load_json(RESULT_FILES[split_name])

        records: List[dict] = []
        metrics_to_run = METRICS if args.full_report else [("eer", "EER")]
        levels_to_run = ("seed", "split") if args.full_report else ("seed",)
        for metric_key, metric_label in metrics_to_run:
            for comparator_method in FIXED_LEARNED_COMPARATORS:
                label = f"Proposed vs {METHOD_LABELS[comparator_method]}"
                if "seed" in levels_to_run:
                    records.append(
                        build_record(
                            split_name,
                            f"{comparator_method}_seed_level",
                            label,
                            "seed",
                            metric_key,
                            metric_label,
                            seed_level_deltas_fixed(results, comparator_method, metric_key),
                        )
                    )
                if "split" in levels_to_run:
                    records.append(
                        build_record(
                            split_name,
                            f"{comparator_method}_split_level",
                            label,
                            "split",
                            metric_key,
                            metric_label,
                            split_level_deltas_fixed(results, comparator_method, metric_key),
                        )
                    )

        holm_adjust(records)
        summary["splits"][split_name] = {
            "records": records,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.full_report:
        markdown = build_markdown(summary, args.output_json, args.alpha)
    else:
        markdown = build_compact_markdown(summary, args.output_json, args.alpha)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
