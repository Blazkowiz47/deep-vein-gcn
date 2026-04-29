import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from scripts.unenrolled_eval import FAR_TARGETS_PERCENT, get_metric_summary
from utils import calculate_eer


RESULT_FILES = {
    "half": Path("ablation/half_subjects_results.json"),
    "full": Path("ablation/full_subjects_results.json"),
}

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
    "resnet": "Chen et al",
    "snakegraph2": "Proposed Method",
}

METHOD_ORDER = [
    "mcp",
    "rlt",
    "wld",
    "arcvein",
    "lgfin",
    "fv-vit",
    "veinAttNet",
    "resnet",
    "snakegraph2",
]

METRIC_ORDER = [
    ("auc", "AUC (%)"),
    ("eer", "EER (%)"),
    ("tar_far_0.1", "TAR@FAR=0.1% (%)"),
    ("tar_far_1", "TAR@FAR=1% (%)"),
    ("tar_far_10", "TAR@FAR=10% (%)"),
]

METRIC_DIRECTION = {
    "auc": "max",
    "eer": "min",
    "tar_far_0.1": "max",
    "tar_far_1": "max",
    "tar_far_10": "max",
}

BASELINE_METHODS = ("mcp", "rlt", "wld")


def load_json(path: Path) -> dict:
    with open(path, "r") as fp:
        return json.load(fp)


def z_value_95(df: int) -> float:
    t_lookup = {
        1: 12.706204736432095,
        2: 4.302652729696142,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636314,
        6: 2.4469118511449692,
        7: 2.3646242515927844,
        8: 2.306004135204166,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
    }
    return t_lookup.get(df, 1.959963984540054)


def mean_ci(values: Iterable[float]) -> tuple[float, float | None, float | None]:
    vals = np.asarray(list(values), dtype=float)
    mean = float(vals.mean())
    if vals.size <= 1:
        return mean, None, None

    sem = float(vals.std(ddof=1) / math.sqrt(vals.size))
    delta = z_value_95(vals.size - 1) * sem
    return mean, mean - delta, mean + delta


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def format_metric(values: List[float], scale: float = 1.0) -> str:
    scaled = [value * scale for value in values]
    mean, lo, hi = mean_ci(scaled)
    mean = clamp(mean)
    if lo is None or hi is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ({clamp(lo):.2f} - {clamp(hi):.2f})"


def ensure_auc_in_results(results: dict) -> dict:
    updated = False
    for dataset_methods in results.values():
        for method_seeds in dataset_methods.values():
            for metrics in method_seeds.values():
                if "auc" not in metrics:
                    raise ValueError(
                        "Missing 'auc' in result JSON. Run scripts/unenrolled_eval.py first."
                    )
                updated = True
    if not updated:
        raise ValueError("Result JSON is empty.")
    return results


def baseline_rate_paths(method: str, dataset: str) -> tuple[Path, Path]:
    return (
        Path(f"tmp/{method}/{dataset}/far_scores.npy"),
        Path(f"tmp/{method}/{dataset}/frr_scores.npy"),
    )


def baseline_score_paths(method: str, dataset: str) -> tuple[Path, Path]:
    return (
        Path(f"tmp/{method}/{dataset}/genuine.txt"),
        Path(f"tmp/{method}/{dataset}/imposter.txt"),
    )


def load_baseline_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASET_LABELS:
        results[dataset] = {}
        for method in BASELINE_METHODS:
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


def build_table(split_name: str, results: dict, baselines: dict) -> str:
    lines = [f"## {split_name.title()} Subjects", ""]
    headers = ["Train -> Test Dataset", "Algorithm"] + [
        label for _, label in METRIC_ORDER
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for dataset, dataset_label in DATASET_LABELS.items():
        rows = []
        for method in METHOD_ORDER:
            if method in BASELINE_METHODS:
                metric_dict = baselines.get(dataset, {}).get(method)
                if metric_dict is None:
                    continue
                rows.append((method, metric_dict, False))
                continue

            metric_dicts = results.get(dataset, {}).get(method, {})
            if not metric_dicts:
                continue
            rows.append((method, metric_dicts, True))

        if not rows:
            continue

        best_values = {}
        for metric_key, _ in METRIC_ORDER:
            metric_means = []
            for method, metric_source, use_ci in rows:
                if use_ci:
                    values = [
                        seed_metrics[metric_key]
                        for seed_metrics in metric_source.values()
                    ]
                    scale = 100.0 if metric_key.startswith("tar_") else 1.0
                    metric_means.append((method, float(np.mean(values) * scale)))
                else:
                    value = float(metric_source[metric_key])
                    if metric_key.startswith("tar_"):
                        value *= 100.0
                    metric_means.append((method, value))
            if METRIC_DIRECTION[metric_key] == "min":
                best_values[metric_key] = min(value for _, value in metric_means)
            else:
                best_values[metric_key] = max(value for _, value in metric_means)

        for idx, (method, metric_source, use_ci) in enumerate(rows):
            dataset_cell = dataset_label if idx == 0 else ""
            row = [dataset_cell, METHOD_LABELS[method]]
            for metric_key, _ in METRIC_ORDER:
                if use_ci:
                    values = [
                        seed_metrics[metric_key]
                        for seed_metrics in metric_source.values()
                    ]
                    scale = 100.0 if metric_key.startswith("tar_") else 1.0
                    formatted = format_metric(values, scale=scale)
                    current_value = float(np.mean(values) * scale)
                else:
                    current_value = float(metric_source[metric_key])
                    if metric_key.startswith("tar_"):
                        current_value *= 100.0
                    formatted = f"{current_value:.2f}"
                if np.isclose(current_value, best_values[metric_key], atol=1e-9):
                    formatted = f"**{formatted}**"
                row.append(formatted)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    return "\n".join(lines)


def main() -> None:
    half_results = ensure_auc_in_results(load_json(RESULT_FILES["half"]))
    full_results = ensure_auc_in_results(load_json(RESULT_FILES["full"]))
    baselines = load_baseline_metrics()

    content = [
        "# Final Tables",
        "",
        f"Generated from `{RESULT_FILES['half']}` and `{RESULT_FILES['full']}`.",
        f"TAR values are reported in percent at FAR targets {', '.join(f'{x:g}%' for x in FAR_TARGETS_PERCENT)}.",
        "",
        build_table("half", half_results, baselines),
        build_table("full", full_results, baselines),
    ]

    with open("final_tables.md", "w") as fp:
        fp.write("\n".join(content).rstrip() + "\n")


if __name__ == "__main__":
    main()
