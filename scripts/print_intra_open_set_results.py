import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


RESULTS_PATH = Path("ablation/intra_open_set_results.jsonl")
EXPECTED_SEEDS = {0, 1, 2, 3, 4}
DATASET_ORDER = ["fv300", "fvusm", "mmcbnu"]
METHOD_ORDER = ["arcvein", "lgfin", "fvit", "veinAttNet", "chen", "proposed"]
METRIC_SPECS = [
    ("auc", False),
    ("eer", False),
    ("tar_far_0.1", True),
    ("tar_far_1", True),
    ("tar_far_10", True),
]
METRIC_DIRECTION = {
    "auc": "max",
    "eer": "min",
    "tar_far_0.1": "max",
    "tar_far_1": "max",
    "tar_far_10": "max",
}
DATASET_LABELS = {
    "fv300": "FV-300",
    "fvusm": "FV-USM",
    "mmcbnu": "MMCBNU",
}
METHOD_LABELS = {
    "proposed": "Proposed Method",
    "lgfin": "LGFIN",
    "fvit": "FV-ViT",
    "arcvein": "ArcVein",
    "veinAttNet": "VeinAttNet",
    "chen": "Chen et al",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print intra open-set results with 95% CI for completed 5-seed groups."
    )
    parser.add_argument(
        "--results-path",
        default=str(RESULTS_PATH),
        help="Path to intra open-set results JSONL.",
    )
    parser.add_argument(
        "--caption",
        default="Intra-database open-set results with 95\\% confidence intervals over five statistical seeds.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:intra_open_set_results",
        help="LaTeX table label.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["key"]] = record["value"]
    return records


def t_critical_95(df: int) -> float:
    table = {
        1: 12.7062047364,
        2: 4.30265272975,
        3: 3.18244630528,
        4: 2.7764451052,
        5: 2.57058183564,
        6: 2.44691185114,
        7: 2.36462425101,
        8: 2.30600413503,
        9: 2.26215716285,
        10: 2.22813885196,
    }
    return table[df]


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        return values[0], values[0], values[0]
    avg = mean(values)
    sd = stdev(values)
    half = t_critical_95(len(values) - 1) * sd / math.sqrt(len(values))
    return avg, avg - half, avg + half


def format_metric(values: list[float], as_percent: bool) -> str:
    avg, lo, hi = mean_ci(values)
    scale = 100.0 if as_percent else 1.0
    lo = max(0.0, lo)
    hi = min(1.0 if as_percent else 100.0, hi)
    return f"{avg * scale:.2f} ({lo * scale:.2f}--{hi * scale:.2f})"


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    if dataset in DATASET_ORDER:
        return (DATASET_ORDER.index(dataset), dataset)
    return (len(DATASET_ORDER), dataset)


def method_sort_key(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def display_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def scaled_mean(values: list[float], as_percent: bool) -> float:
    scale = 100.0 if as_percent else 1.0
    return mean(values) * scale


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.results_path))

    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, value in records.items():
        dataset, seed_str, method = key.split(":")
        if method == 'resnet':
            continue
        grouped[(dataset, method)][int(seed_str)] = value

    dataset_rows: dict[str, list[dict]] = defaultdict(list)

    for dataset, method in sorted(
        grouped.keys(), key=lambda x: (dataset_sort_key(x[0]), method_sort_key(x[1]))
    ):
        seed_map = grouped[(dataset, method)]
        if set(seed_map.keys()) != EXPECTED_SEEDS:
            continue

        metric_values = {
            metric_key: [seed_map[seed][metric_key] for seed in sorted(EXPECTED_SEEDS)]
            for metric_key, _ in METRIC_SPECS
        }
        dataset_rows[dataset].append({"method": method, "metrics": metric_values})
    ordered_datasets = [
        dataset
        for dataset in sorted(dataset_rows.keys(), key=dataset_sort_key)
        if dataset_rows[dataset]
    ]
    if not ordered_datasets:
        print("% No complete 5-seed result groups found.")
        return

    print(r"\begin{table*}[ht!]")
    print(r"    \centering")
    print(r"    \scriptsize")
    print(r"    \resizebox{\textwidth}{!}{%")
    print(r"    \begin{tabular}{|c|c|c|c|c|c|c|}")
    print(r"        \hline")
    print(
        r"        Dataset & Algorithm & AUC (\%) & EER (\%) & TAR@FAR$=0.1\%$ (\%) & TAR@FAR$=1\%$ (\%) & TAR@FAR$=10\%$ (\%) \\"
    )
    print(r"        \hline")
    for dataset in ordered_datasets:
        method_rows = dataset_rows[dataset]
        best_values = {}
        for metric_key, as_percent in METRIC_SPECS:
            metric_means = [
                scaled_mean(row["metrics"][metric_key], as_percent)
                for row in method_rows
            ]
            if METRIC_DIRECTION[metric_key] == "min":
                best_values[metric_key] = min(metric_means)
            else:
                best_values[metric_key] = max(metric_means)

        dataset_cell = (
            rf"\multirow{{{len(method_rows)}}}{{*}}{{{display_dataset(dataset)}}}"
        )
        for row_idx, row in enumerate(method_rows):
            prefix = dataset_cell if row_idx == 0 else ""
            cells = [display_method(row["method"])]
            for metric_key, as_percent in METRIC_SPECS:
                values = row["metrics"][metric_key]
                formatted = format_metric(values, as_percent=as_percent)
                current_value = scaled_mean(values, as_percent)
                if math.isclose(
                    current_value, best_values[metric_key], rel_tol=0.0, abs_tol=1e-9
                ):
                    formatted = rf"\textbf{{{formatted}}}"
                cells.append(formatted)
            print(f"        {prefix} & " + " & ".join(cells) + r" \\")
        print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    }")
    print(f"    \\caption{{{args.caption}}}")
    print(f"    \\label{{{args.label}}}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
