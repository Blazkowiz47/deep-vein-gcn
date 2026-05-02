import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from run_name_mappings import final_runs


ABLATION_EERS_PATH = Path("ablation/ablation_input_loss_eers.jsonl")
FULL_RESULTS_PATH = Path("ablation/full_subjects_results.json")
DATASET = "fvusm"
PROPOSED_METHOD = "snakegraph2"
COMMON_SEEDS = [0, 1, 2, 3]
LOSS_ORDER = ["arcface", "cosface", "magface", "adaface"]
LOSS_LABELS = {
    "arcface": "ArcFace",
    "cosface": "CosFace",
    "magface": "MagFace",
    "adaface": "AdaFace",
    "proposed": "Proposed CAH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table for the input-loss ablation with Wilcoxon tests."
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help="Dataset key to use. Defaults to fvusm.",
    )
    parser.add_argument(
        "--ablation-eers",
        default=str(ABLATION_EERS_PATH),
        help="Input ablation EER JSONL path.",
    )
    parser.add_argument(
        "--full-results",
        default=str(FULL_RESULTS_PATH),
        help="Full-subject results JSON path for the official proposed runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        records = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["key"]] = record["value"]
        return records


def format_p_value(p_value: float) -> str:
    if p_value < 1e-4:
        return "$<10^{-4}$"
    return f"{p_value:.4f}"


def get_proposed_seed_eers(dataset: str, full_results: dict) -> list[float]:
    if dataset not in final_runs or PROPOSED_METHOD not in final_runs[dataset]:
        raise ValueError(f"No official proposed runs found for dataset={dataset}")

    method_results = full_results.get(dataset, {}).get(PROPOSED_METHOD, {})
    missing = [seed for seed in COMMON_SEEDS if str(seed) not in method_results]
    if missing:
        raise ValueError(
            f"Missing official proposed EERs for dataset={dataset}, seeds={missing}"
        )

    return [float(method_results[str(seed)]["eer"]) for seed in COMMON_SEEDS]


def get_loss_seed_eers(
    dataset: str, ablation_eers: dict, loss_name: str
) -> list[float] | None:
    values = []
    for seed in COMMON_SEEDS:
        key = f"{dataset}:{seed}:{loss_name}:none"
        if key not in ablation_eers:
            return None
        values.append(float(ablation_eers[key]))
    return values


def build_rows(dataset: str, ablation_eers: dict, full_results: dict):
    proposed = get_proposed_seed_eers(dataset, full_results)
    rows = []
    skipped_losses = []
    for loss_name in LOSS_ORDER:
        comparator = get_loss_seed_eers(dataset, ablation_eers, loss_name)
        if comparator is None:
            skipped_losses.append(loss_name)
            continue
        _stat, p_value = wilcoxon(comparator, proposed, alternative="greater")
        rows.append(
            {
                "label": LOSS_LABELS[loss_name],
                "values": comparator,
                "mean": float(np.mean(comparator)),
                "p_value": float(p_value),
            }
        )

    rows.append(
        {
            "label": LOSS_LABELS["proposed"],
            "values": proposed,
            "mean": float(np.mean(proposed)),
            "p_value": None,
        }
    )
    return rows, skipped_losses


def print_latex_table(rows, dataset: str, skipped_losses: list[str]) -> None:
    if skipped_losses:
        skipped = ", ".join(LOSS_LABELS.get(loss, loss) for loss in skipped_losses)
        print(
            rf"% Skipped incomplete losses for {dataset}: {skipped} (missing one or more of seeds 0--3)."
        )
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Loss Function & Mean EER (\%) & Wilcoxon $p$ vs. CAH \\")
    print(r"\hline")
    for row in rows:
        mean_cell = f"{row['mean']:.2f}"
        if row["label"] == LOSS_LABELS["proposed"]:
            print(rf"\textbf{{{row['label']}}} & \textbf{{{mean_cell}}} & -- \\")
        else:
            print(
                rf"{row['label']} & {mean_cell} & {format_p_value(row['p_value'])} \\"
            )
    print(r"\hline")
    print(r"\end{tabular}")
    print(
        rf"\caption{{Loss comparison on the $ABDE \rightarrow C$ protocol ({dataset.upper()}) using the first four matched seeds only. The proposed CAH row uses the official leave-one-out proposed runs from \texttt{{run\_name\_mappings.py}} and \texttt{{ablation/full\_subjects\_results.json}}. One-sided Wilcoxon signed-rank tests use the alternative hypothesis that the comparator yields higher EER than CAH.}}"
    )
    print(r"\label{tab:input_loss_wilcoxon}")
    print(r"\end{table}")
    print()
    print("% Seed-wise EERs used for the test")
    for row in rows:
        values = ", ".join(f"{value:.4f}" for value in row["values"])
        print(rf"% {row['label']}: [{values}]")


def main():
    args = parse_args()
    ablation_eers = load_jsonl(Path(args.ablation_eers))
    full_results = load_json(Path(args.full_results))
    rows, skipped_losses = build_rows(args.dataset, ablation_eers, full_results)
    print_latex_table(rows, args.dataset, skipped_losses)


if __name__ == "__main__":
    main()
