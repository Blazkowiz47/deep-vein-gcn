import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from intra_open_set_eval import evaluate


DEFAULT_DATASETS = ["fv300", "fvusm", "mmcbnu"]
DEFAULT_STAT_SEEDS = [0, 1, 2, 3, 4]
RUNS_PATH = Path("ablation/intra_open_set_runs.jsonl")
RESULTS_PATH = Path("ablation/intra_open_set_results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate completed intra-database VeinAttNet feature exports and update JSONL summaries."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--stat-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_STAT_SEEDS,
        help="Stat seeds to evaluate.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Training identity fraction used by the intra split.",
    )
    parser.add_argument(
        "--config",
        default="configs/veinattnet.yaml",
        help="VeinAttNet config file.",
    )
    parser.add_argument(
        "--logger-level",
        default="ERROR",
        help="Logger level for the evaluator.",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Force reevaluation even if a JSONL result already exists.",
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


def upsert_jsonl(path: Path, key: str, value) -> None:
    records = load_jsonl(path)
    records[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record_key in sorted(records.keys()):
            fp.write(
                json.dumps(
                    {"key": record_key, "value": records[record_key]}, sort_keys=True
                )
                + "\n"
            )


def checkpoint_name(dataset: str, stat_seed: int) -> str:
    return f"veinAttNet_intra_{dataset}_seed_{stat_seed}"


def feature_dir(dataset: str, stat_seed: int) -> Path:
    return Path("features") / checkpoint_name(dataset, stat_seed)


def checkpoint_path(dataset: str, stat_seed: int) -> Path:
    return (
        Path("tmp")
        / checkpoint_name(dataset, stat_seed)
        / "checkpoints"
        / "best_model.mat"
    )


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    runs = load_jsonl(RUNS_PATH)
    results = load_jsonl(RESULTS_PATH)

    for dataset in args.datasets:
        for stat_seed in args.stat_seeds:
            run_name = checkpoint_name(dataset, stat_seed)
            run_key = f"{dataset}:{stat_seed}:veinAttNet"
            features_path = feature_dir(dataset, stat_seed)
            ckpt_path = checkpoint_path(dataset, stat_seed)

            if not features_path.exists():
                print(
                    f"Skipping {run_key}: features not found at {features_path}"
                )
                continue

            if not ckpt_path.exists():
                print(
                    f"Skipping {run_key}: checkpoint not found at {ckpt_path}"
                )
                continue

            if run_key in results and not args.reeval:
                print(f"Existing results: {run_key} -> {results[run_key]}")
                if run_key not in runs:
                    upsert_jsonl(
                        RUNS_PATH,
                        run_key,
                        {
                            "dataset": dataset,
                            "method": "veinAttNet",
                            "run_name": run_name,
                            "stat_seed": stat_seed,
                        },
                    )
                continue

            eval_args = SimpleNamespace(
                config=args.config,
                checkpoint=run_name,
                dataset=dataset,
                method="veinAttNet",
                stat_seed=stat_seed,
                partition_split=args.partition_split,
                batch_size=128,
                logger_level=args.logger_level,
            )
            summary = evaluate(eval_args, dict(config))
            upsert_jsonl(
                RUNS_PATH,
                run_key,
                {
                    "dataset": dataset,
                    "method": "veinAttNet",
                    "run_name": run_name,
                    "stat_seed": stat_seed,
                },
            )
            upsert_jsonl(RESULTS_PATH, run_key, summary)
            print(f"Saved results: {run_key} -> {summary}")


if __name__ == "__main__":
    main()
