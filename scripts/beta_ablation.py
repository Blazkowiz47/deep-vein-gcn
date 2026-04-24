import argparse
import copy
import json
from pathlib import Path

import yaml

from test import parallel_driver
from train import main as train_main


DEFAULT_CONFIG = Path("configs/dscgrapher2.yaml")
RESULTS_PATH = Path("./ablation/ablation_beta_runs.jsonl")
EERS_PATH = Path("./ablation/ablation_beta_eers.jsonl")
DEFAULT_DATASET = "fvusm"
DEFAULT_STAT_SEEDS = [0]
DEFAULT_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as fp:
        return yaml.safe_load(fp)


def load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as fp:
        records = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["key"]] = record["value"]
        return records


def append_jsonl(path: Path, key: str, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps({"key": key, "value": value}, sort_keys=True) + "\n")


def get_run_name_from_record(record) -> str:
    if isinstance(record, dict):
        return record["run_name"]
    return record


def beta_name(beta: float) -> str:
    return f"beta_{beta:.1f}".replace(".", "p")


def build_variant_config(config: dict, beta: float) -> dict:
    local_config = copy.deepcopy(config)
    local_config["loss"] = "proposed"
    local_config["beta"] = beta
    return local_config


def build_train_args(config_path: Path, dataset: str, stat_seed: int, wandb: bool):
    return argparse.Namespace(
        config=str(config_path),
        seed=stat_seed,
        leave=dataset,
        wandb=wandb,
        dataset="leaveoneout",
        model_name=None,
        logger_level="INFO",
        continue_model=None,
    )


def build_eval_args(config_path: Path, dataset: str, checkpoint: str):
    return argparse.Namespace(
        config=str(config_path),
        checkpoint=checkpoint,
        dataset=dataset,
        logger_level="ERROR",
        continue_model=None,
    )


def run_variant(args, beta: float, stat_seed: int) -> None:
    config_path = Path(args.config).resolve()
    base_config = load_config(config_path)
    variant_config = build_variant_config(base_config, beta)
    variant_config["seed"] = stat_seed
    variant_config["stat_seed"] = stat_seed

    variant_name = beta_name(beta)
    run_key = f"{args.dataset}:{stat_seed}:{variant_name}"
    runs = load_jsonl(RESULTS_PATH)
    if run_key in runs and not args.retrain:
        run_name = get_run_name_from_record(runs[run_key])
        print(f"Using existing run: {run_key} -> {run_name}")
    else:
        train_args = build_train_args(config_path, args.dataset, stat_seed, args.wandb)
        run_name = train_main(train_args, variant_config)
        append_jsonl(
            RESULTS_PATH,
            run_key,
            {
                "run_name": run_name,
                "wandb_run_name": run_name if args.wandb else None,
                "beta": beta,
            },
        )
        print(f"Saved run: {run_key} -> {run_name}")

    if args.skip_eval:
        return

    eers = load_jsonl(EERS_PATH)
    if run_key in eers and not args.reeval:
        print(f"Existing EER: {run_key} -> {eers[run_key]}")
        return

    eval_args = build_eval_args(config_path, args.dataset, run_name)
    eer = parallel_driver(eval_args, variant_config)
    append_jsonl(EERS_PATH, run_key, {"eer": eer, "beta": beta})
    print(f"Saved EER: {run_key} -> {eer}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file to use.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Leave-one-out dataset to train/evaluate against.",
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=DEFAULT_BETAS,
        help="Beta values to sweep for the proposed loss.",
    )
    parser.add_argument(
        "--stat-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_STAT_SEEDS,
        help="Stat seed values to run. Defaults to only seed 0.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging for training runs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train and record the run name.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a run is already recorded.",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Force reevaluation even if an EER is already recorded.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for beta in args.betas:
        for stat_seed in args.stat_seeds:
            print(
                f"Running beta ablation dataset={args.dataset} beta={beta:.1f} stat_seed={stat_seed}"
            )
            run_variant(args, beta, stat_seed)


if __name__ == "__main__":
    main()
