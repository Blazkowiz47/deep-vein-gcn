import argparse
import copy
import json
from pathlib import Path

import yaml

from test import parallel_driver
from train import main as train_main


DEFAULT_CONFIG = Path("configs/dscgrapher2.yaml")
RESULTS_PATH = Path("./ablation/ablation_component_runs.jsonl")
EERS_PATH = Path("./ablation/ablation_component_eers.jsonl")
DATASET = "fvusm"
STAT_SEEDS = [0, 1, 2, 3, 4]


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
    with open(path, "a") as fp:
        fp.write(json.dumps({"key": key, "value": value}, sort_keys=True) + "\n")


def get_run_name_from_record(record) -> str:
    if isinstance(record, dict):
        return record["run_name"]
    return record


def ablation_name(no_gcn: bool, no_dsc: bool) -> str:
    disabled = []
    if no_gcn:
        disabled.append("gcn")
    if no_dsc:
        disabled.append("dsc")
    return "no_" + "_".join(disabled)


def build_variant_config(config: dict, no_gcn: bool, no_dsc: bool) -> dict:
    local_config = copy.deepcopy(config)
    local_config["switch_gcn"] = no_gcn
    local_config["switch_dsc"] = no_dsc
    return local_config


def build_train_args(config_path: Path, stat_seed: int, wandb: bool):
    return argparse.Namespace(
        config=str(config_path),
        seed=stat_seed,
        leave=DATASET,
        wandb=wandb,
        dataset="leaveoneout",
        model_name=None,
        logger_level="INFO",
        continue_model=None,
    )


def build_eval_args(config_path: Path, checkpoint: str):
    return argparse.Namespace(
        config=str(config_path),
        checkpoint=checkpoint,
        dataset=DATASET,
        logger_level="ERROR",
        continue_model=None,
    )


def run_variant(args, stat_seed: int) -> None:
    config_path = Path(args.config).resolve()
    base_config = load_config(config_path)
    variant_config = build_variant_config(base_config, args.no_gcn, args.no_dsc)
    variant_config["seed"] = stat_seed
    variant_config["stat_seed"] = stat_seed

    variant_name = ablation_name(args.no_gcn, args.no_dsc)
    run_key = f"{DATASET}:{stat_seed}:{variant_name}"
    runs = load_jsonl(RESULTS_PATH)
    if run_key in runs and not args.retrain:
        run_name = get_run_name_from_record(runs[run_key])
        print(f"Using existing run: {run_key} -> {run_name}")
    else:
        train_args = build_train_args(config_path, stat_seed, args.wandb)
        run_name = train_main(train_args, variant_config)
        append_jsonl(
            RESULTS_PATH,
            run_key,
            {
                "run_name": run_name,
                "wandb_run_name": run_name if args.wandb else None,
            },
        )
        print(f"Saved run: {run_key} -> {run_name}")

    if args.skip_eval:
        return

    eers = load_jsonl(EERS_PATH)
    if run_key in eers and not args.reeval:
        print(f"Existing EER: {run_key} -> {eers[run_key]}")
        return

    eval_args = build_eval_args(config_path, run_name)
    eer = parallel_driver(eval_args, variant_config)
    append_jsonl(EERS_PATH, run_key, eer)
    print(f"Saved EER: {run_key} -> {eer}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file to use.",
    )
    parser.add_argument(
        "--no-gcn",
        action="store_true",
        help="Disable the GCN grapher blocks.",
    )
    parser.add_argument(
        "--no-dsc",
        action="store_true",
        help="Disable the DSC stem and replace it with standard convolutions.",
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
    args = parser.parse_args()
    if not args.no_gcn and not args.no_dsc:
        parser.error("Specify at least one ablation flag: --no-gcn, --no-dsc, or both.")
    return args


def main():
    args = parse_args()
    variant_name = ablation_name(args.no_gcn, args.no_dsc)
    for stat_seed in STAT_SEEDS:
        print(
            f"Running ablation={variant_name} dataset={DATASET} stat_seed={stat_seed}"
        )
        run_variant(args, stat_seed)


if __name__ == "__main__":
    main()
