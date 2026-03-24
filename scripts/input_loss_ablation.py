import argparse
import copy
import json
from pathlib import Path

import yaml

from test import parallel_driver
from train import main as train_main


DEFAULT_CONFIG = Path("configs/dscgrapher2.yaml")
RESULTS_PATH = Path("./ablation/ablation_input_loss_runs.jsonl")
EERS_PATH = Path("./ablation/ablation_input_loss_eers.jsonl")
LOSS_MAP = {
    "arcface": "arcface",
    "adaface": "adaface",
    "adafaceq": "adaface_q",
    "magface": "magface",
    "proposed": "proposed",
    "qualityaware": "qualityawareproposed",
    "crossentropy": "crossentropy",
}
MODES = [None]


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


def build_variant_config(config: dict, loss_name: str, mode: str | None) -> dict:
    local_config = copy.deepcopy(config)
    local_config["loss"] = LOSS_MAP[loss_name]
    local_config["mode"] = mode
    return local_config


def build_train_args(config_path: Path, dataset: str, seed: int, wandb: bool):
    return argparse.Namespace(
        config=str(config_path),
        seed=seed,
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


def mode_key(mode: str | None) -> str:
    return "none" if mode is None else mode


def run_variant(args, mode: str | None) -> None:
    config_path = Path(args.config).resolve()
    base_config = load_config(config_path)
    variant_config = build_variant_config(base_config, args.loss, mode)
    variant_config["seed"] = args.seed
    variant_config["stat_seed"] = args.seed

    run_key = f"{args.dataset}:{args.seed}:{args.loss}:{mode_key(mode)}"
    runs = load_jsonl(RESULTS_PATH)
    if run_key in runs and not args.retrain:
        run_name = get_run_name_from_record(runs[run_key])
        print(f"Using existing run: {run_key} -> {run_name}")
    else:
        train_args = build_train_args(config_path, args.dataset, args.seed, args.wandb)
        run_name = train_main(train_args, variant_config)
        append_jsonl(
            RESULTS_PATH,
            run_key,
                run_name,
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
        "--loss",
        required=True,
        choices=sorted(LOSS_MAP.keys()),
        help="Loss to run across all input modes.",
    )
    parser.add_argument(
        "--dataset",
        default="fvusm",
        help="Held-out dataset for leave-one-out training and evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Statistical seed / folder index.",
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
    for mode in MODES:
        print(f"Running loss={args.loss} mode={mode_key(mode)}")
        run_variant(args, mode)


if __name__ == "__main__":
    main()
