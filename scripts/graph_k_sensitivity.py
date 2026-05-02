import argparse
import copy
import json
from pathlib import Path

import yaml

from test import parallel_driver
from train import main as train_main


DEFAULT_CONFIG = Path("configs/dscgrapher2.yaml")
RESULTS_PATH = Path("./ablation/ablation_graph_k_runs.jsonl")
EERS_PATH = Path("./ablation/ablation_graph_k_eers.jsonl")
DATASET = "fvusm"
DEFAULT_STAT_SEED = 0
PROPOSED_SETTING = (18, 9)
SETTINGS = [(9, 9), (18, 18), (9, 18)]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        records = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["key"]] = record["value"]
        return records


def append_jsonl(path: Path, key: str, value) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps({"key": key, "value": value}, sort_keys=True) + "\n")


def get_run_name_from_record(record) -> str:
    if isinstance(record, dict):
        return record["run_name"]
    return record


def build_variant_config(config: dict, stage1_k: int, stage2_k: int, stat_seed: int) -> dict:
    local_config = copy.deepcopy(config)
    local_config["seed"] = stat_seed
    local_config["stat_seed"] = stat_seed
    local_config["backbone"]["block0"]["neighbour_number"] = stage1_k
    local_config["backbone"]["block1"]["neighbour_number"] = stage2_k
    return local_config


def make_model_name(stage1_k: int, stage2_k: int, stat_seed: int) -> str:
    return f"graph_k_leaveoneout_{DATASET}_seed_{stat_seed}_k1_{stage1_k}_k2_{stage2_k}"


def build_train_args(
    config_path: Path,
    stat_seed: int,
    wandb: bool,
    stage1_k: int,
    stage2_k: int,
):
    return argparse.Namespace(
        config=str(config_path),
        seed=stat_seed,
        leave=DATASET,
        wandb=wandb,
        dataset="leaveoneout",
        model_name=make_model_name(stage1_k, stage2_k, stat_seed),
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


def run_variant(args, stage1_k: int, stage2_k: int) -> None:
    config_path = Path(args.config).resolve()
    base_config = load_config(config_path)
    variant_config = build_variant_config(base_config, stage1_k, stage2_k, args.stat_seed)

    run_key = f"{DATASET}:{args.stat_seed}:k1_{stage1_k}:k2_{stage2_k}"
    runs = load_jsonl(RESULTS_PATH)
    if run_key in runs and not args.retrain:
        run_name = get_run_name_from_record(runs[run_key])
        print(f"Using existing run: {run_key} -> {run_name}")
    else:
        train_args = build_train_args(
            config_path, args.stat_seed, args.wandb, stage1_k, stage2_k
        )
        run_name = train_main(train_args, variant_config)
        append_jsonl(
            RESULTS_PATH,
            run_key,
            {
                "run_name": run_name,
                "wandb_run_name": run_name if args.wandb else None,
                "stage1_k": stage1_k,
                "stage2_k": stage2_k,
                "stat_seed": args.stat_seed,
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
    append_jsonl(
        EERS_PATH,
        run_key,
        {
            "stage1_k": stage1_k,
            "stage2_k": stage2_k,
            "stat_seed": args.stat_seed,
            "eer": eer,
        },
    )
    print(f"Saved EER: {run_key} -> {eer}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the remaining graph-k sensitivity ablations for FV-USM."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file to use.",
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=DEFAULT_STAT_SEED,
        help="Statistical seed to run. Default is 0.",
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
    print(
        "Running graph-k sensitivity for the remaining settings. "
        f"Proposed setting ({PROPOSED_SETTING[0]}, {PROPOSED_SETTING[1]}) is assumed to be available already."
    )
    for stage1_k, stage2_k in SETTINGS:
        print(
            f"Running dataset={DATASET} stat_seed={args.stat_seed} "
            f"stage1_k={stage1_k} stage2_k={stage2_k}"
        )
        run_variant(args, stage1_k, stage2_k)


if __name__ == "__main__":
    main()
