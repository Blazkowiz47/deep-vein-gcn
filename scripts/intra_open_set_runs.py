import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from cdatasets import get_dataset
from losses import get_loss
from models import get_model
from run_name_mappings import get_config_file
from intra_open_set_eval import METHOD_ALIASES, SUPPORTED_DATASETS, evaluate
from train import main as train_main
from torch.optim import SGD
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from utils import (
    initialise_dirs,
    logger as logger_utils,
    set_seeds,
)


DEFAULT_DATASETS = ["fv300", "fvusm", "mmcbnu"]
DEFAULT_METHODS = ["proposed", "lgfin", "fvit", "arcvein", "resnet"]
DEFAULT_STAT_SEEDS = [0, 1, 2, 3, 4]
RUNS_PATH = Path("./ablation/intra_open_set_runs.jsonl")
RESULTS_PATH = Path("./ablation/intra_open_set_results.jsonl")


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


def normalize_method(method: str) -> str:
    if method not in METHOD_ALIASES:
        raise ValueError(f"Unsupported method: {method}")
    return METHOD_ALIASES[method]


def display_method(method: str) -> str:
    if method == "snakegraph2":
        return "proposed"
    if method == "fv-vit":
        return "fvit"
    return method


def config_path_for_method(method: str) -> str:
    return get_config_file(method)


def build_train_args(config_path: str, dataset: str, stat_seed: int, wandb: bool):
    return argparse.Namespace(
        config=config_path,
        seed=stat_seed,
        leave=None,
        wandb=wandb,
        dataset="intra",
        model_name=None,
        logger_level="INFO",
        continue_model=None,
    )


def build_eval_args(
    config_path: str,
    checkpoint: str,
    dataset: str,
    method: str,
    stat_seed: int,
    partition_split: float,
):
    return argparse.Namespace(
        config=config_path,
        checkpoint=checkpoint,
        dataset=dataset,
        method=method,
        stat_seed=stat_seed,
        partition_split=partition_split,
        batch_size=128,
        logger_level="ERROR",
    )


def build_variant_config(config: dict, dataset: str, stat_seed: int) -> dict:
    local_config = copy.deepcopy(config)
    local_config["seed"] = stat_seed
    local_config["stat_seed"] = stat_seed
    local_config["main_dataset"] = dataset
    return local_config


def make_model_name(method: str, dataset: str, stat_seed: int) -> str:
    return f"{method}_intra_{dataset}_seed_{stat_seed}"


def train_arcvein_intra(args, config: dict) -> str:
    model_name = args.model_name
    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/train.log"
    ckptdir = rf"tmp/{model_name}/checkpoints"
    log = logger_utils.get_logger(model_name, logfile, args.logger_level)
    log.info(f"Training started for: {model_name}.")
    log.info("Config:")

    set_seeds(log, config["seed"])
    epochs = config["epochs"]
    validate_after_epochs = config["validate_after_epochs"]
    device = config["device"]

    wrapper = get_dataset(args.dataset, config, log)
    config["num_classes"] = wrapper.num_classes

    log.info(str(config))
    model = get_model(config["model"], config, log).to(device)
    log.info("Model:")
    log.info(str(model))

    criterion = get_loss(config["loss"], config, log).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    params.extend([p for p in criterion.parameters() if p.requires_grad])

    metric = MulticlassAccuracy(num_classes=config["num_classes"]).to(device)
    optimizer = SGD(
        params,
        lr=config["lr"],
        nesterov=True,
        momentum=0.9,
        weight_decay=5e-4,
    )

    best_acc_val = 0.0
    early_stop = 10
    validation_acc_didnt_increase = early_stop

    validationds = wrapper.get_split("validation", batch_size=8)
    for epoch in range(epochs):
        model.train()
        criterion.train()
        metric.reset()
        train_losses = []
        step1_train_losses = []
        step2_train_losses = []
        pbar = tqdm(wrapper.get_split("train"), desc=f"Epoch {epoch + 1}")
        for image, label in pbar:
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            preds = model(image)
            loss1, loss2 = criterion(
                preds, label, freeze_centroids=epoch > config["freeze_centroids"]
            )
            step_loss = loss1 + loss2
            step_loss.backward()
            optimizer.step()

            with torch.no_grad():
                logits = criterion.softmax(criterion.fc(preds))
                metric.update(logits.exp(), label.argmax(dim=1))

            train_losses.append(step_loss.detach().cpu().item())
            step1_train_losses.append(loss1.detach().cpu().item())
            step2_train_losses.append(loss2.detach().cpu().item())
            pbar.set_postfix({"loss": float(np.mean(train_losses))})
        pbar.close()

        log.info(f"Average train step loss: {np.mean(train_losses)}")
        log.info(f"Train accuracy: {metric.compute().detach().cpu().item()}")

        torch.save(model.state_dict(), os.path.join(ckptdir, f"epoch_{epoch}.pt"))

        if epoch % validate_after_epochs:
            continue

        model.eval()
        criterion.eval()
        metric.reset()
        validation_losses = []
        step1_losses = []
        step2_losses = []
        pbar = tqdm(validationds, desc="Validation")
        with torch.no_grad():
            for image, label in pbar:
                image, label = image.to(device), label.to(device)
                preds = model(image)
                loss1, loss2 = criterion(preds, label)
                logits = criterion.softmax(criterion.fc(preds))
                step_loss = loss1 + loss2
                validation_losses.append(step_loss.detach().cpu().item())
                metric.update(logits.exp(), label.argmax(dim=1))
                step1_losses.append(loss1.detach().cpu().item())
                step2_losses.append(loss2.detach().cpu().item())
                pbar.set_postfix({"loss": float(np.mean(validation_losses))})
        pbar.close()

        validation_acc = metric.compute().detach().cpu().item()
        log.info(f"Average validation step loss: {np.mean(validation_losses)}")
        log.info(f"Validation accuracy: {validation_acc}")
        if validation_acc > best_acc_val:
            best_acc_val = validation_acc
            torch.save(model.state_dict(), os.path.join(ckptdir, "best_model.pt"))
            validation_acc_didnt_increase = early_stop
        else:
            validation_acc_didnt_increase -= 1
            if validation_acc_didnt_increase == 0:
                log.info(
                    f"Validation accuracy didn't improve for {early_stop} epochs. Stopping training."
                )
                break

    log.info(f"Training completed for: {model_name}.")
    return model_name


def run_variant(
    dataset: str,
    method_arg: str,
    stat_seed: int,
    partition_split: float,
    wandb: bool,
    retrain: bool,
    reeval: bool,
    skip_eval: bool,
) -> None:
    method = normalize_method(method_arg)
    config_path = config_path_for_method(method)
    with open(config_path, "r", encoding="utf-8") as fp:
        base_config = yaml.safe_load(fp)
    variant_config = build_variant_config(base_config, dataset, stat_seed)
    wrapper_log = logger_utils.get_logger(
        f"intra_wrapper_{dataset}_{stat_seed}", level="ERROR"
    )
    wrapper = get_dataset("intra", variant_config, wrapper_log)
    variant_config["num_classes"] = wrapper.num_classes
    if (
        variant_config.get("loss") == "crossentropy"
        and "embedding_size" not in variant_config
    ):
        variant_config["embedding_size"] = variant_config["num_classes"]

    run_key = f"{dataset}:{stat_seed}:{display_method(method)}"
    model_name = make_model_name(display_method(method), dataset, stat_seed)
    runs = load_jsonl(RUNS_PATH)
    if run_key in runs and not retrain:
        run_name = get_run_name_from_record(runs[run_key])
        print(f"Using existing run: {run_key} -> {run_name}")
    else:
        train_args = build_train_args(config_path, dataset, stat_seed, wandb)
        train_args.model_name = model_name
        if method == "arcvein":
            run_name = train_arcvein_intra(train_args, variant_config)
        else:
            run_name = train_main(train_args, variant_config)
        append_jsonl(
            RUNS_PATH,
            run_key,
            {
                "run_name": run_name,
                "dataset": dataset,
                "stat_seed": stat_seed,
                "method": display_method(method),
            },
        )
        print(f"Saved run: {run_key} -> {run_name}")

    if skip_eval:
        return

    results = load_jsonl(RESULTS_PATH)
    if run_key in results and not reeval:
        print(f"Existing results: {run_key} -> {results[run_key]}")
        return

    eval_args = build_eval_args(
        config_path, run_name, dataset, display_method(method), stat_seed, partition_split
    )
    summary = evaluate(eval_args, variant_config)
    append_jsonl(RESULTS_PATH, run_key, summary)
    print(f"Saved results: {run_key} -> {summary}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate intra-database open-set runs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(SUPPORTED_DATASETS),
        help="Datasets to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(set(METHOD_ALIASES.keys()) - {"veinattnet", "veinAttNet"}),
        help="Methods to run.",
    )
    parser.add_argument(
        "--stat-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_STAT_SEEDS,
        help="Stat seeds to run.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Training identity fraction used by the intra wrapper.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb for training runs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train and record run names.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a run exists.",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Force reevaluation even if cached results exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        for method in args.methods:
            for stat_seed in args.stat_seeds:
                print(
                    "Running intra open-set "
                    f"dataset={dataset} method={method} stat_seed={stat_seed}"
                )
                run_variant(
                    dataset=dataset,
                    method_arg=method,
                    stat_seed=stat_seed,
                    partition_split=args.partition_split,
                    wandb=args.wandb,
                    retrain=args.retrain,
                    reeval=args.reeval,
                    skip_eval=args.skip_eval,
                )


if __name__ == "__main__":
    main()
