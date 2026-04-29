import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from cdatasets import get_dataset
from models import get_model
from run_name_mappings import get_config_file
from utils import calculate_eer, initialise_dirs, logger, set_seeds


FAR_TARGETS_PERCENT = (0.1, 1.0, 10.0)
SUPPORTED_DATASETS = ("fv300", "fvusm", "mmcbnu")
METHOD_ALIASES = {
    "proposed": "snakegraph2",
    "snakegraph2": "snakegraph2",
    "arcvein": "arcvein",
    "lgfin": "lgfin",
    "fvit": "fv-vit",
    "fv-vit": "fv-vit",
    "chen": "chen",
    "resnet": "resnet",
    "veinattnet": "veinAttNet",
    "veinAttNet": "veinAttNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intra-database open-set evaluation on left-out identities."
    )
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--checkpoint", required=True, help="Run/checkpoint name.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Main dataset for intra-database training/evaluation.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Method name. Supports proposed, arcvein, lgfin, fvit, veinAttNet.",
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        required=True,
        help="Dataset subject split seed.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Fraction of sorted subject IDs used for training identities.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--logger-level",
        default="ERROR",
        help="Logger level.",
    )
    return parser.parse_args()


def normalize_method(method: str) -> str:
    if method not in METHOD_ALIASES:
        raise ValueError(f"Unsupported method: {method}")
    return METHOD_ALIASES[method]


def transform_image(fname: str, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(fname).resize(size)
    imgarray = np.array(img)
    imgarray = imgarray / 255.0
    imgarray = np.stack([imgarray, imgarray, imgarray], axis=0)
    return torch.tensor(imgarray).float()


def get_subject_ids(dataset: str, stat_seed: int) -> List[str]:
    train_dir = Path("data") / dataset / str(stat_seed) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    return sorted([entry.name for entry in train_dir.iterdir() if entry.is_dir()])


def get_left_out_subject_ids(
    dataset: str, stat_seed: int, partition_split: float
) -> List[str]:
    subject_ids = get_subject_ids(dataset, stat_seed)
    total_ids = int(len(subject_ids) * partition_split)
    return subject_ids[total_ids:]


def get_left_out_subject_images(
    dataset: str, stat_seed: int, partition_split: float
) -> Dict[str, List[str]]:
    left_out_ids = set(get_left_out_subject_ids(dataset, stat_seed, partition_split))
    if not left_out_ids:
        raise ValueError(
            f"No left-out identities found for dataset={dataset} stat_seed={stat_seed}"
        )

    split_root = Path("data") / dataset / str(stat_seed)
    subjects: Dict[str, List[str]] = {sid: [] for sid in sorted(left_out_ids)}
    for split_name in ["train", "test"]:
        split_dir = split_root / split_name
        if not split_dir.exists():
            continue
        for subject_id in sorted(left_out_ids):
            subject_dir = split_dir / subject_id
            if not subject_dir.exists():
                continue
            for img_path in sorted(subject_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    subjects[subject_id].append(str(img_path))

    subjects = {sid: images for sid, images in subjects.items() if images}
    if not subjects:
        raise ValueError(
            f"No left-out subject images found for dataset={dataset} stat_seed={stat_seed}"
        )
    return subjects


def get_rate_paths(model_name: str, dataset: str) -> Dict[str, str]:
    base = f"tmp/{model_name}/{dataset}"
    return {
        "far": f"{base}/far_scores.npy",
        "frr": f"{base}/frr_scores.npy",
        "genuine": f"{base}/genuine_scores.npy",
        "imposter": f"{base}/imposter_scores.npy",
        "summary": f"{base}/intra_open_set_summary.json",
    }


def load_saved_rates(model_name: str, dataset: str) -> Dict[str, np.ndarray] | None:
    paths = get_rate_paths(model_name, dataset)
    required = [paths["far"], paths["frr"]]
    if not all(os.path.exists(path) for path in required):
        return None
    return {"far": np.load(paths["far"]), "frr": np.load(paths["frr"])}


def tar_at_far(far: np.ndarray, frr: np.ndarray, target_far_percent: float) -> float:
    idx = int(np.argmin(np.abs(far - target_far_percent)))
    return 1.0 - (float(frr[idx]) / 100.0)


def get_auc_from_rates(far: np.ndarray, frr: np.ndarray) -> float:
    x = far / 100.0
    y = (100.0 - frr) / 100.0
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]) * 100.0)


def get_eer_from_rates(far: np.ndarray, frr: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


def get_metric_summary(far: np.ndarray, frr: np.ndarray) -> Dict[str, float]:
    return {
        "auc": get_auc_from_rates(far, frr),
        "eer": get_eer_from_rates(far, frr),
        "tar_far_0.1": tar_at_far(far, frr, 0.1),
        "tar_far_1": tar_at_far(far, frr, 1.0),
        "tar_far_10": tar_at_far(far, frr, 10.0),
    }


def load_pytorch_model(config: dict, checkpoint_name: str, device: str, log):
    model = get_model(config["model"], config, log).to(device)
    checkpoint_path = Path("tmp") / checkpoint_name / "checkpoints" / "best_model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    return model


def extract_pytorch_embeddings(
    model,
    subjects: Dict[str, List[str]],
    image_size: Tuple[int, int],
    device: str,
) -> Dict[str, torch.Tensor]:
    subject_embeddings: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for subject_id, images in tqdm(subjects.items(), desc="Fetching embeddings"):
            feats = []
            for start in range(0, len(images), 128):
                batch_files = images[start : start + 128]
                batch = torch.stack(
                    [transform_image(fname, image_size) for fname in batch_files], dim=0
                ).to(device)
                outputs = model(batch).detach().cpu()
                for feat in outputs:
                    feats.append(feat.squeeze().unsqueeze(0))
            subject_embeddings[subject_id] = torch.cat(feats, dim=0)
    return subject_embeddings


def extract_veinattnet_embeddings(
    checkpoint_name: str,
    dataset: str,
    stat_seed: int,
    partition_split: float,
) -> Dict[str, torch.Tensor]:
    feature_root = Path("features") / checkpoint_name
    if not feature_root.exists():
        raise FileNotFoundError(
            f"Expected VeinAttNet features at {feature_root}. Run the MATLAB intra script first."
        )

    left_out_ids = get_left_out_subject_ids(dataset, stat_seed, partition_split)
    subject_embeddings: Dict[str, torch.Tensor] = {}
    for subject_id in left_out_ids:
        feature_files = sorted((feature_root / subject_id).glob("*.txt"))
        if not feature_files:
            continue
        subject_embeddings[subject_id] = torch.cat(
            [
                torch.tensor(np.loadtxt(feature_file)).squeeze().unsqueeze(0)
                for feature_file in feature_files
            ],
            dim=0,
        )
    if not subject_embeddings:
        raise ValueError(f"No VeinAttNet features found under {feature_root}")
    return subject_embeddings


def compute_scores(
    subject_embeddings: Dict[str, torch.Tensor], device: str
) -> Tuple[List[float], List[float]]:
    genuine_scores: List[float] = []
    imposter_scores: List[float] = []
    subjects = sorted(subject_embeddings.keys())

    for subject in tqdm(subjects, desc="Calculating genuine scores"):
        x = subject_embeddings[subject].to(device)
        x = F.normalize(x, p=2, dim=1)
        sims = (x @ x.T).detach().cpu()
        lower = sims[torch.tril(torch.ones_like(sims, dtype=torch.bool), diagonal=-1)]
        genuine_scores.extend(lower.tolist())

    for i, probe in tqdm(
        enumerate(subjects),
        desc="Calculating imposter scores",
        total=len(subjects),
    ):
        for enroll in subjects[i + 1 :]:
            x = F.normalize(subject_embeddings[enroll].to(device), p=2, dim=1)
            y = F.normalize(subject_embeddings[probe].to(device), p=2, dim=1)
            sims = (x @ y.T).detach().cpu().flatten()
            imposter_scores.extend(sims.tolist())

    return genuine_scores, imposter_scores


def evaluate(args: argparse.Namespace, config: dict) -> Dict[str, float]:
    method = normalize_method(args.method)
    config["stat_seed"] = args.stat_seed
    config["seed"] = args.stat_seed
    config["main_dataset"] = args.dataset

    model_name = args.checkpoint
    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/eval_intra_{args.dataset}.log"
    log = logger.get_logger(model_name, logfile, args.logger_level)

    set_seeds(log, config["seed"])
    saved_rates = load_saved_rates(model_name, args.dataset)
    if saved_rates is not None:
        summary = get_metric_summary(saved_rates["far"], saved_rates["frr"])
        log.error(f"Loaded cached intra FAR/FRR for {model_name} on {args.dataset}")
        log.error(f"Summary: {summary}")
        return summary

    device = config["device"]
    if method == "veinAttNet":
        subject_embeddings = extract_veinattnet_embeddings(
            args.checkpoint, args.dataset, args.stat_seed, args.partition_split
        )
    else:
        wrapper = get_dataset("intra", config, log)
        config["num_classes"] = wrapper.num_classes
        model = load_pytorch_model(config, args.checkpoint, device, log)
        subject_images = get_left_out_subject_images(
            args.dataset, args.stat_seed, args.partition_split
        )
        subject_embeddings = extract_pytorch_embeddings(
            model,
            subject_images,
            (config.get("width", 224), config.get("height", 224)),
            device,
        )

    genuine_scores, imposter_scores = compute_scores(subject_embeddings, device)
    far, frr = None, None
    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)

    paths = get_rate_paths(model_name, args.dataset)
    os.makedirs(Path(paths["far"]).parent, exist_ok=True)
    np.save(paths["genuine"], np.array(genuine_scores))
    np.save(paths["imposter"], np.array(imposter_scores))
    np.save(paths["far"], far)
    np.save(paths["frr"], frr)

    summary = get_metric_summary(far, frr)
    summary["eer"] = eer
    Path(paths["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.error(f"{model_name} Dataset: {args.dataset} intra open-set summary: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    method = normalize_method(args.method)
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    config["model"] = config.get("model", method if method != "snakegraph2" else "dscgrapher")
    summary = evaluate(args, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
