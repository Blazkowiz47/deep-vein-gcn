import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import torch

from intra_open_set_eval import (
    SUPPORTED_DATASETS,
    get_left_out_subject_images,
    get_metric_summary,
)


SUPPORTED_METHODS = ("mcp", "rlt", "wld")
RESULTS_PATH = Path("ablation/intra_open_static_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate static methods on intra-dataset held-out identities."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS),
        choices=SUPPORTED_METHODS,
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=0,
        help="Canonical intra-dataset image layout to evaluate.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Fraction of sorted subject IDs used for model development.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("features"),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Recompute scores even when cached arrays are present.",
    )
    return parser.parse_args()


def feature_path_for_image(
    image_path: str, method: str, dataset: str, feature_root: Path
) -> Path:
    path = Path(image_path)
    source_split = path.parts[-3]
    subject_id = path.parent.name
    return feature_root / method / dataset / source_split / subject_id / f"{path.stem}.mat"


def load_subject_features(
    subject_images: Dict[str, List[str]],
    method: str,
    dataset: str,
    feature_root: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    feature_shape = None

    for label, (subject_id, image_paths) in enumerate(subject_images.items()):
        for image_path in image_paths:
            feature_path = feature_path_for_image(
                image_path, method, dataset, feature_root
            )
            if not feature_path.is_file():
                raise FileNotFoundError(
                    f"Missing {method.upper()} feature for {image_path}: {feature_path}"
                )
            mat = scipy.io.loadmat(feature_path)
            if "features" not in mat:
                raise KeyError(f"No 'features' array in {feature_path}")
            feature = np.asarray(mat["features"])
            if feature_shape is None:
                feature_shape = feature.shape
            elif feature.shape != feature_shape:
                raise ValueError(
                    f"Feature shape mismatch in {feature_path}: "
                    f"expected {feature_shape}, got {feature.shape}"
                )
            features.append(feature.reshape(-1))
            labels.append(label)

    if not features:
        raise ValueError(f"No {method.upper()} features loaded for {dataset}")
    return np.stack(features), np.asarray(labels, dtype=np.int64), len(subject_images)


def normalize_for_correlation(
    features: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, int]:
    embeddings = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    embeddings -= embeddings.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
    zero_norm_count = int((norms.squeeze(1) == 0).sum().item())
    embeddings = torch.where(norms > 0, embeddings / norms.clamp_min(1e-12), 0)
    return embeddings, zero_norm_count


def compute_scores(
    features: np.ndarray, labels: np.ndarray, device: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    torch_device = torch.device(device)
    embeddings, zero_norm_count = normalize_for_correlation(features, torch_device)
    label_tensor = torch.from_numpy(labels).to(torch_device)

    if zero_norm_count == embeddings.shape[0]:
        label_counts = np.bincount(labels)
        genuine_count = int(np.sum(label_counts * (label_counts - 1) // 2))
        total_count = embeddings.shape[0] * (embeddings.shape[0] - 1) // 2
        return (
            np.zeros(genuine_count, dtype=np.float32),
            np.zeros(total_count - genuine_count, dtype=np.float32),
            zero_norm_count,
        )

    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    similarities = embeddings @ embeddings.T
    row_indices, column_indices = torch.triu_indices(
        similarities.shape[0], similarities.shape[1], offset=1, device=torch_device
    )
    same_subject = label_tensor[row_indices] == label_tensor[column_indices]
    pair_scores = similarities[row_indices, column_indices]
    genuine_scores = pair_scores[same_subject].detach().cpu().numpy()
    imposter_scores = pair_scores[~same_subject].detach().cpu().numpy()
    return genuine_scores, imposter_scores, zero_norm_count


def calculate_rates(
    genuine_scores: np.ndarray, imposter_scores: np.ndarray, bins: int = 10_001
) -> Tuple[float, np.ndarray, np.ndarray]:
    genuine = np.sort(np.asarray(genuine_scores))
    imposter = np.sort(np.asarray(imposter_scores))
    minimum = min(float(genuine[0]), float(imposter[0]))
    maximum = max(float(genuine[-1]), float(imposter[-1]))
    thresholds = np.linspace(minimum, maximum, bins)

    false_rejects = np.searchsorted(genuine, thresholds, side="right")
    false_accepts = imposter.size - np.searchsorted(
        imposter, thresholds, side="left"
    )
    frr = false_rejects * 100.0 / genuine.size
    far = false_accepts * 100.0 / imposter.size
    index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[index] + frr[index]) / 2.0)
    return eer, far, frr


def score_paths(method: str, dataset: str, stat_seed: int) -> Dict[str, Path]:
    output_dir = Path("tmp") / f"{method}_intra_{dataset}_seed_{stat_seed}" / dataset
    return {
        "output_dir": output_dir,
        "genuine": output_dir / "genuine_scores.npy",
        "imposter": output_dir / "imposter_scores.npy",
    }


def evaluate(
    method: str,
    dataset: str,
    stat_seed: int,
    partition_split: float,
    feature_root: Path,
    device: str,
    reeval: bool,
) -> Dict[str, object]:
    paths = score_paths(method, dataset, stat_seed)
    if paths["genuine"].is_file() and paths["imposter"].is_file() and not reeval:
        genuine_scores = np.load(paths["genuine"])
        imposter_scores = np.load(paths["imposter"])
        subject_images = get_left_out_subject_images(
            dataset, stat_seed, partition_split
        )
        subject_count = len(subject_images)
        image_count = sum(len(images) for images in subject_images.values())
        zero_norm_count = -1
    else:
        subject_images = get_left_out_subject_images(
            dataset, stat_seed, partition_split
        )
        features, labels, subject_count = load_subject_features(
            subject_images, method, dataset, feature_root
        )
        image_count = features.shape[0]
        genuine_scores, imposter_scores, zero_norm_count = compute_scores(
            features, labels, device
        )
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        np.save(paths["genuine"], genuine_scores)
        np.save(paths["imposter"], imposter_scores)

    eer, far, frr = calculate_rates(genuine_scores, imposter_scores)
    correct_metrics = get_metric_summary(far, frr)
    correct_metrics["eer"] = eer

    # utils.metrics.calculate_eer currently returns FAR and FRR in reverse order.
    # Preserve that convention separately so rows can be compared with the
    # existing intra table without presenting it as the corrected calculation.
    legacy_table_metrics = get_metric_summary(frr, far)
    legacy_table_metrics["eer"] = eer

    return {
        "dataset": dataset,
        "method": method,
        "stat_seed": stat_seed,
        "subject_count": subject_count,
        "image_count": image_count,
        "genuine_count": int(genuine_scores.size),
        "imposter_count": int(imposter_scores.size),
        "zero_norm_feature_count": zero_norm_count,
        "legacy_table_metrics": legacy_table_metrics,
        "correct_metrics": correct_metrics,
    }


def load_results(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results = load_results(args.results_path)
    for dataset in args.datasets:
        for method in args.methods:
            key = f"{dataset}:{method}"
            print(f"Evaluating {key} on stat_seed={args.stat_seed}")
            results[key] = evaluate(
                method=method,
                dataset=dataset,
                stat_seed=args.stat_seed,
                partition_split=args.partition_split,
                feature_root=args.feature_root,
                device=args.device,
                reeval=args.reeval,
            )
            args.results_path.parent.mkdir(parents=True, exist_ok=True)
            args.results_path.write_text(
                json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps(results[key], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
