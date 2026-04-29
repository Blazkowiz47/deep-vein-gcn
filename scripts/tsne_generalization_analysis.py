import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from cdatasets import get_dataset
from models import get_model
from run_name_mappings import final_runs, get_config_file
from utils import logger, set_seeds


DEFAULT_DATASET = "fvusm"
DEFAULT_METHOD = "proposed"
METHOD_ALIASES = {
    "proposed": "snakegraph2",
    "snakegraph2": "snakegraph2",
    "arcvein": "arcvein",
    "lgfin": "lgfin",
    "fvit": "fv-vit",
    "fv-vit": "fv-vit",
    "resnet": "resnet",
    "veinattnet": "veinAttNet",
    "veinAttNet": "veinAttNet",
}
FEATURES_FLAG_METHODS = {"lgfin", "fv-vit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-wise t-SNE analysis for why a leave-one-dataset-out model generalises well."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Held-out dataset.")
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=sorted(METHOD_ALIASES.keys()),
        help="Method to analyse.",
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=None,
        help="Dataset stat seed for the held-out target dataset. If omitted, it is inferred.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint/run name. If omitted, the script infers it from the stat seed or full-subject results.",
    )
    parser.add_argument(
        "--n",
        "--n-classes",
        dest="n_classes",
        type=int,
        default=8,
        help="Number of best and worst classes to plot.",
    )
    parser.add_argument(
        "--top-k-impostors",
        type=int,
        default=5,
        help="Number of hardest impostor similarities per sample used in the class margin.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=20.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--learning-rate",
        default="auto",
        help="t-SNE learning rate.",
    )
    parser.add_argument(
        "--tsne-backend",
        default="tsnecuda",
        choices=["auto", "tsnecuda", "sklearn"],
        help="t-SNE backend to use. Default is tsnecuda.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Ignore cached embeddings and re-extract them.",
    )
    parser.add_argument(
        "--logger-level",
        default="ERROR",
        help="Logger level.",
    )
    parser.add_argument(
        "--output-root",
        default="rebuttal/tsne_generalization",
        help="Directory for plots and class-summary outputs.",
    )
    return parser.parse_args()


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


def resolve_checkpoint(dataset: str, method: str, stat_seed: int, checkpoint: str) -> str:
    if checkpoint:
        return checkpoint
    normalized = normalize_method(method)
    return final_runs[dataset][normalized][stat_seed]


def infer_stat_seed_from_checkpoint(dataset: str, method: str, checkpoint: str) -> int | None:
    normalized = normalize_method(method)
    for seed, run_name in final_runs.get(dataset, {}).get(normalized, {}).items():
        if run_name == checkpoint:
            return int(seed)
    return None


def select_best_seed_from_full_results(dataset: str, method: str) -> int:
    results_path = Path("ablation") / "full_subjects_results.json"
    with results_path.open("r", encoding="utf-8") as fp:
        results = json.load(fp)

    normalized = normalize_method(method)
    method_results = results.get(dataset, {}).get(normalized, {})
    if not method_results:
        raise ValueError(
            f"No full-subject results found for dataset={dataset}, method={normalized} in {results_path}"
        )

    best_seed = None
    best_eer = None
    for seed_str, metrics in method_results.items():
        eer = metrics.get("eer")
        if eer is None:
            continue
        seed = int(seed_str)
        if best_eer is None or eer < best_eer:
            best_eer = eer
            best_seed = seed

    if best_seed is None:
        raise ValueError(
            f"No EER values found for dataset={dataset}, method={normalized} in {results_path}"
        )
    return best_seed


def resolve_analysis_target(
    dataset: str,
    method: str,
    stat_seed: int | None,
    checkpoint: str | None,
) -> tuple[int, str, str]:
    if checkpoint:
        inferred_seed = infer_stat_seed_from_checkpoint(dataset, method, checkpoint)
        resolved_seed = stat_seed if stat_seed is not None else inferred_seed
        if resolved_seed is None:
            raise ValueError(
                "Could not infer stat seed from checkpoint. Pass --stat-seed explicitly."
            )
        return resolved_seed, checkpoint, "checkpoint"

    if stat_seed is not None:
        return stat_seed, resolve_checkpoint(dataset, method, stat_seed, None), "stat_seed"

    best_seed = select_best_seed_from_full_results(dataset, method)
    best_checkpoint = resolve_checkpoint(dataset, method, best_seed, None)
    return best_seed, best_checkpoint, "full_subjects_results"


def cache_paths(checkpoint: str, dataset: str) -> Dict[str, Path]:
    base = Path("tmp") / checkpoint / dataset / "tsne_generalization"
    base.mkdir(parents=True, exist_ok=True)
    return {"embeddings": base / "cached_embeddings.pt"}


def artifact_paths(output_root: str, checkpoint: str, dataset: str) -> Dict[str, Path]:
    safe_name = f"{dataset}_{checkpoint}"
    base = Path(output_root) / safe_name
    base.mkdir(parents=True, exist_ok=True)
    return {
        "class_stats_csv": base / "class_stats.csv",
        "class_stats_json": base / "class_stats.json",
        "all_plot": base / "all_classes_tsne.png",
        "best_plot": base / "best_classes_tsne.png",
        "worst_plot": base / "worst_classes_tsne.png",
    }


def transform_image(fname: str, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(fname).resize(size)
    imgarray = np.array(img)
    if imgarray.ndim == 3:
        imgarray = imgarray[..., 0]
    imgarray = (imgarray - imgarray.min()) / (imgarray.max() - imgarray.min() + 1e-8)
    imgarray = np.stack([imgarray, imgarray, imgarray], axis=0)
    return torch.tensor(imgarray).float()


def build_subject_pairs(
    dataset: str, config: dict, log, stat_seed: int
) -> List[Tuple[str, int, str]]:
    wrapper = get_dataset(dataset, config, log, partition_split=0, stat_seed=stat_seed)
    pairs = wrapper.loop_splitset("test")
    enriched = []
    for img_path, label in pairs:
        subject_id = Path(img_path).parent.name
        enriched.append((img_path, int(label), subject_id))
    return enriched


def load_model_for_analysis(method: str, checkpoint: str, config: dict, log):
    model = get_model(config["model"], config, log).to(config["device"])
    checkpoint_path = Path("final_runs") / checkpoint / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path("tmp") / checkpoint / "checkpoints" / "best_model.pt"
    state_dict = torch.load(
        checkpoint_path, map_location=config["device"], weights_only=True
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def extract_embeddings(
    checkpoint: str,
    method: str,
    dataset: str,
    stat_seed: int,
    config: dict,
    args: argparse.Namespace,
):
    paths = cache_paths(checkpoint, dataset)
    if paths["embeddings"].exists() and not args.force_reextract:
        return torch.load(paths["embeddings"])

    log = logger.get_logger(f"tsne_{checkpoint}", level=args.logger_level)
    set_seeds(log, config.get("seed", 2025))
    pairs = build_subject_pairs(dataset, config, log, stat_seed)

    if method == "veinAttNet":
        feature_root = Path("features") / checkpoint
        rows = []
        for img_path, _, subject_id in pairs:
            split_name = Path(img_path).parent.parent.name
            stem = Path(img_path).stem
            feature_path = feature_root / subject_id / f"{split_name}_{stem}.txt"
            if not feature_path.exists():
                continue
            emb = torch.tensor(np.loadtxt(feature_path)).float()
            rows.append(
                {
                    "embedding": emb,
                    "path": img_path,
                    "subject_id": subject_id,
                }
            )
        if not rows:
            raise ValueError(f"No VeinAttNet features found under {feature_root}")
    else:
        model = load_model_for_analysis(method, checkpoint, config, log)
        rows = []
        image_size = (config.get("width", 224), config.get("height", 224))
        with torch.no_grad():
            for start in tqdm(range(0, len(pairs), args.batch_size), desc="Extracting embeddings"):
                batch_pairs = pairs[start : start + args.batch_size]
                batch = torch.stack(
                    [transform_image(img_path, image_size) for img_path, _, _ in batch_pairs],
                    dim=0,
                ).to(config["device"])
                if method in FEATURES_FLAG_METHODS:
                    outputs = model(batch, features=True).detach().cpu()
                else:
                    outputs = model(batch).detach().cpu()
                for (img_path, _, subject_id), emb in zip(batch_pairs, outputs):
                    rows.append(
                        {
                            "embedding": emb.squeeze().float(),
                            "path": img_path,
                            "subject_id": subject_id,
                        }
                    )

    embeddings = torch.stack([row["embedding"] for row in rows], dim=0)
    cache = {
        "embeddings": embeddings,
        "paths": [row["path"] for row in rows],
        "subject_ids": [row["subject_id"] for row in rows],
        "checkpoint": checkpoint,
        "dataset": dataset,
        "method": method,
        "stat_seed": stat_seed,
    }
    torch.save(cache, paths["embeddings"])
    return cache


def compute_class_stats(
    embeddings: torch.Tensor,
    subject_ids: List[str],
    top_k_impostors: int,
) -> List[Dict[str, float | str | int]]:
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sims = normalized @ normalized.T
    subject_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, subject_id in enumerate(subject_ids):
        subject_to_indices[subject_id].append(idx)

    stats = []
    all_indices = torch.arange(len(subject_ids))
    for subject_id, indices in subject_to_indices.items():
        idx = torch.tensor(indices, dtype=torch.long)
        subject_sims = sims[idx][:, idx]
        genuine_mask = torch.tril(torch.ones_like(subject_sims, dtype=torch.bool), diagonal=-1)
        genuine_scores = subject_sims[genuine_mask]
        if genuine_scores.numel() == 0:
            genuine_mean = float("nan")
        else:
            genuine_mean = float(genuine_scores.mean().item())

        impostor_indices = all_indices[torch.isin(all_indices, idx, invert=True)]
        impostor_matrix = sims[idx][:, impostor_indices]
        per_sample_topk = []
        for row in impostor_matrix:
            if row.numel() == 0:
                continue
            k = min(top_k_impostors, row.numel())
            per_sample_topk.append(float(torch.topk(row, k=k).values.mean().item()))
        impostor_topk_mean = (
            float(np.mean(per_sample_topk)) if per_sample_topk else float("nan")
        )
        margin = genuine_mean - impostor_topk_mean
        stats.append(
            {
                "subject_id": subject_id,
                "num_images": len(indices),
                "genuine_mean": genuine_mean,
                "impostor_topk_mean": impostor_topk_mean,
                "margin": margin,
            }
        )

    stats.sort(key=lambda row: row["margin"], reverse=True)
    for rank, row in enumerate(stats, start=1):
        row["rank"] = rank
    return stats


def save_class_stats(
    stats: List[Dict[str, float | str | int]],
    output_root: str,
    checkpoint: str,
    dataset: str,
) -> None:
    paths = artifact_paths(output_root, checkpoint, dataset)
    with paths["class_stats_json"].open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    fieldnames = [
        "rank",
        "subject_id",
        "num_images",
        "genuine_mean",
        "impostor_topk_mean",
        "margin",
    ]
    with paths["class_stats_csv"].open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


def select_embeddings(
    embeddings: torch.Tensor,
    subject_ids: List[str],
    selected_subject_ids: Iterable[str],
) -> Tuple[np.ndarray, List[str]]:
    selected = set(selected_subject_ids)
    indices = [idx for idx, subject_id in enumerate(subject_ids) if subject_id in selected]
    return embeddings[indices].cpu().numpy(), [subject_ids[idx] for idx in indices]


def plot_tsne(
    embedding_array: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    perplexity: float,
    learning_rate,
    backend: str,
    show_legend: bool = False,
    annotate_centroids: bool = False,
) -> None:
    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 classes for t-SNE plotting.")
    effective_perplexity = min(perplexity, max(5.0, len(embedding_array) - 1))
    xy = run_tsne(
        embedding_array,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        backend=backend,
    )

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    for idx, subject_id in enumerate(unique_labels):
        points = xy[np.array([label == subject_id for label in labels])]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            alpha=0.8,
            label=subject_id,
            color=cmap(idx),
        )
        if annotate_centroids:
            plt.text(points[:, 0].mean(), points[:, 1].mean(), subject_id, fontsize=8)

    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    if show_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_tsne(
    embedding_array: np.ndarray,
    perplexity: float,
    learning_rate,
    backend: str,
) -> np.ndarray:
    requested_backend = backend
    if backend == "auto":
        try:
            backend = "tsnecuda"
        except Exception:
            backend = "sklearn"

    if backend == "tsnecuda":
        try:
            from tsnecuda import TSNE as CudaTSNE  # type: ignore

            lr = 200.0 if learning_rate == "auto" else float(learning_rate)
            tsne = CudaTSNE(
                n_components=2,
                perplexity=float(perplexity),
                learning_rate=lr,
                n_iter=1000,
                init="random",
                random_seed=2025,
                verbose=1,
            )
            return tsne.fit_transform(
                np.asarray(embedding_array, dtype=np.float32, order="C")
            )
        except Exception as exc:
            warnings.warn(
                f"tsnecuda backend failed ({exc}). Falling back to sklearn TSNE.",
                RuntimeWarning,
            )
            backend = "sklearn"

    if backend == "sklearn":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            init="pca",
            learning_rate=learning_rate,
            random_state=2025,
            max_iter=1000,
        )
        return tsne.fit_transform(embedding_array)

    raise ValueError(f"Unsupported t-SNE backend: {requested_backend}")


def main() -> None:
    args = parse_args()
    method = normalize_method(args.method)
    stat_seed, checkpoint, checkpoint_source = resolve_analysis_target(
        args.dataset, method, args.stat_seed, args.checkpoint
    )
    config_path = get_config_file(method)

    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    config["leaveoutds"] = args.dataset
    config["stat_seed"] = stat_seed
    config["seed"] = stat_seed

    cache = extract_embeddings(checkpoint, method, args.dataset, stat_seed, config, args)
    class_stats = compute_class_stats(
        cache["embeddings"], cache["subject_ids"], args.top_k_impostors
    )
    save_class_stats(class_stats, args.output_root, checkpoint, args.dataset)

    n_classes = min(args.n_classes, len(class_stats) // 2)
    best_subjects = [row["subject_id"] for row in class_stats[:n_classes]]
    worst_subjects = [row["subject_id"] for row in class_stats[-n_classes:]]

    cache_path_info = cache_paths(checkpoint, args.dataset)
    paths = artifact_paths(args.output_root, checkpoint, args.dataset)
    best_embeddings, best_labels = select_embeddings(
        cache["embeddings"], cache["subject_ids"], best_subjects
    )
    worst_embeddings, worst_labels = select_embeddings(
        cache["embeddings"], cache["subject_ids"], worst_subjects
    )
    all_embeddings = cache["embeddings"].cpu().numpy()
    all_labels = cache["subject_ids"]

    plot_tsne(
        all_embeddings,
        all_labels,
        title=f"{display_method(method)} on {args.dataset}: all classes",
        out_path=paths["all_plot"],
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        backend=args.tsne_backend,
        show_legend=False,
        annotate_centroids=False,
    )

    plot_tsne(
        best_embeddings,
        best_labels,
        title=f"{display_method(method)} on {args.dataset}: best {n_classes} classes",
        out_path=paths["best_plot"],
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        backend=args.tsne_backend,
    )
    plot_tsne(
        worst_embeddings,
        worst_labels,
        title=f"{display_method(method)} on {args.dataset}: worst {n_classes} classes",
        out_path=paths["worst_plot"],
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        backend=args.tsne_backend,
    )

    print(json.dumps(
        {
            "checkpoint": checkpoint,
            "checkpoint_source": checkpoint_source,
            "dataset": args.dataset,
            "method": display_method(method),
            "stat_seed": stat_seed,
            "cache_file": str(cache_path_info["embeddings"]),
            "class_stats_csv": str(paths["class_stats_csv"]),
            "all_plot": str(paths["all_plot"]),
            "best_plot": str(paths["best_plot"]),
            "worst_plot": str(paths["worst_plot"]),
            "best_subjects": best_subjects,
            "worst_subjects": worst_subjects,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
