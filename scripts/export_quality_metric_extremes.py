from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_DATASETS = ("vera", "polyu", "mmcbnu", "fvusm", "fv300")
DEFAULT_METRICS = ("gradient", "gradient_c", "laplacian", "contrast")


def export_metric_extremes(
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
    top_k: int,
    max_per_subject: int,
    num_subjects: int,
    ablation_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}

    for dataset in datasets:
        cache_path = ablation_dir / f"{dataset}_quality.json"
        if not cache_path.exists():
            print(f"[WARN] Missing cache for {dataset}: {cache_path}")
            continue

        with cache_path.open("r") as fp:
            results = json.load(fp)

        manifest[dataset] = {}
        for metric in metrics:
            if metric not in results or not results[metric]:
                print(f"[WARN] Missing metric '{metric}' for {dataset}")
                continue

            dataset_metric_dir = output_dir / dataset / metric
            if dataset_metric_dir.exists():
                shutil.rmtree(dataset_metric_dir)
            worst_dir = dataset_metric_dir / "worst"
            best_dir = dataset_metric_dir / "best"
            worst_dir.mkdir(parents=True, exist_ok=True)
            best_dir.mkdir(parents=True, exist_ok=True)

            items = sorted(
                ((Path(path), float(score)) for path, score in results[metric].items()),
                key=lambda item: item[1],
            )
            worst_items = select_extremes(
                items,
                limit=top_k,
                max_per_subject=max_per_subject,
                num_subjects=num_subjects,
            )
            best_items = select_extremes(
                list(reversed(items)),
                limit=top_k,
                max_per_subject=max_per_subject,
                num_subjects=num_subjects,
            )

            manifest[dataset][metric] = {"worst": [], "best": []}

            for rank_idx, (image_path, score) in enumerate(worst_items, start=1):
                relative_suffix = "__".join(image_path.parts[-3:])
                destination = worst_dir / f"{rank_idx:02d}_{relative_suffix}"
                shutil.copy2(image_path, destination)
                manifest[dataset][metric]["worst"].append(
                    {
                        "rank": rank_idx,
                        "score": score,
                        "source_path": str(image_path),
                        "copied_path": str(destination),
                    }
                )

            for rank_idx, (image_path, score) in enumerate(best_items, start=1):
                relative_suffix = "__".join(image_path.parts[-3:])
                destination = best_dir / f"{rank_idx:02d}_{relative_suffix}"
                shutil.copy2(image_path, destination)
                manifest[dataset][metric]["best"].append(
                    {
                        "rank": rank_idx,
                        "score": score,
                        "source_path": str(image_path),
                        "copied_path": str(destination),
                    }
                )

            print(
                f"{dataset}/{metric}: exported {len(worst_items)} worst and {len(best_items)} best images"
            )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"saved manifest to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export top/bottom quality-metric examples from cached quality JSON files."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to export.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to export.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of best and worst images per dataset/metric.",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=2,
        help="Maximum number of images to export per subject folder.",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=5,
        help="Maximum number of distinct subjects to include in best/worst exports.",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("./ablation"),
        help="Directory containing <dataset>_quality.json caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./ablation/quality/metric_extremes"),
        help="Directory where exported images will be saved.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.max_per_subject <= 0:
        raise ValueError("--max-per-subject must be positive.")
    if args.num_subjects <= 0:
        raise ValueError("--num-subjects must be positive.")

    export_metric_extremes(
        datasets=tuple(args.datasets),
        metrics=tuple(args.metrics),
        top_k=args.top_k,
        max_per_subject=args.max_per_subject,
        num_subjects=args.num_subjects,
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
    )


def select_extremes(
    items: list[tuple[Path, float]],
    limit: int,
    max_per_subject: int,
    num_subjects: int,
) -> list[tuple[Path, float]]:
    selected: list[tuple[Path, float]] = []
    subject_counts: dict[str, int] = {}

    for image_path, score in items:
        subject_key = image_path.parent.name
        if subject_key not in subject_counts and len(subject_counts) >= num_subjects:
            continue
        if subject_counts.get(subject_key, 0) >= max_per_subject:
            continue

        selected.append((image_path, score))
        subject_counts[subject_key] = subject_counts.get(subject_key, 0) + 1

        if len(selected) >= limit and len(subject_counts) >= num_subjects:
            break

    return selected


if __name__ == "__main__":
    main()
