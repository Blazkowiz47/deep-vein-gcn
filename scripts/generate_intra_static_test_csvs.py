import argparse
import csv
from pathlib import Path


SUPPORTED_DATASETS = ("fv300", "fvusm", "mmcbnu")
STATIC_METHODS = ("mcp", "rlt", "wld")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate held-out intra-dataset manifests for static methods."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=0,
        help="Canonical data layout to use. Held-out image unions must match all seeds.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Fraction of sorted subject IDs used for model development.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation/intra_static_test_csvs"),
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    return path.as_posix()


def generate_manifest(
    dataset: str, stat_seed: int, partition_split: float, output_dir: Path
) -> tuple[int, int]:
    dataset_root = Path("data") / dataset / str(stat_seed)
    train_dir = dataset_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    subject_ids = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    split_index = int(len(subject_ids) * partition_split)
    held_out_ids = subject_ids[split_index:]
    rows: list[dict[str, str | int]] = []

    for subject_id in held_out_ids:
        for source_split in ("train", "test"):
            subject_dir = dataset_root / source_split / subject_id
            if not subject_dir.is_dir():
                continue
            for image_path in sorted(subject_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                feature_paths = {
                    method: Path("features")
                    / method
                    / dataset
                    / source_split
                    / subject_id
                    / f"{image_path.stem}.mat"
                    for method in STATIC_METHODS
                }
                missing = [str(path) for path in feature_paths.values() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing static features for {image_path}: {', '.join(missing)}"
                    )

                rows.append(
                    {
                        "Dataset": dataset,
                        "CanonicalSeed": stat_seed,
                        "SubjectID": subject_id,
                        "ImageID": image_path.stem,
                        "SourceSplit": source_split,
                        "ImagePath": relative_path(image_path),
                        "MCPFeaturePath": relative_path(feature_paths["mcp"]),
                        "RLTFeaturePath": relative_path(feature_paths["rlt"]),
                        "WLDFeaturePath": relative_path(feature_paths["wld"]),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_test.csv"
    fieldnames = [
        "Dataset",
        "CanonicalSeed",
        "SubjectID",
        "ImageID",
        "SourceSplit",
        "ImagePath",
        "MCPFeaturePath",
        "RLTFeaturePath",
        "WLDFeaturePath",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(held_out_ids), len(rows)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        subject_count, image_count = generate_manifest(
            dataset, args.stat_seed, args.partition_split, args.output_dir
        )
        print(
            f"Generated {args.output_dir / f'{dataset}_test.csv'}: "
            f"{subject_count} held-out subjects, {image_count} images"
        )


if __name__ == "__main__":
    main()
