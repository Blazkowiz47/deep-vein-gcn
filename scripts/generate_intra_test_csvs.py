import argparse
import csv
from pathlib import Path


SUPPORTED_DATASETS = ("fv300", "fvusm", "mmcbnu")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact held-out image manifests for intra-dataset evaluation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--stat-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
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
        default=Path("ablation/intra_test_csvs"),
    )
    return parser.parse_args()


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
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "Dataset": dataset,
                            "StatSeed": stat_seed,
                            "SubjectID": subject_id,
                            "ImageID": image_path.stem,
                            "SourceSplit": source_split,
                            "Path": image_path.as_posix(),
                        }
                    )

    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_output_dir / f"seed_{stat_seed}_test.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Dataset",
                "StatSeed",
                "SubjectID",
                "ImageID",
                "SourceSplit",
                "Path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(held_out_ids), len(rows)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        for stat_seed in args.stat_seeds:
            subject_count, image_count = generate_manifest(
                dataset, stat_seed, args.partition_split, args.output_dir
            )
            output_path = args.output_dir / dataset / f"seed_{stat_seed}_test.csv"
            print(
                f"Generated {output_path}: "
                f"{subject_count} held-out subjects, {image_count} images"
            )


if __name__ == "__main__":
    main()
