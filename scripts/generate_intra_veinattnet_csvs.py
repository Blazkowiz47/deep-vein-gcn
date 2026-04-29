import argparse
import csv
from pathlib import Path


SUPPORTED_DATASETS = ("fv300", "fvusm", "mmcbnu")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate intra-database VeinAttNet train/validation CSVs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=list(SUPPORTED_DATASETS),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--stat-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Dataset stat seeds to process.",
    )
    parser.add_argument(
        "--partition-split",
        type=float,
        default=0.8,
        help="Fraction of sorted subject IDs used for training identities.",
    )
    parser.add_argument(
        "--output-root",
        default="tmp/intra_dataset_csv",
        help="Directory where generated CSVs will be stored.",
    )
    return parser.parse_args()


def list_subject_ids(train_dir: Path) -> list[str]:
    return sorted([entry.name for entry in train_dir.iterdir() if entry.is_dir()])


def iter_subject_images(subject_dir: Path) -> list[str]:
    if not subject_dir.exists():
        return []
    return sorted(
        str(path.resolve())
        for path in subject_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_for_dataset(dataset: str, stat_seed: int, partition_split: float, output_root: Path) -> None:
    dataset_root = Path("data") / dataset / str(stat_seed)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    subject_ids = list_subject_ids(train_dir)
    total_ids = int(len(subject_ids) * partition_split)
    train_subject_ids = subject_ids[:total_ids]
    left_out_subject_ids = subject_ids[total_ids:]

    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    leftout_rows: list[dict[str, str]] = []

    for subject_id in train_subject_ids:
        for image_path in iter_subject_images(train_dir / subject_id):
            train_rows.append({"Path": image_path, "Label": subject_id})
        for image_path in iter_subject_images(test_dir / subject_id):
            validation_rows.append({"Path": image_path, "Label": subject_id})

    for subject_id in left_out_subject_ids:
        for image_path in iter_subject_images(train_dir / subject_id):
            leftout_rows.append(
                {"Path": image_path, "Label": subject_id, "Split": "train"}
            )
        for image_path in iter_subject_images(test_dir / subject_id):
            leftout_rows.append(
                {"Path": image_path, "Label": subject_id, "Split": "test"}
            )

    output_dir = output_root / dataset / str(stat_seed)
    write_csv(output_dir / "train.csv", train_rows, ["Path", "Label"])
    write_csv(output_dir / "validation.csv", validation_rows, ["Path", "Label"])
    write_csv(output_dir / "leftout.csv", leftout_rows, ["Path", "Label", "Split"])
    print(
        f"Generated {dataset} stat_seed={stat_seed}: "
        f"{len(train_rows)} train rows, "
        f"{len(validation_rows)} validation rows, "
        f"{len(leftout_rows)} left-out rows"
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for dataset in args.datasets:
        for stat_seed in args.stat_seeds:
            generate_for_dataset(dataset, stat_seed, args.partition_split, output_root)


if __name__ == "__main__":
    main()
