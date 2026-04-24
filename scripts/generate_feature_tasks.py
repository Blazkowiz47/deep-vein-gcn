#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


DATASETS = ("vera", "mmcbnu", "fvusm", "fv300", "polyu")
SEEDS = ("0", "1", "2", "3")
SPLITS = ("train", "test")


def iter_tasks(root: Path, datasets: tuple[str, ...], seeds: tuple[str, ...], create_dirs: bool):
    for seed in seeds:
        for dataset in datasets:
            data_dir = root / "data" / dataset / "0"
            checkpoint_path = root / "tmp" / f"leaveoutds_veinAttNet_{dataset}_seed_{seed}" / "checkpoints" / "best_model.mat"
            output_dir = root / "features" / f"leaveoutds_veinAttNet_{dataset}_seed_{seed}"

            if not checkpoint_path.is_file():
                print(f"SKIP missing checkpoint: {checkpoint_path}")
                continue
            if not data_dir.is_dir():
                print(f"SKIP missing data dir: {data_dir}")
                continue

            for split in SPLITS:
                split_input_dir = data_dir / split
                split_output_dir = output_dir / split
                if not split_input_dir.is_dir():
                    print(f"SKIP missing split: {split_input_dir}")
                    continue

                for class_dir in sorted(p for p in split_input_dir.iterdir() if p.is_dir()):
                    class_output_dir = split_output_dir / class_dir.name
                    if create_dirs:
                        class_output_dir.mkdir(parents=True, exist_ok=True)

                    for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
                        stem = image_path.stem
                        output_path = class_output_dir / f"{stem}.mat"
                        output_text_path = class_output_dir / f"{stem}.txt"

                        if output_text_path.is_file():
                            continue

                        yield {
                            "checkpointPath": str(checkpoint_path),
                            "imagePath": str(image_path),
                            "outputPath": str(output_path),
                            "outputTextPath": str(output_text_path),
                        }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate remaining VeinAttNet feature extraction tasks.")
    parser.add_argument("--root", default=".", type=Path, help="Repository root.")
    parser.add_argument("--output", default="feature_tasks.csv", type=Path, help="Output CSV path.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to scan.")
    parser.add_argument("--seeds", nargs="*", default=SEEDS, help="Seeds to scan.")
    parser.add_argument(
        "--no-create-dirs",
        action="store_true",
        help="Do not create feature output directories while generating the task CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpointPath",
        "imagePath",
        "outputPath",
        "outputTextPath",
    ]

    count = 0
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task in iter_tasks(root, tuple(args.datasets), tuple(args.seeds), not args.no_create_dirs):
            writer.writerow(task)
            count += 1

    print(f"WROTE {count} tasks to {output}")


if __name__ == "__main__":
    main()
