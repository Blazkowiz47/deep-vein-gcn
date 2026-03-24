import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from utils import compute_quality_components

height, width = 224, 224
output_dir = Path("./scripts/sobel_outputs")
output_dir.mkdir(exist_ok=True)
quality_dark_threshold = 0.45


def normalise_for_plot(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    image = image - image.min()
    image = image / (image.max() + 1e-6)
    return image.numpy()


def resize_for_plot(image: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def build_mode_filters(dtype: torch.dtype) -> list[tuple[str, tuple[str, torch.Tensor], tuple[str, torch.Tensor]]]:
    return [
        (
            "XY",
            (
                "Sobel X",
                torch.tensor(
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
            (
                "Sobel Y",
                torch.tensor(
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
        ),
        (
            "DR",
            (
                "Diag R X",
                torch.tensor(
                    [[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
            (
                "Diag R Y",
                torch.tensor(
                    [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
        ),
        (
            "DL",
            (
                "Diag L X",
                torch.tensor(
                    [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
            (
                "Diag L Y",
                torch.tensor(
                    [[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]],
                    dtype=dtype,
                ).view(1, 1, 3, 3),
            ),
        ),
    ]


def build_combined_image(response1: torch.Tensor, response2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(response1.square() + response2.square() + 1e-6)


for dataset in Path("./data").iterdir():
    if not dataset.is_dir() or "enhanced_" in dataset.name:
        continue
    all_images = [
        f for f in dataset.rglob("*") if f.suffix.lower() in [".png", ".jpg", ".bmp"]
    ]
    if not all_images:
        continue
    random_image = random.choice(all_images)
    imgarray = np.array(Image.open(random_image))
    imgarray = (imgarray - imgarray.min()) / (imgarray.max() - imgarray.min())
    grayscale = torch.from_numpy(imgarray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    image_batch = grayscale.repeat(1, 3, 1, 1)
    quality, contrast, grad_energy, dark_ratio, grad_mag = compute_quality_components(
        image_batch,
        dark_threshold=quality_dark_threshold,
    )

    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    grayscale_2d = grayscale.squeeze(0).squeeze(0)
    mode_filters = build_mode_filters(grayscale.dtype)
    for row, (mode_name, (name1, kernel1), (name2, kernel2)) in enumerate(mode_filters):
        response1 = F.conv2d(grayscale, kernel1, padding=1).squeeze(0).squeeze(0)
        response2 = F.conv2d(grayscale, kernel2, padding=1).squeeze(0).squeeze(0)
        combined = build_combined_image(response1, response2)

        axes[row, 0].imshow(normalise_for_plot(resize_for_plot(grayscale_2d)), cmap="gray")
        axes[row, 0].set_title("Original")
        axes[row, 1].imshow(normalise_for_plot(resize_for_plot(response1)), cmap="gray")
        axes[row, 1].set_title(name1)
        axes[row, 2].imshow(normalise_for_plot(resize_for_plot(response2)), cmap="gray")
        axes[row, 2].set_title(name2)
        axes[row, 3].imshow(normalise_for_plot(resize_for_plot(combined)), cmap="gray")
        axes[row, 3].set_title("Combined Response")
        axes[row, 0].set_ylabel(mode_name, rotation=90, labelpad=12, fontsize=12)

    for axis in axes.flatten():
        axis.axis("off")

    figure.suptitle(f"{dataset.name}: {random_image.name}")
    figure.tight_layout()
    sobel_filtered_image_path = output_dir / f"{dataset.name}.png"
    figure.savefig(sobel_filtered_image_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        f"{dataset.name} | {random_image.name} | "
        f"quality={quality.item():.4f} contrast={contrast.item():.4f} "
        f"grad_energy={grad_energy.item():.4f} dark_ratio={dark_ratio.item():.4f}"
    )
    print(f"Saved {sobel_filtered_image_path}")
