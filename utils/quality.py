import torch
import torch.nn.functional as F


def get_grayscale_channel(image_batch: torch.Tensor) -> torch.Tensor:
    return image_batch[:, :1]


def compute_quality_components(
    image_batch: torch.Tensor,
    dark_threshold: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grayscale = get_grayscale_channel(image_batch)
    contrast = grayscale.flatten(1).std(dim=1)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=grayscale.device,
        dtype=grayscale.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=grayscale.device,
        dtype=grayscale.dtype,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(grayscale, sobel_x, padding=1)
    grad_y = F.conv2d(grayscale, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    grad_energy = grad_mag.mean(dim=(1, 2, 3))
    dark_ratio = (grayscale < dark_threshold).float().mean(dim=(1, 2, 3))
    quality = contrast + 0.5 * grad_energy + 0.25 * dark_ratio

    return quality, contrast, grad_energy, dark_ratio, grad_mag

