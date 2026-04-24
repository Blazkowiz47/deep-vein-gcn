import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from fvcore.nn import FlopCountAnalysis
from thop import profile

from models import get_model
from run_name_mappings import final_runs, get_config_file


METHODS = ["arcvein", "lgfin", "fv-vit", "snakegraph2"]
METHOD_LABELS = {
    "arcvein": "ArcVein",
    "lgfin": "LGFIN",
    "fv-vit": "FV-ViT",
    "snakegraph2": "Proposed Method",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime and FLOPs for fv300 SOTA checkpoints."
    )
    parser.add_argument("--dataset", default="fv300", help="Dataset split to benchmark.")
    parser.add_argument("--seed", type=int, default=0, help="Seed index to load.")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for the dummy input."
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup forward passes before timing.",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=100,
        help="Measured forward passes.",
    )
    parser.add_argument(
        "--output",
        default="ablation/fv300_runtime_benchmark.json",
        help="Path to save the benchmark summary JSON.",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logger = logging.getLogger("runtime_benchmark")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.ERROR)
    return logger


def load_config(method: str, device: str) -> Dict[str, Any]:
    with open(get_config_file(method), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["device"] = device
    return config


def checkpoint_path(dataset: str, method: str, seed: int) -> Path:
    run_name = final_runs[dataset][method][seed]
    return Path("final_runs") / run_name / "best_model.pt"


def load_model_for_device(
    method: str, dataset: str, seed: int, device: str, logger: logging.Logger
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    config = load_config(method, device)
    model = get_model(config["model"], config, logger).to(device)
    state_dict = torch.load(
        checkpoint_path(dataset, method, seed),
        map_location=device,
        weights_only=True,
    )
    model_state = model.state_dict()
    compatible_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()
    return model, config


def make_dummy_input(config: Dict[str, Any], batch_size: int, device: str) -> torch.Tensor:
    return torch.randn(
        batch_size,
        3,
        config["height"],
        config["width"],
        device=device,
    )


def benchmark_latency(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    warmup_iters: int,
    timed_iters: int,
) -> Dict[str, float]:
    device = dummy_input.device.type
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(timed_iters):
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    latency_ms = (elapsed_s / timed_iters) * 1000.0
    throughput = timed_iters * dummy_input.shape[0] / elapsed_s
    return {
        "latency_ms": latency_ms,
        "throughput_samples_per_s": throughput,
    }


def compute_flops_thop(model: torch.nn.Module, dummy_input: torch.Tensor) -> Optional[float]:
    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


def compute_flops_fvcore(
    model: torch.nn.Module, dummy_input: torch.Tensor
) -> Optional[float]:
    try:
        flops = FlopCountAnalysis(model, dummy_input).total()
        return float(flops)
    except Exception:
        return None


def format_number(value: Optional[float], scale: float = 1.0, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value / scale:.{digits}f}"


def summarize_results(results: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Method | Params (M) | FLOPs (G) | CPU Latency (ms) | CPU Throughput | GPU Latency (ms) | GPU Throughput |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        metrics = results[method]
        cpu = metrics["cpu"]
        gpu = metrics["gpu"]
        lines.append(
            "| "
            + " | ".join(
                [
                    METHOD_LABELS[method],
                    format_number(metrics["params"], scale=1e6),
                    format_number(metrics["flops"], scale=1e9),
                    format_number(cpu.get("latency_ms")),
                    format_number(cpu.get("throughput_samples_per_s")),
                    format_number(gpu.get("latency_ms")),
                    format_number(gpu.get("throughput_samples_per_s")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logger = build_logger()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    has_cuda = torch.cuda.is_available()

    for method in METHODS:
        cpu_model, cpu_config = load_model_for_device(
            method, args.dataset, args.seed, "cpu", logger
        )
        cpu_input = make_dummy_input(cpu_config, args.batch_size, "cpu")
        cpu_metrics = benchmark_latency(
            cpu_model, cpu_input, args.warmup_iters, args.timed_iters
        )
        params = float(sum(param.numel() for param in cpu_model.parameters()))
        flops = compute_flops_thop(cpu_model, cpu_input)
        if flops is None:
            flops = compute_flops_fvcore(cpu_model, cpu_input)

        gpu_metrics: Dict[str, Optional[float]] = {
            "latency_ms": None,
            "throughput_samples_per_s": None,
        }
        if has_cuda:
            gpu_model, gpu_config = load_model_for_device(
                method, args.dataset, args.seed, "cuda", logger
            )
            gpu_input = make_dummy_input(gpu_config, args.batch_size, "cuda")
            gpu_metrics = benchmark_latency(
                gpu_model, gpu_input, args.warmup_iters, args.timed_iters
            )
            del gpu_model
            del gpu_input
            torch.cuda.empty_cache()

        results[method] = {
            "label": METHOD_LABELS[method],
            "dataset": args.dataset,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "params": params,
            "flops": flops,
            "cpu": cpu_metrics,
            "gpu": gpu_metrics,
        }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summarize_results(results))
    print(f"\nSaved JSON to {output_path}")


if __name__ == "__main__":
    main()
