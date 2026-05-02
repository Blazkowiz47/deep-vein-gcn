import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.nn import CosineSimilarity, Module
from tqdm import tqdm

from cdatasets import get_dataset
from models import get_model
from run_name_mappings import final_runs, get_config_file
from utils import calculate_eer, initialise_dirs, logger, set_seeds


FAR_TARGETS_PERCENT = (0.1, 1.0, 10.0)


def transform(fname: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    img = Image.open(fname).resize(size)
    imgarray = np.array(img)
    imgarray = (imgarray.squeeze() - imgarray.min()) / (imgarray.max() - imgarray.min())
    imgarray = np.stack([imgarray, imgarray, imgarray], axis=0)

    return torch.tensor(imgarray).float()


def get_scores(
    model: Module, data: List[Tuple[torch.Tensor, torch.Tensor]], device
) -> List[float]:
    """
    Get the scores for the pairs.
    """
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    scores = []
    with torch.no_grad():
        for sample1, sample2 in tqdm(data):
            sample1 = sample1.to(device)
            sample2 = sample2.to(device)
            emb1 = model(sample1)
            emb2 = model(sample2)
            score = cosine_sim(emb1, emb2)
            scores.extend(score.detach().cpu().numpy().tolist())

    return scores


def get_model_name(method: str, checkpoint: str) -> str:
    if method == "veinAttNet":
        return "_".join(checkpoint.split("_")[1:])
    return checkpoint


def get_rate_paths(model_name: str, dataset: str) -> Dict[str, str]:
    base = f"tmp/{model_name}/{dataset}"
    return {
        "half_far": f"{base}/far_half_scores.npy",
        "half_frr": f"{base}/frr_half_scores.npy",
        "full_far": f"{base}/far_full_scores.npy",
        "full_frr": f"{base}/frr_full_scores.npy",
    }


def load_saved_rates(model_name: str, dataset: str) -> Dict[str, np.ndarray] | None:
    paths = get_rate_paths(model_name, dataset)
    if not all(os.path.exists(path) for path in paths.values()):
        return None

    return {name: np.load(path) for name, path in paths.items()}


def tar_at_far(far: np.ndarray, frr: np.ndarray, target_far_percent: float) -> float:
    idx = int(np.argmin(np.abs(far - target_far_percent)))
    return 1.0 - (float(frr[idx]) / 100.0)


def get_tar_summary(
    far: np.ndarray,
    frr: np.ndarray,
    far_targets_percent: Tuple[float, ...] = FAR_TARGETS_PERCENT,
) -> Dict[str, float]:
    return {
        f"far_{target:g}": tar_at_far(far, frr, target)
        for target in far_targets_percent
    }


def get_eer_from_rates(far: np.ndarray, frr: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


def get_auc_from_rates(far: np.ndarray, frr: np.ndarray) -> float:
    x = far / 100.0
    y = (100.0 - frr) / 100.0
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]) * 100.0)


def get_metric_summary(far: np.ndarray, frr: np.ndarray) -> Dict[str, float]:
    tar_summary = get_tar_summary(far, frr)
    return {
        "auc": get_auc_from_rates(far, frr),
        "eer": get_eer_from_rates(far, frr),
        "tar_far_0.1": tar_summary["far_0.1"],
        "tar_far_1": tar_summary["far_1"],
        "tar_far_10": tar_summary["far_10"],
    }


def parallel_driver(args, config) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Wrapper for the driver.
    """

    dataset = args.dataset
    model = config["model"]

    checkpoint = args.checkpoint
    model_name = checkpoint
    checkpoint = os.path.join("./final_runs", checkpoint, "best_model.pt")

    initialise_dirs(model_name)
    logfile = rf"final_runs/{model_name}/eval_{dataset}.log"
    log = logger.get_logger(model_name, logfile, args.logger_level)

    set_seeds(log, config["seed"])
    device = config["device"]  # You can change this to cpu.

    saved_rates = load_saved_rates(model_name, dataset)
    if saved_rates is not None:
        half_summary = get_metric_summary(
            saved_rates["half_far"], saved_rates["half_frr"]
        )
        full_summary = get_metric_summary(
            saved_rates["full_far"], saved_rates["full_frr"]
        )
        log.error(f"Loaded cached FAR/FRR for {model_name} on {dataset}")
        log.error(f"Half summary: {half_summary}")
        log.error(f"Full summary: {full_summary}")
        return half_summary, full_summary

    config["leaveoutds"] = args.dataset
    leaveoutds = get_dataset("leaveoneout", config, log)
    config["num_classes"] = leaveoutds.num_classes

    wrapper = get_dataset(args.dataset, config, log, partition_split=0)
    testds = wrapper.get_split("test", batch_size=128)

    subject_embeddings: Dict[int, torch.Tensor] = {}
    raw_subject_embeddings: Dict[int, List[torch.Tensor]] = {}
    if model != "veinAttNet":
        model = get_model(model, config, log).to(device)
        model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
        model.eval()
        model.to(device)
        log.info(str(model))

        with torch.no_grad():
            for images, labels in tqdm(testds, desc="Fetching Embeddings"):
                feats = model(images.to(device)).detach().cpu()
                labels = labels.argmax(dim=1).numpy().tolist()
                for feat, label in zip(feats, labels):
                    if label not in raw_subject_embeddings:
                        raw_subject_embeddings[label] = []

                    raw_subject_embeddings[label].append(feat.squeeze().unsqueeze(0))
    else:
        path = os.path.join("./features/" + args.checkpoint)
        for ssplit in ["test", "train"]:
            spath = os.path.join(path, ssplit)
            for sid, subject in tqdm(
                enumerate(os.listdir(spath)), desc="Fetching Embeddings"
            ):
                for fname in os.listdir(os.path.join(spath, subject)):
                    if fname.endswith(".txt"):
                        raw_subject_embeddings.setdefault(sid,[]).append(
                            torch.tensor(
                                np.loadtxt(os.path.join(spath, subject, fname))
                            )
                            .squeeze()
                            .unsqueeze(0)
                        )

    for subject in raw_subject_embeddings:
        subject_embeddings[subject] = torch.cat(raw_subject_embeddings[subject], dim=0)

    genuine_scores: List[float] = []
    imposter_scores: List[float] = []
    genuine_scores_full: List[float] = []
    imposter_scores_full: List[float] = []
    log.error(f"Total subjects: {len(subject_embeddings)}")

    # Now split the total subjects in 2 groups.
    # Enroll only half of them and probe remaining half.
    # Get genuine scores from enrolled and imposter from unenrolled probes.
    sorted_subjects = list(sorted(subject_embeddings.keys()))
    enrolled_subjects = sorted_subjects[: len(sorted_subjects) // 2]
    probe_subjects = sorted_subjects[len(sorted_subjects) // 2 :]

    for subject in tqdm(enrolled_subjects, desc="Calculating genuine scores"):
        x = subject_embeddings[subject].cuda()
        x = F.normalize(x, p=2, dim=1)
        x = (x @ x.T).cpu()
        lower = x[torch.tril(torch.ones_like(x, dtype=torch.bool), diagonal=-1)]
        genuine_scores.extend(lower.tolist())

    genuine_scores_full.extend(genuine_scores)
    for subject in tqdm(probe_subjects, desc="Calculating remaining genuine scores"):
        x = subject_embeddings[subject].cuda()
        x = F.normalize(x, p=2, dim=1)
        x = (x @ x.T).cpu()
        lower = x[torch.tril(torch.ones_like(x, dtype=torch.bool), diagonal=-1)]
        genuine_scores_full.extend(lower.tolist())

    for probe in tqdm(probe_subjects, desc="Calculating probe scores"):
        for enrol in enrolled_subjects:
            x = subject_embeddings[enrol].cuda()
            y = subject_embeddings[probe].cuda()
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)
            x = (x @ y.T).cpu().flatten()
            imposter_scores.extend(x.tolist())

    # imposter_scores_full.extend(imposter_scores)

    subjects = list(subject_embeddings.keys())
    for i, probe in tqdm(
        enumerate(subjects),
        desc="Calculating remaining probe scores",
        total=len(subjects),
    ):
        for enrol in subjects[i + 1 :]:
            x = subject_embeddings[enrol].cuda()
            y = subject_embeddings[probe].cuda()
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)
            x = (x @ y.T).cpu().flatten()
            imposter_scores_full.extend(x.tolist())

    log.error(f"Total genuine scores: {len(genuine_scores)}")
    log.error(f"Total imposter scores: {len(imposter_scores)}")

    log.error(f"Total genuine scores full: {len(genuine_scores_full)}")
    log.error(f"Total imposter scores full: {len(imposter_scores_full)}")

    print("Saving Scores")
    os.makedirs(f"tmp/{model_name}/{dataset}", exist_ok=True)
    np.save(
        f"tmp/{model_name}/{dataset}/genuine_half_scores.npy", np.array(genuine_scores)
    )
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_half_scores.npy",
        np.array(imposter_scores),
    )
    np.save(
        f"tmp/{model_name}/{dataset}/genuine_full_scores.npy",
        np.array(genuine_scores_full),
    )
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_full_scores.npy",
        np.array(imposter_scores_full),
    )

    genuine_scores = np.load(f"tmp/{model_name}/{dataset}/genuine_half_scores.npy")
    imposter_scores = np.load(f"tmp/{model_name}/{dataset}/imposter_half_scores.npy")

    genuine_scores_full = np.load(f"tmp/{model_name}/{dataset}/genuine_full_scores.npy")
    imposter_scores_full = np.load(
        f"tmp/{model_name}/{dataset}/imposter_full_scores.npy"
    )
    log.error(f"Total genuine scores: {len(genuine_scores)}")
    log.error(f"Total imposter scores: {len(imposter_scores)}")

    log.error(f"Total genuine scores full: {len(genuine_scores_full)}")
    log.error(f"Total imposter scores full: {len(imposter_scores_full)}")

    torch.cuda.empty_cache()
    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    eer_full, far_full, frr_full, _ = calculate_eer(
        genuine_scores_full, imposter_scores_full
    )
    log.error(f"{model_name} Dataset: {dataset} EER: {eer}")
    log.error(f"{model_name} Dataset: {dataset} EER: {eer_full}")
    np.save(f"tmp/{model_name}/{dataset}/far_half_scores.npy", far)
    np.save(f"tmp/{model_name}/{dataset}/frr_half_scores.npy", frr)
    np.save(f"tmp/{model_name}/{dataset}/far_full_scores.npy", far_full)
    np.save(f"tmp/{model_name}/{dataset}/frr_full_scores.npy", frr_full)
    return get_metric_summary(far, frr), get_metric_summary(far_full, frr_full)


def final_runs_eval():
    results = {}
    results_full = {}
    for dataset, methods in final_runs.items():
        results[dataset] = {}
        results_full[dataset] = {}
        for method, seeds in methods.items():
            results[dataset][method] = {}
            results_full[dataset][method] = {}
            with open(get_config_file(method), "r") as fp:
                config = yaml.safe_load(fp)
            for seed, run_name in seeds.items():
                if method == "veinAttNet":
                    run_name = f"leaveoutds_veinAttNet_{dataset}_seed_{seed}"

                print(f"Ongoing: {method} {seed}: {run_name} on {dataset}")
                config["leaveoutds"] = dataset
                args = argparse.Namespace(
                    config=get_config_file(method),
                    checkpoint=run_name,
                    dataset=dataset,
                    logger_level="ERROR",
                    continue_model=None,
                )
                half_metrics, full_metrics = parallel_driver(args, config)
                results[dataset][method][seed] = half_metrics
                results_full[dataset][method][seed] = full_metrics
                print(
                    f"Dataset: {dataset} Method: {method} Seed: {seed} "
                    f"half={half_metrics} full={full_metrics}"
                )
                torch.cuda.empty_cache()

    with open("./ablation/half_subjects_results.json", "w+") as fp:
        json.dump(results, fp, indent=4)
    with open("./ablation/full_subjects_results.json", "w+") as fp:
        json.dump(results_full, fp, indent=4)


if __name__ == "__main__":
    final_runs_eval()
