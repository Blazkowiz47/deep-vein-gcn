import argparse
from functools import partial
from multiprocessing import Pool
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import torch
import os
from torch.nn import CosineSimilarity, Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from cdatasets import get_dataset
from models import get_model
from utils import initialise_dirs, logger, set_seeds, calculate_eer
from logging import Logger

parser = argparse.ArgumentParser(
    description="Training Config",
    add_help=True,
)


parser.add_argument(
    "-c",
    "--config",
    # default="./configs/dscgrapher.yaml",
    default="./configs/dscgrapher2.yaml",
    # default="./configs/arcvein.yaml",
    type=str,
    help="Train config file.",
)


parser.add_argument(
    "-d",
    "--dataset",
    default="fvusm",
    type=str,
    help="""
    Give a single dataset name or multiple datasets to chain together.
    eg: -d fv300
    """,
)

parser.add_argument(
    "-ckpt",
    "--checkpoint",
    type=str,
    default=None,
    help="Load initial weights from partially/pretrained model.",
)


parser.add_argument(
    "--logger-level",
    type=str,
    default="ERROR",
    help="Logger level",
)


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


def get_genuine_scores_batched(
    subjects_chunk: Tuple[int, List[int]],
    subjects_embeddings: Dict[int, List[torch.Tensor]],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    _, subjects = subjects_chunk
    for subject in subjects:
        maxn = min(10, len(subjects_embeddings[subject]))
        x = list(range(len(subjects_embeddings[subject])))
        for emb1id in random.sample(x, maxn):
            for emb2id in random.sample(x, maxn):
                if emb1id == emb2id:
                    continue
                emb1 = subjects_embeddings[subject][emb1id]
                emb2 = subjects_embeddings[subject][emb2id]
                if emb1.shape[0] != 1:
                    emb1 = emb1.unsqueeze(0)

                if emb2.shape[0] != 1:
                    emb2 = emb2.unsqueeze(0)
                sim = cosine_sim(emb1, emb2).squeeze()
                results.append(sim.item())
    return results


def get_imposter_scores_batched(
    subjects_chunk: Tuple[int, List[int]],
    subjects_embeddings: Dict[int, List[torch.Tensor]],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    _, subjects = subjects_chunk
    for subject1 in subjects:
        for subject2 in subjects_embeddings:
            if subject1 != subject2:
                maxn1 = min(3, len(subjects_embeddings[subject1]))
                maxn2 = min(3, len(subjects_embeddings[subject2]))
                for emb1id in random.sample(
                    list(range(len(subjects_embeddings[subject1]))), maxn1
                ):
                    for emb2id in random.sample(
                        list(range(len(subjects_embeddings[subject2]))), maxn2
                    ):
                        emb1 = subjects_embeddings[subject1][emb1id]
                        emb2 = subjects_embeddings[subject2][emb2id]
                        if emb1.shape[0] != 1:
                            emb1 = emb1.unsqueeze(0)

                        if emb2.shape[0] != 1:
                            emb2 = emb2.unsqueeze(0)
                        sim = cosine_sim(emb1, emb2).squeeze()
                        results.append(sim.item())
    return results


def chunkify(lst: List[int], n: int) -> List[Tuple[int, List[int]]]:
    return [(i, lst[i::n]) for i in range(n)]


def parallel_driver(args, config)->float:
    """
    Wrapper for the driver.
    """

    checkpoint = args.checkpoint
    dataset = args.dataset
    model = config["model"]
    model_name = checkpoint
    checkpoint = os.path.join("./tmp", checkpoint, "checkpoints", "best_model.pt")

    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/eval_{dataset}.log"
    log = logger.get_logger(model_name, logfile, args.logger_level)

    set_seeds(log, config["seed"])
    device = config["device"]  # You can change this to cpu.

    config["leaveoutds"] = args.dataset
    leaveoutds = get_dataset("leaveoneout", config, log)
    config["num_classes"] = leaveoutds.num_classes

    wrapper = get_dataset(args.dataset, config, log, partition_split=0)
    testds = wrapper.get_split("test")

    model = get_model(model, config, log).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
    model.eval()
    model.to(device)
    log.info(str(model))

    with torch.no_grad():
        subject_embeddings: Dict[int, List[torch.Tensor]] = {}
        for images, labels in tqdm(testds, desc="Fetching Embeddings", position=0):
            preds = model(images.to(device)).detach().cpu()
            labels = labels.argmax(dim=1).numpy().tolist()
            for pred, label in zip(preds, labels):
                if label not in subject_embeddings:
                    subject_embeddings[label] = []

                subject_embeddings[label].append(pred.squeeze().unsqueeze(0))

    genuine_scores: List[float] = []
    imposter_scores: List[float] = []

    with Pool(workers := 8) as p:
        partial_func = partial(
            get_genuine_scores_batched,
            subjects_embeddings=subject_embeddings,
            log=wrapper.log,
        )
        chunkified_genuine_scores = p.map(
            partial_func,
            chunkifiedsubjects := chunkify(list(subject_embeddings.keys()), workers),
        )
        for cs in chunkified_genuine_scores:
            genuine_scores.extend(cs)
        partial_func = partial(
            get_imposter_scores_batched,
            subjects_embeddings=subject_embeddings,
            log=wrapper.log,
        )
        chunkified_imposter_scores = p.map(
            partial_func,
            chunkifiedsubjects,
        )
        for cs in chunkified_imposter_scores:
            imposter_scores.extend(cs)

    print("Saving Scores")
    log.error(f"Total genuine scores: {len(genuine_scores)}")
    log.error(f"Total imposter scores: {len(imposter_scores)}")
    os.makedirs(f"tmp/{model_name}/{dataset}", exist_ok=True)
    np.save(f"tmp/{model_name}/{dataset}/genuine_scores.npy", np.array(genuine_scores))
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_scores.npy", np.array(imposter_scores)
    )

    log.info(f"Dataset: {dataset} Genuine Scores: {len(genuine_scores)}")
    log.info(f"Dataset: {dataset} Imposter Scores: {len(imposter_scores)}")

    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    log.error(f"{model_name} Dataset: {dataset} EER: {eer}")
    np.save(f"tmp/{model_name}/{dataset}/far_scores.npy", far)
    np.save(f"tmp/{model_name}/{dataset}/frr_scores.npy", frr)
    return eer


def driver(args, config):
    """
    Wrapper for the driver.
    """

    checkpoint = args.checkpoint
    dataset = args.dataset
    model = config["model"]
    model_name = checkpoint
    checkpoint = os.path.join("./tmp", checkpoint, "checkpoints", "best_model.pt")

    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/eval_{dataset}.log"
    log = logger.get_logger(model_name, logfile, args.logger_level)

    set_seeds(log, config["seed"])
    device = config["device"]  # You can change this to cpu.

    config["leaveoutds"] = args.dataset
    leaveoutds = get_dataset("leaveoneout", config, log)
    config["num_classes"] = leaveoutds.num_classes

    wrapper = get_dataset(args.dataset, config, log, partition_split=0)
    _ = wrapper.get_split("validation")

    model = get_model(model, config, log).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
    model.eval()
    model.to(device)
    log.info(str(model))
    subjects_samples = wrapper.test_data

    with torch.no_grad():
        subjects_embeddings: Dict[str, List[torch.Tensor]] = {}
        for subject in tqdm(subjects_samples, desc="Extracting Embeddings"):
            i = 0
            for sample in subjects_samples[subject]:
                if subject not in subjects_embeddings:
                    subjects_embeddings[subject] = []
                img = (
                    transform(sample, size=(config["height"], config["width"]))
                    .unsqueeze(0)
                    .to(device)
                )
                emb = model(img, features=True).detach().cpu()
                subjects_embeddings[subject].append(emb)
                i += 1
                if i == 3:
                    continue

    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    genuine_scores: List[float] = []
    imposter_scores: List[float] = []

    for subject in tqdm(subjects_embeddings, desc="Calculating Genuine Scores"):
        maxn = min(10, len(subjects_embeddings[subject]))
        for emb1 in random.sample(subjects_embeddings[subject], maxn):
            for emb2 in random.sample(subjects_embeddings[subject], maxn):
                if emb1.shape[0] != 1:
                    emb1 = emb1.unsqueeze(0)

                if emb2.shape[0] != 1:
                    emb2 = emb2.unsqueeze(0)
                if (emb1 == emb2).all():
                    continue
                sim = cosine_sim(emb1, emb2).squeeze()
                genuine_scores.append(sim.item())

    for subject1 in tqdm(subjects_embeddings, desc="Calculating Imposter Scores"):
        for subject2 in subjects_embeddings:
            if subject1 != subject2:
                maxn1 = min(3, len(subjects_embeddings[subject1]))
                maxn2 = min(3, len(subjects_embeddings[subject2]))
                for emb1 in random.sample(subjects_embeddings[subject1], maxn1):
                    for emb2 in random.sample(subjects_embeddings[subject2], maxn2):
                        if emb1.shape[0] != 1:
                            emb1 = emb1.unsqueeze(0)

                        if emb2.shape[0] != 1:
                            emb2 = emb2.unsqueeze(0)
                        sim = cosine_sim(emb1, emb2).squeeze()
                        imposter_scores.append(sim.item())

    print("Saving Scores")
    log.error(f"Total genuine scores: {len(genuine_scores)}")
    log.error(f"Total imposter scores: {len(imposter_scores)}")
    os.makedirs(f"tmp/{model_name}/{dataset}", exist_ok=True)
    np.save(f"tmp/{model_name}/{dataset}/genuine_scores.npy", np.array(genuine_scores))
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_scores.npy", np.array(imposter_scores)
    )

    log.info(f"Dataset: {dataset} Genuine Scores: {len(genuine_scores)}")
    log.info(f"Dataset: {dataset} Imposter Scores: {len(imposter_scores)}")

    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    log.error(f"{model_name} Dataset: {dataset} EER: {eer}")
    np.save(f"tmp/{model_name}/{dataset}/far_scores.npy", far)
    np.save(f"tmp/{model_name}/{dataset}/frr_scores.npy", frr)


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    parallel_driver(args, config)
