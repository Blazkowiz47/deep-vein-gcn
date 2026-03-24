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
    subjects_embeddings: Dict[int, torch.Tensor],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    _, subjects = subjects_chunk

    for subject in subjects:
        embeddings = subjects_embeddings[subject].cuda()
        for id1 in range(embeddings.shape[0]):
            emb1 = embeddings[id1 : id1 + 1, :].cuda()
            sims = cosine_sim(emb1, embeddings).squeeze()
            sims = sims.detach().cpu()
            for id2, sim in enumerate(sims):
                if id1 != id2:
                    results.append(sim.item())
            del emb1
        embeddings = embeddings.detach().cpu()

    return results


def get_imposter_scores_batched(
    subjects_chunk: Tuple[int, List[int]],
    subjects_embeddings: Dict[int, torch.Tensor],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    _, subjects = subjects_chunk

    for subject1 in subjects:
        subjects_embeddings[subject1] = subjects_embeddings[subject1].cuda()
        for subject2 in subjects_embeddings:
            subjects_embeddings[subject2] = subjects_embeddings[subject2].cuda()
            for id1 in range(subjects_embeddings[subject1].shape[0]):
                if subject1 != subject2:
                    emb1 = subjects_embeddings[subject1][id1 : id1 + 1, :].cuda()
                    sims = cosine_sim(emb1, subjects_embeddings[subject2]).squeeze()
                    sims = sims.detach().cpu()
                    for sim in sims:
                        results.append(sim.item())
                    del emb1
            subjects_embeddings[subject2] = subjects_embeddings[subject2].detach().cpu()

        subjects_embeddings[subject1] = subjects_embeddings[subject1].detach().cpu()

    return results


def chunkify(lst: List[int], n: int) -> List[Tuple[int, List[int]]]:
    return [(i, lst[i::n]) for i in range(n)]


def parallel_driver(args, config) -> float:
    """
    Wrapper for the driver.
    """

    dataset = args.dataset
    model = config["model"]

    if model == "veinAttNet":
        checkpoint = "_".join(args.checkpoint.split("_")[1:])
        args.checkpoint = args.checkpoint.split("_")[0]
    else:
        checkpoint = args.checkpoint
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
    testds = wrapper.get_split("test", batch_size=16)

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
        path = os.path.join("./features/leaveout_" + args.checkpoint, args.dataset)
        for sid, subject in tqdm(
            enumerate(os.listdir(path)), desc="Fetching Embeddings"
        ):
            raw_subject_embeddings[sid] = []
            for fname in os.listdir(os.path.join(path, subject)):
                if fname.endswith(".txt"):
                    raw_subject_embeddings[sid].append(
                        torch.tensor(np.loadtxt(os.path.join(path, subject, fname)))
                        .squeeze()
                        .unsqueeze(0)
                    )

    for subject in raw_subject_embeddings:
        subject_embeddings[subject] = torch.cat(raw_subject_embeddings[subject], dim=0)

    genuine_scores: List[float] = []
    imposter_scores: List[float] = []
    log.error(f"Total subjects: {len(subject_embeddings)}")

    # with Pool(workers := 8) as p:
    workers = 8
    # partial_func = partial(
    #     get_genuine_scores_batched,
    #     subjects_embeddings=subject_embeddings,
    #     log=wrapper.log,
    # )
    chunkifiedsubjects = chunkify(list(subject_embeddings.keys()), workers)
    for chunk in tqdm(chunkifiedsubjects):
        genuine_scores.extend(
            get_genuine_scores_batched(chunk, subject_embeddings, wrapper.log)
        )
        imposter_scores.extend(
            get_imposter_scores_batched(chunk, subject_embeddings, wrapper.log)
        )

    # chunkified_genuine_scores = list(
    #     p.map(
    #         partial_func,
    #         chunkifiedsubjects,
    #     )
    # )
    # for cs in chunkified_genuine_scores:
    #     genuine_scores.extend(cs)

    log.error(f"Total genuine scores: {len(genuine_scores)}")
    # partial_func = partial(
    #     get_imposter_scores_batched,
    #     subjects_embeddings=subject_embeddings,
    #     log=wrapper.log,
    # )

    # chunkified_imposter_scores = list(
    #     p.map(
    #         partial_func,
    #         chunkifiedsubjects,
    #     )
    # )
    # for cs in chunkified_imposter_scores:
    #     imposter_scores.extend(cs)
    log.error(f"Total imposter scores: {len(imposter_scores)}")

    print("Saving Scores")
    os.makedirs(f"tmp/{model_name}/{dataset}", exist_ok=True)
    np.save(f"tmp/{model_name}/{dataset}/genuine_scores.npy", np.array(genuine_scores))
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_scores.npy", np.array(imposter_scores)
    )

    genuine_scores = np.load(f"tmp/{model_name}/{dataset}/genuine_scores.npy")
    imposter_scores = np.load(f"tmp/{model_name}/{dataset}/imposter_scores.npy")

    log.error(f"Total genuine scores: {len(genuine_scores)}")
    log.error(f"Total imposter scores: {len(imposter_scores)}")

    torch.cuda.empty_cache()
    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    log.error(f"{model_name} Dataset: {dataset} EER: {eer}")
    np.save(f"tmp/{model_name}/{dataset}/far_scores.npy", far)
    np.save(f"tmp/{model_name}/{dataset}/frr_scores.npy", frr)
    return eer


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    parallel_driver(args, config)
