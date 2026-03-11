from functools import partial
from logging import Logger
from multiprocessing import Pool, set_start_method
from typing import Dict, List, Tuple

import torch
import random
from PIL import Image
from torch.nn import CosineSimilarity, Module
from tqdm import tqdm

from .common_functions import Wrapper
from .metrics import calculate_eer

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass


def loadimg(fname: str, wrapper: Wrapper) -> torch.Tensor:
    img = Image.open(fname)
    img, _ = wrapper.transform((img, 0))
    return img


def loadimgembbatch(
    subjects_chunk: Tuple[int, List[str]],
    subjects_samples: Dict[str, List[str]],
    wrapper: Wrapper,
    model: Module,
    device: str,
    log: Logger,
) -> Dict[str, List[torch.Tensor]]:
    results = {}
    chunkid, subjects = subjects_chunk
    for k in tqdm(subjects, desc=f"Chunk {chunkid}", leave=True, position=chunkid):
        results[k] = []
        log.debug(f"Processing {k}")
        for fname in subjects_samples[k]:
            log.debug(f"Processing {fname}")
            img = loadimg(fname, wrapper).unsqueeze(0).to(device)
            results[k].append(model(img).detach().cpu())

    log.debug(f"Finished processing {len(subjects)} subjects")

    return results


def get_genuine_scores_batched(
    subjects_chunk: Tuple[int, List[int]],
    subjects_embeddings: Dict[int, List[torch.Tensor]],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    chunkid, subjects = subjects_chunk
    for subject in tqdm(
        subjects, desc=f"Calculating Genuine Scores {chunkid}", position=chunkid
    ):
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
                results.append(sim.item())
    return results


def get_imposter_scores_batched(
    subjects_chunk: Tuple[int, List[int]],
    subjects_embeddings: Dict[int, List[torch.Tensor]],
    log: Logger,
) -> List[float]:
    results: List[float] = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    chunkid, subjects = subjects_chunk
    for subject1 in tqdm(
        subjects, desc=f"Calculating Imposter Scores {chunkid}", position=chunkid
    ):
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
                        results.append(sim.item())
    return results


def chunkify(lst: List[int], n: int) -> List[Tuple[int, List[int]]]:
    return [(i, lst[i::n]) for i in range(n)]


def compute_eer_mp(
    model: Module,
    wrapper: Wrapper,
    workers: int = 3,
    data_split: str = "validation",
    device: str = "cuda",
) -> Tuple[float, List[float], List[float]]:
    """
    Computes eer with multiprocessing
    Returns (eer, genuine_scores, impostor_scores)
    """
    testds = wrapper.get_split(data_split)

    model.eval()
    model.to(device)
    subject_embeddings: Dict[int, List[torch.Tensor]] = {}
    for images, labels in tqdm(testds, desc="Fetching Embeddings", position=0):
        preds = model(images.to(device)).detach().cpu()
        labels = labels.argmax(dim=1)
        for i, label in enumerate(labels):
            if label not in subject_embeddings:
                subject_embeddings[label] = []
            subject_embeddings[label].append(preds[i].squeeze())

    with Pool(workers) as p:
        partial_func = partial(
            get_genuine_scores_batched,
            subjects_embeddings=subject_embeddings,
            log=wrapper.log,
        )
        chunkified_genuine_scores = p.map(
            partial_func,
            chunkifiedsubjects := chunkify(list(subject_embeddings.keys()), workers),
        )
        genuine_scores: List[float] = []
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
        imposter_scores: List[float] = []
        for cs in chunkified_imposter_scores:
            imposter_scores.extend(cs)
        wrapper.log.info("Genuine Scores:", len(genuine_scores))
        wrapper.log.info("Imposter Scores:", len(imposter_scores))
    return (
        calculate_eer(genuine_scores, imposter_scores)[0],
        genuine_scores,
        imposter_scores,
    )
