import csv
from tqdm import tqdm
import os
import random
from os.path import join as pjoin
from typing import Dict, List
from multiprocessing import Process

import numpy as np

# import pandas as pd
import torch
from torch.nn import CosineSimilarity

from utils import calculate_eer, initialise_dirs, logger, set_seeds


def get_files(
    path: str, dataset_name: str, opath: str | None = None
) -> Dict[str, List[str]]:
    """
    Get the files in the path.
    """

    files: Dict[str, List[str]] = {"Path": [], "Label": []}
    if opath:
        files["OPath"] = []
    num_classes = 0
    for subjectid in os.listdir(path):
        for filename in os.listdir(pjoin(path, subjectid)):
            files["Path"].append(
                pjoin(path, subjectid, filename),
            )
            files["Label"].append(
                dataset_name + "_" + subjectid,
            )
            if opath:
                files["OPath"].append(
                    pjoin(
                        opath,
                        dataset_name,
                        subjectid,
                        ".".join(filename.split(".")[:-1]) + ".txt",
                    )
                )
        num_classes += 1
    print(f"Dataset {dataset_name} has {num_classes} classes.")

    return files


def generate_dataset_csv() -> None:
    """
    Run the experiments.
    """
    datasets = ["fv300", "fvusm", "mmcbnu", "polyu", "vera"]

    for seed in range(4):
        for leaveoutdataset in datasets:
            train_set: Dict[str, List[str]] = {"Path": [], "Label": []}
            validation_set: Dict[str, List[str]] = {"Path": [], "Label": []}
            test_set: Dict[str, List[str]] = {"Path": [], "Label": [], "OPath": []}
            for dataset in datasets:
                if leaveoutdataset == dataset:
                    continue
                path = pjoin("./data", dataset, str(seed), "train")
                files = get_files(path, dataset)
                train_set["Path"].extend(files["Path"])
                train_set["Label"].extend(files["Label"])

                path = pjoin("./data", dataset, str(seed), "test")
                files = get_files(path, dataset)
                validation_set["Path"].extend(files["Path"])
                validation_set["Label"].extend(files["Label"])

            files = get_files(
                pjoin("./data", leaveoutdataset, str(seed), "test"),
                leaveoutdataset,
                "./features",
            )
            test_set["Path"].extend(files["Path"])
            test_set["Label"].extend(files["Label"])
            test_set["OPath"].extend(files["OPath"])

            files = get_files(
                pjoin("./data", leaveoutdataset, str(seed), "train"),
                leaveoutdataset,
                "./features",
            )
            test_set["Path"].extend(files["Path"])
            test_set["Label"].extend(files["Label"])
            test_set["OPath"].extend(files["OPath"])
            dir = pjoin(
                "./data", "leaveoutds_" + leaveoutdataset + "_seed_" + str(seed)
            )
            os.makedirs(dir, exist_ok=True)
            df = pd.DataFrame(train_set)
            df.to_csv(pjoin(dir, "train.csv"), index=False)
            df = pd.DataFrame(validation_set)
            df.to_csv(pjoin(dir, "validation.csv"), index=False)
            df = pd.DataFrame(test_set)
            df.to_csv(pjoin(dir, "test.csv"), index=False)


def compute_eer(rdir: str, model_name: str, dataset: str) -> None:
    """
    Compute the EER.
    """
    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/eval_{dataset}.log"
    log = logger.get_logger(model_name, logfile, "ERROR")

    set_seeds(log, 2025)  # hard coded as per experiments
    subjects_embeddings: Dict[str, List[torch.Tensor]] = {}
    # for identity in tqdm(os.listdir(rdir)):
    for identity in os.listdir(rdir):
        for sample in os.listdir(pjoin(rdir, identity)):
            if sample.endswith(".txt"):
                feature = np.loadtxt(pjoin(rdir, identity, sample))
                if identity not in subjects_embeddings:
                    subjects_embeddings[identity] = []
                subjects_embeddings[identity].append(torch.tensor(feature))

    genuine_scores = []
    imposter_scores = []
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)

    # for subject in tqdm(subjects_embeddings, desc="Calculating Genuine Scores"):
    for subject in subjects_embeddings:
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

    # for subject1 in tqdm(subjects_embeddings, desc="Calculating Imposter Scores"):
    for subject1 in subjects_embeddings:
        for subject2 in subjects_embeddings:
            if subject1 != subject2:
                maxn1 = min(4, len(subjects_embeddings[subject1]))
                maxn2 = min(4, len(subjects_embeddings[subject2]))
                for emb1 in random.sample(subjects_embeddings[subject1], maxn1):
                    for emb2 in random.sample(subjects_embeddings[subject2], maxn2):
                        if emb1.shape[0] != 1:
                            emb1 = emb1.unsqueeze(0)

                        if emb2.shape[0] != 1:
                            emb2 = emb2.unsqueeze(0)
                        sim = cosine_sim(emb1, emb2).squeeze()
                        imposter_scores.append(sim.item())

    os.makedirs(f"tmp/{model_name}/{dataset}", exist_ok=True)
    np.save(f"tmp/{model_name}/{dataset}/genuine_scores.npy", np.array(genuine_scores))
    np.save(
        f"tmp/{model_name}/{dataset}/imposter_scores.npy", np.array(imposter_scores)
    )

    log.info(f"Dataset: {dataset} Genuine Scores: {len(genuine_scores)}")
    log.info(f"Dataset: {dataset} Imposter Scores: {len(imposter_scores)}")

    eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    log.error(
        f"{model_name} Dataset: {dataset} (G,I) ({len(genuine_scores)},{len(imposter_scores)}) EER: {eer}"
    )
    np.save(f"tmp/{model_name}/{dataset}/far_scores.npy", far)
    np.save(f"tmp/{model_name}/{dataset}/frr_scores.npy", frr)


def get_eer_from_matlab_features() -> None:
    datasets = [
        "mmcbnu",
        "vera",
        "fvusm",
        "polyu",
        "fv300",
    ]
    temp = []
    for dataset in datasets:
        # if evaldataset != dataset:
        #     continue
        for seed in range(4):
            rdir = pjoin(
                f"/root/code/features/leaveout_{dataset}",
                str(seed),
                dataset,
            )
            temp.append(
                Process(
                    target=compute_eer,
                    args=(
                        rdir,
                        f"leaveoutds_veinAttNet_{dataset}_seed_{seed}",
                        dataset,
                    ),
                )
            )
            # break
            temp[-1].start()
    for p in temp:
        p.join()


if __name__ == "__main__":
    # generate_dataset_csv()
    # get_eer_from_matlab_features()
    import itertools

    g = [2, 4, 6, 8]
    print(list(itertools.product(g, g)))
