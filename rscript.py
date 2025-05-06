import csv
import shutil
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


def check_runs():
    runs = [
        "dscgrapher_leaveoneout_03_03_25_18_01_2_219",
        "arcvein_leaveoneout_06_03_25_10_35_2_223",
        "lgfin_leaveoneout_06_03_25_13_04_2_236",
        "fvit_leaveoneout_07_03_25_11_11_2_201",
        "leaveoutds_veinAttNet_vera_seed_0",
        "dscgrapher_leaveoneout_24_04_25_02_57_2_253",
        "dscgrapher_leaveoneout_04_03_25_06_34_2_229",
        "arcvein_leaveoneout_06_03_25_10_35_2_225",
        "lgfin_leaveoneout_06_03_25_14_46_2_255",
        "fvit_leaveoneout_07_03_25_11_11_2_203",
        "leaveoutds_veinAttNet_vera_seed_1",
        "dscgrapher_leaveoneout_22_04_25_09_59_2_225",
        "dscgrapher_leaveoneout_05_03_25_14_05_2_231",
        "arcvein_leaveoneout_06_03_25_10_35_2_228",
        "lgfin_leaveoneout_06_03_25_16_30_2_210",
        "fvit_leaveoneout_07_03_25_11_11_2_207",
        "leaveoutds_veinAttNet_vera_seed_2",
        "dscgrapher_leaveoneout_22_04_25_16_23_2_230",
        "dscgrapher_leaveoneout_05_03_25_23_01_2_233",
        "arcvein_leaveoneout_06_03_25_10_35_2_235",
        "lgfin_leaveoneout_06_03_25_18_13_2_247",
        "fvit_leaveoneout_07_03_25_11_11_2_213",
        "leaveoutds_veinAttNet_vera_seed_3",
        "dscgrapher_leaveoneout_22_04_25_22_47_2_249",
        "dscgrapher_leaveoneout_05_03_25_00_54_2_253",
        "arcvein_leaveoneout_06_03_25_11_40_2_248",
        "lgfin_leaveoneout_07_03_25_13_35_2_248",
        "fvit_leaveoneout_07_03_25_12_54_2_207",
        "leaveoutds_veinAttNet_polyu_seed_0",
        "dscgrapher_leaveoneout_23_04_25_20_21_2_231",
        "dscgrapher_leaveoneout_04_03_25_19_22_2_251",
        "arcvein_leaveoneout_06_03_25_11_40_2_249",
        "lgfin_leaveoneout_07_03_25_13_35_2_249",
        "fvit_leaveoneout_07_03_25_12_54_2_210",
        "leaveoutds_veinAttNet_polyu_seed_1",
        "dscgrapher_leaveoneout_21_04_25_07_39_2_217",
        "dscgrapher_leaveoneout_04_03_25_07_18_2_232",
        "arcvein_leaveoneout_06_03_25_11_40_2_255",
        "lgfin_leaveoneout_07_03_25_19_18_2_248",
        "fvit_leaveoneout_07_03_25_12_54_2_214",
        "leaveoutds_veinAttNet_polyu_seed_2",
        "dscgrapher_leaveoneout_21_04_25_14_48_2_203",
        "dscgrapher_leaveoneout_03_03_25_18_01_2_221",
        "arcvein_leaveoneout_06_03_25_11_41_2_200",
        "lgfin_leaveoneout_07_03_25_19_18_2_249",
        "fvit_leaveoneout_07_03_25_12_54_2_220",
        "leaveoutds_veinAttNet_polyu_seed_3",
        "dscgrapher_leaveoneout_21_04_25_21_09_2_255",
        "dscgrapher_leaveoneout_04_03_25_18_13_2_231",
        "arcvein_leaveoneout_06_03_25_13_26_2_229",
        "lgfin_leaveoneout_07_03_25_05_34_2_201",
        "fvit_leaveoneout_07_03_25_14_18_2_225",
        "leaveoutds_veinAttNet_mmcbnu_seed_0",
        "dscgrapher_leaveoneout_23_04_25_14_48_2_203",
        "dscgrapher_leaveoneout_04_03_25_22_19_2_212",
        "arcvein_leaveoneout_06_03_25_13_26_2_230",
        "lgfin_leaveoneout_07_03_25_05_34_2_202",
        "fvit_leaveoneout_07_03_25_14_18_2_227",
        "leaveoutds_veinAttNet_mmcbnu_seed_1",
        "dscgrapher_leaveoneout_19_04_25_21_42_2_259",
        "dscgrapher_leaveoneout_05_03_25_02_44_2_223",
        "arcvein_leaveoneout_06_03_25_13_26_2_234",
        "lgfin_leaveoneout_07_03_25_05_34_2_206",
        "fvit_leaveoneout_07_03_25_14_18_2_230",
        "leaveoutds_veinAttNet_mmcbnu_seed_2",
        "dscgrapher_leaveoneout_20_04_25_08_01_2_244",
        "dscgrapher_leaveoneout_05_03_25_07_09_2_212",
        "arcvein_leaveoneout_06_03_25_13_26_2_241",
        "lgfin_leaveoneout_07_03_25_05_34_2_212",
        "fvit_leaveoneout_07_03_25_14_18_2_236",
        "leaveoutds_veinAttNet_mmcbnu_seed_3",
        "dscgrapher_leaveoneout_20_04_25_13_46_2_236",
        "dscgrapher_leaveoneout_04_03_25_18_33_2_251",
        "arcvein_leaveoneout_06_03_25_14_25_2_240",
        "lgfin_leaveoneout_06_03_25_17_05_2_224",
        "fvit_leaveoneout_07_03_25_14_47_2_226",
        "leaveoutds_veinAttNet_fvusm_seed_0",
        "dscgrapher_leaveoneout_18_04_25_11_49_2_227",
        "dscgrapher_leaveoneout_04_03_25_22_55_2_257",
        "arcvein_leaveoneout_06_03_25_14_25_2_242",
        "lgfin_leaveoneout_06_03_25_17_05_2_224",
        "fvit_leaveoneout_07_03_25_14_47_2_228",
        "leaveoutds_veinAttNet_fvusm_seed_1",
        "dscgrapher_leaveoneout_18_04_25_18_13_2_209",
        "dscgrapher_leaveoneout_05_03_25_03_18_2_232",
        "arcvein_leaveoneout_06_03_25_14_25_2_246",
        "lgfin_leaveoneout_06_03_25_17_05_2_227",
        "fvit_leaveoneout_07_03_25_14_47_2_232",
        "leaveoutds_veinAttNet_fvusm_seed_2",
        "dscgrapher_leaveoneout_19_04_25_02_38_2_246",
        "dscgrapher_leaveoneout_05_03_25_07_40_2_224",
        "arcvein_leaveoneout_06_03_25_14_25_2_252",
        "lgfin_leaveoneout_06_03_25_17_05_2_233",
        "fvit_leaveoneout_07_03_25_14_47_2_238",
        "leaveoutds_veinAttNet_fvusm_seed_3",
        "dscgrapher_leaveoneout_19_04_25_10_40_2_239",
        "dscgrapher_leaveoneout_05_03_25_13_51_2_243",
        "arcvein_leaveoneout_06_03_25_15_22_2_228",
        "lgfin_leaveoneout_06_03_25_15_53_2_211",
        "fvit_leaveoneout_07_03_25_15_13_2_222",
        "leaveoutds_veinAttNet_fv300_seed_0",
        "dscgrapher_leaveoneout_24_04_25_09_59_2_251",
        "dscgrapher_leaveoneout_05_03_25_14_37_2_211",
        "arcvein_leaveoneout_06_03_25_15_22_2_229",
        "lgfin_leaveoneout_06_03_25_15_53_2_212",
        "fvit_leaveoneout_07_03_25_15_13_2_224",
        "leaveoutds_veinAttNet_fv300_seed_1",
        "dscgrapher_leaveoneout_23_04_25_07_20_2_231",
        "dscgrapher_leaveoneout_05_03_25_15_22_2_245",
        "arcvein_leaveoneout_06_03_25_15_22_2_233",
        "lgfin_leaveoneout_06_03_25_15_53_2_217",
        "fvit_leaveoneout_07_03_25_15_13_2_228",
        "leaveoutds_veinAttNet_fv300_seed_2",
        "dscgrapher_leaveoneout_23_04_25_09_27_2_255",
        "dscgrapher_leaveoneout_05_03_25_16_07_2_245",
        "arcvein_leaveoneout_06_03_25_15_22_2_240",
        "lgfin_leaveoneout_06_03_25_15_53_2_223",
        "fvit_leaveoneout_07_03_25_15_13_2_234",
        "leaveoutds_veinAttNet_fv300_seed_3",
        "dscgrapher_leaveoneout_23_04_25_11_31_2_247",
        # stem ablation
        "dscgrapher_leaveoneout_04_04_25_11_21_2_236",
        "dscgrapher_leaveoneout_04_04_25_14_58_2_255",
        "dscgrapher_leaveoneout_04_04_25_19_22_2_231",
        "dscgrapher_leaveoneout_05_04_25_00_07_2_209",
        "dscgrapher_leaveoneout_05_04_25_05_46_2_242",
        "dscgrapher_leaveoneout_05_04_25_10_56_2_246",
        "dscgrapher_leaveoneout_05_04_25_15_15_2_257",
        "dscgrapher_leaveoneout_05_04_25_20_09_2_210",
        # backbone ablation
        "dscgrapher_leaveoneout_07_04_25_15_36_2_241",
        "dscgrapher_leaveoneout_07_04_25_20_06_2_225",
        "dscgrapher_leaveoneout_08_04_25_00_48_2_253",
        "dscgrapher_leaveoneout_08_04_25_06_16_2_246",
        "dscgrapher_leaveoneout_08_04_25_17_14_2_237",
        "dscgrapher_leaveoneout_09_04_25_01_47_2_255",
        "dscgrapher_leaveoneout_09_04_25_10_49_2_216",
        "dscgrapher_leaveoneout_09_04_25_17_59_2_227",
        "dscgrapher_leaveoneout_10_04_25_04_14_2_211",
        "dscgrapher_leaveoneout_10_04_25_13_05_2_208",
        "dscgrapher_leaveoneout_10_04_25_23_14_2_243",
        "dscgrapher_leaveoneout_11_04_25_10_02_2_222",
        "dscgrapher_leaveoneout_11_04_25_20_38_2_247",
        "dscgrapher_leaveoneout_12_04_25_02_38_2_247",
        "dscgrapher_leaveoneout_12_04_25_09_01_2_224",
        "dscgrapher_leaveoneout_12_04_25_16_03_2_224",
    ]
    avoid = [
        "dscgrapher_leaveoneout_03_03_25_18_01_2_219",
        "dscgrapher_leaveoneout_04_03_25_06_34_2_229",
        "dscgrapher_leaveoneout_05_03_25_14_05_2_231",
        "dscgrapher_leaveoneout_05_03_25_23_01_2_233",
        "dscgrapher_leaveoneout_05_03_25_00_54_2_253",
        "dscgrapher_leaveoneout_04_03_25_19_22_2_251",
        "dscgrapher_leaveoneout_04_03_25_07_18_2_232",
        "dscgrapher_leaveoneout_03_03_25_18_01_2_221",
        "dscgrapher_leaveoneout_04_03_25_18_13_2_231",
        "dscgrapher_leaveoneout_04_03_25_22_19_2_212",
        "dscgrapher_leaveoneout_05_03_25_02_44_2_223",
        "dscgrapher_leaveoneout_05_03_25_07_09_2_212",
        "dscgrapher_leaveoneout_04_03_25_18_33_2_251",
        "dscgrapher_leaveoneout_04_03_25_22_55_2_257",
        "dscgrapher_leaveoneout_05_03_25_03_18_2_232",
        "dscgrapher_leaveoneout_05_03_25_07_40_2_224",
        "dscgrapher_leaveoneout_05_03_25_13_51_2_243",
        "dscgrapher_leaveoneout_05_03_25_14_37_2_211",
        "dscgrapher_leaveoneout_05_03_25_15_22_2_245",
        "dscgrapher_leaveoneout_05_03_25_16_07_2_245",
    ]
    print(len(runs), len(avoid))
    exit()
    for run in tqdm(runs):
        if run in avoid:
            continue
        os.system(f"cp -r tmp/{run} final_runs/{run}")


def copy_enhanced_fvusm():
    rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/fvusm"
    odir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/enhanced_fvusm"
    edir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/Published_database_FV-USM_Dec2013/Enhanced"
    for seed in range(5):
        for cid in ["test", "train"]:
            trdir = pjoin(rdir, str(seed), cid)
            for subjectid in tqdm(os.listdir(trdir)):
                for filename in os.listdir(pjoin(trdir, subjectid)):
                    if filename.endswith(".jpg"):
                        os.makedirs(
                            pjoin(odir, str(seed), cid, subjectid), exist_ok=True
                        )
                        if "test" in filename:
                            shutil.copy(
                                pjoin(edir, "test", subjectid, filename.split("_")[-1]),
                                pjoin(odir, str(seed), cid, subjectid, filename),
                            )
                        else:
                            shutil.copy(
                                pjoin(
                                    edir, "train", subjectid, filename.split("_")[-1]
                                ),
                                pjoin(odir, str(seed), cid, subjectid, filename),
                            )


def clean_final_runs():
    for run in tqdm(os.listdir("final_runs")):
        files = list(os.listdir(pjoin("final_runs", run)))
        for file in files:
            if file == "checkpoints":
                for chkpts in os.listdir(pjoin("final_runs", run, file)):
                    if "best" in chkpts:
                        os.system(
                            f"cp final_runs/{run}/{file}/{chkpts} final_runs/{run}/{chkpts}"
                        )
                os.system(f"rm -rf final_runs/{run}/{file}")
            else:
                os.system(f"rm -rf final_runs/{run}/{file}")


def check_final_runs():
    for run in tqdm(os.listdir("final_runs")):
        files = list(os.listdir(pjoin("final_runs", run)))
        if len(files) == 1 and "best" in files[0]:
            continue
        else:
            print(run)


def clean_tmp_runs():
    final_runs = list(os.listdir("final_runs"))
    for run in tqdm(os.listdir("tmp")):
        if run in final_runs:
            continue
        else:
            os.system(f"sudo rm -rf tmp/{run}")


if __name__ == "__main__":
    # check_runs()
    # clean_final_runs()
    # copy_enhanced_fvusm()
    # check_final_runs()
    clean_tmp_runs()
