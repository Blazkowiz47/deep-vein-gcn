import os
import random
from multiprocessing import Pool, Process
import cv2
import json
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import matlab
import matlab.engine
from multiprocessing import set_start_method

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    print("Spawn method already set or not available. Proceeding with default method.")
    pass


def init_matlab_engine() -> matlab.engine.MatlabEngine:
    global ENG
    print("Starting MATLAB engine...")
    ENG = matlab.engine.start_matlab()
    if not isinstance(ENG, matlab.engine.MatlabEngine):
        raise Exception("Failed to start MATLAB engine.")
    ENG.addpath(ENG.genpath(os.path.abspath(".")), nargout=0)
    if isinstance(ENG, matlab.engine.MatlabEngine):
        print("MATLAB engine started successfully.")
        return ENG
    raise Exception("Failed to start MATLAB engine.")


def test():
    ENG = init_matlab_engine()
    # Test the MATLAB engine with a simple command
    try:
        result = ENG.eval("disp('MATLAB engine is working!')", nargout=0)
        print(result)
        # Test a MATLAB function
        path1 = "./data/enhanced_polyu/0/test/27/36_4_f1_1.bmp"
        path2 = "./data/enhanced_polyu/0/test/177/156_4_f1_1.bmp"
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE) / 255.0
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE) / 255.0
        img1 = cv2.resize(img1, (300, 100))
        img2 = cv2.resize(img2, (300, 100))

        fvr1 = ENG.lee_region(matlab.double(img1), 4, 40)
        fvr2 = ENG.lee_region(matlab.double(img2), 4, 40)
        nfvr1 = np.asarray(fvr1)
        nfvr2 = np.asarray(fvr2)
        print(f"Image 1 shape: {img1.shape}")
        print(f"FVR1: {nfvr1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        print(f"FVR2: {nfvr2.shape}")
        score = ENG.mcp(matlab.double(img1), fvr1, matlab.double(img2), fvr2)

    except Exception as e:
        print(f"Error during MATLAB engine test: {e}")
    finally:
        ENG.quit()


def load_fv_images(path, label):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
    if image.shape[0] < image.shape[1]:
        image = cv2.resize(image, (300, 100))
    else:
        image = cv2.resize(image, (100, 300))

    image = matlab.double(image)
    if image is None:
        print(f"Image not loaded: {path}")
        return

    return path, label, image


def driver(dataset, position=0):
    print(f"Evaluating trained except {dataset}: on {dataset} 0")
    testCsv = f"./data/leaveoutds_{dataset}_seed_0/test.csv"

    df = pd.read_csv(testCsv)
    classes = set(df["Label"].values)
    cid_map = {}
    for label in classes:
        cid_map[label] = []

    args = []
    for index, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc=f"Loading images for {dataset}",
        # position=position,
        leave=True,
    ):
        label = row["Label"]
        path = str(row["Path"])
        mcp_path = (
            path.replace("data", "data/mcp")
            .replace("/0/", "/")
            .replace(".bmp", ".mat")
            .replace(".jpg", ".mat")
            .replace(".png", ".mat")
            .replace(".jpeg", ".mat")
            .replace(
                "./data",
                "E:/filestorage/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments",
            )
        )
        rlt_path = (
            path.replace("data", "data/rlt")
            .replace("/0/", "/")
            .replace(".bmp", ".mat")
            .replace(".jpg", ".mat")
            .replace(".png", ".mat")
            .replace(".jpeg", ".mat")
            .replace(
                "./data",
                "E:/filestorage/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments",
            )
        )
        wld_path = (
            path.replace("data", "data/wld")
            .replace("/0/", "/")
            .replace(".bmp", ".mat")
            .replace(".jpg", ".mat")
            .replace(".png", ".mat")
            .replace(".jpeg", ".mat")
            .replace(
                "./data",
                "E:/filestorage/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments",
            )
        )
        cid_map[label].append((mcp_path, rlt_path, wld_path))

    # init_matlab_engine()
    # img1 = img2 = image
    # fvr1 = fvr2 = fvr

    """
    RLT, MCP
    """
    # mcp_scores = {"genuine": [], "imposter": []}
    # rlt_scores = {"genuine": [], "imposter": []}
    args = {"mcp": [[], []], "rlt": [[], []], "wld": [[], []]}

    # Genuine scores
    for cid1, images in tqdm(
        cid_map.items(),
        desc=f"Calculating {dataset} genuine scores",
        # position=position + 1,
        leave=True,
    ):
        for i, path1 in enumerate(images):
            for j, path2 in enumerate(images):
                if i == j:
                    continue
                # Calculate the scores
                args["mcp"][0].append(path1[0])
                args["mcp"][1].append(path2[0])
                args["rlt"][0].append(path1[1])
                args["rlt"][1].append(path2[1])
                args["wld"][0].append(path1[2])
                args["wld"][1].append(path2[2])

    with open(
        f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/{dataset}_mcp_genuine.csv",
        "w+",
    ) as f:
        f.write("path1,path2\n")
        for path1, path2 in zip(args["mcp"][0], args["mcp"][1]):
            f.write(f"{path1},{path2}\n")

    with open(
        f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/{dataset}_rlt_genuine.csv",
        "w+",
    ) as f:
        f.write("path1,path2\n")
        for path1, path2 in zip(args["rlt"][0], args["rlt"][1]):
            f.write(f"{path1},{path2}\n")

    with open(
        f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/{dataset}_wld_genuine.csv",
        "w+",
    ) as f:
        f.write("path1,path2\n")
        for path1, path2 in zip(args["wld"][0], args["wld"][1]):
            f.write(f"{path1},{path2}\n")
    # init_matlab_engine()
    # mcp_scores = ENG.get_scores(args["mcp"][0], args["mcp"][1])
    # rlt_scores = ENG.get_scores(args["rlt"][0], args["rlt"][1])
    # wld_scores = ENG.get_scores(args["wld"][0], args["wld"][1])
    # mcp_scores = ENG.mcp(*args)
    # print(mcp_scores)
    # rlt_scores = ENG.rlt(*args)
    # os.makedirs(f"./tmp/mcp/{dataset}", exist_ok=True)
    # os.makedirs(f"./tmp/rlt/{dataset}", exist_ok=True)
    # os.makedirs(f"./tmp/wld/{dataset}", exist_ok=True)
    #
    # with open(f"./tmp/mcp/{dataset}/genuine.txt", "w+") as f:
    #     for score in mcp_scores:
    #         f.write(f"{score}\n")
    # with open(f"./tmp/rlt/{dataset}/genuine.txt", "w+") as f:
    #     for score in rlt_scores:
    #         f.write(f"{score}\n")
    # with open(f"./tmp/wld/{dataset}/genuine.txt", "w+") as f:
    #     for score in wld_scores:
    #         f.write(f"{score}\n")

    # np.savetxt(f"./tmp/mcp/{dataset}/genuine.txt", np.array(total_mcp_scores))
    # np.savetxt(f"./tmp/rlt/{dataset}/genuine.txt", np.array(total_rlt_scores))

    # Imposter scores


def wld():
    datasets = ["fv300", "polyu", "fvusm", "mmcbnu", "vera"]
    rpath = []
    opath = []
    for dataset in datasets:
        rdir = f"./data/enhanced_{dataset}/0"
        for root, _, files in os.walk(rdir):
            for file in files:
                if file.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")):
                    path = os.path.join(root, file)
                    rpath.append(path)
                    opath.append(
                        path.replace("enhanced_", "unp_wld/")
                        .replace(".bmp", ".mat")
                        .replace(".jpg", ".mat")
                        .replace(".png", ".mat")
                        .replace(".jpeg", ".mat")
                        .replace(".BMP", ".mat")
                        .replace(".JPG", ".mat")
                        .replace(".PNG", ".mat")
                        .replace(".JPEG", ".mat")
                        .replace(
                            "./data",
                            "/home/ubuntu/processed",
                        )
                    )
                    os.makedirs(
                        os.path.dirname(opath[-1]), exist_ok=True
                    )

    ENG.wld(rpath, opath)


def static_methods_loop():
    datasets = ["fv300", "polyu", "fvusm", "mmcbnu", "vera"]
    proceses = []
    for di, dataset in enumerate(datasets):
        driver(dataset, di)


if __name__ == "__main__":
    ENG = None
    init_matlab_engine()
    wld()

    # static_methods_loop()
    # test()
