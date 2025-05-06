import os
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


def driver(dataset, position=0):
    ENG = init_matlab_engine()

    print(f"Evaluating trained except {dataset}: on {dataset} 0")
    testCsv = f"./data/leaveoutds_{dataset}_seed_0/test.csv"

    df = pd.read_csv(testCsv)
    classes = set(df["Label"].values)
    cid_map = {}
    for label in classes:
        cid_map[label] = []

    for index, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc=f"Loading images for {dataset}",
        position=position,
        leave=True,
    ):
        label = row["Label"]
        path = str(row["Path"])
        path = path.replace(dataset, f"enhanced_{dataset}")

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
        if image.shape[0] < image.shape[1]:
            image = cv2.resize(image, (300, 100))
        else:
            image = cv2.resize(image, (100, 300))

        image = matlab.double(image)
        if image is None:
            print(f"Image not loaded: {path}")
            continue

        fvr = ENG.lee_region(image, 4, 40)
        cid_map[label].append((path, image, fvr))
        # img1 = img2 = image
        # fvr1 = fvr2 = fvr

    """
    RLT, MCP
    """
    mcp_scores = {"genuine": [], "imposter": []}
    rlt_scores = {"genuine": [], "imposter": []}
    # Genuine scores
    for cid1, images in tqdm(
        cid_map.items(),
        desc=f"Calculating {dataset} genuine scores",
        position=position+1,
        leave=True,
    ):
        for i, (path1, img1, fvr1) in enumerate(images):
            for j in range(i + 1, len(images)):
                path2, img2, fvr2 = images[j]
                # Calculate the scores
                try:
                    mcp_score = ENG.mcp(img1, fvr1, img2, fvr2)
                    mcp_scores["genuine"].append(mcp_score)
                except Exception as e:
                    print(f"Error calculating MCP score for {path1} and {path2}: {e}")
                    continue
                try:
                    rlt_score = ENG.rlt(img1, fvr1, img2, fvr2)
                    rlt_scores["genuine"].append(rlt_score)
                except Exception as e:
                    print(f"Error calculating RLT score for {path1} and {path2}: {e}")
                    continue
    os.makedirs(f"./tmp/mcp/{dataset}", exist_ok=True)
    os.makedirs(f"./tmp/rlt/{dataset}", exist_ok=True)

    np.savetxt(f"./tmp/mcp/{dataset}/genuine.txt", np.array(mcp_scores["genuine"]))
    np.savetxt(f"./tmp/rlt/{dataset}/genuine.txt", np.array(rlt_scores["genuine"]))
    # Imposter scores
    for cid1, images1 in tqdm(
        cid_map.items(),
        desc=f"Calculating {dataset} imposter scores",
        position=position+2,
        leave=True,
    ):
        for cid2, images2 in cid_map.items():
            for i, (path1, img1, fvr1) in enumerate(images1):
                for j, (path2, img2, fvr2) in enumerate(images2):
                    if cid1 == cid2:
                        continue
                    # Calculate the scores
                    try:
                        mcp_score = ENG.mcp(img1, fvr1, img2, fvr2)
                        mcp_scores["imposter"].append(mcp_score)
                    except Exception as e:
                        print(
                            f"Error calculating MCP score for {path1} and {path2}: {e}"
                        )
                        continue
                    try:
                        rlt_score = ENG.rlt(img1, fvr1, img2, fvr2)
                        rlt_scores["imposter"].append(rlt_score)
                    except Exception as e:
                        print(
                            f"Error calculating RLT score for {path1} and {path2}: {e}"
                        )
                        continue

    # Save the scores to CSV files
    np.savetxt(f"./tmp/mcp/{dataset}/imposter.txt", np.array(mcp_scores["imposter"]))
    np.savetxt(f"./tmp/rlt/{dataset}/imposter.txt", np.array(rlt_scores["imposter"]))

    mcp_df = pd.DataFrame(mcp_scores)
    mcp_df.to_csv(f"mcp_scores_{dataset}.csv", index=False)
    rlt_df = pd.DataFrame(rlt_scores)
    rlt_df.to_csv(f"rlt_scores_{dataset}.csv", index=False)

    ENG.quit()


def static_methods_loop():
    datasets = ["mmcbnu", "fvusm", "fv300", "polyu", "vera"]
    proceses = []
    for di, dataset in enumerate(datasets):
        p = Process(
            target=driver,
            args=(
                dataset,
                di*3,
            ),
        )
        p.start()
        proceses.append(p)
    for p in proceses:
        p.join()

        # Uncomment the following lines if you want to print the cid_map
        # with open(testCsv, "r") as file:
        #     reader = csv.reader(file)
        #     classes = set()
        #     cid_map = {}
        #     for row in reader:
        #         label = row["Label"]
        #         path = row["Path"]
        #         path = path.replace(dataset, f"enhanced_{dataset}")
        #         classes.add(label)
        #         if label in cid_map:
        #             cid_map[label].append(path)
        #         else:
        #             cid_map[label] = [path]


if __name__ == "__main__":
    static_methods_loop()
    # test()
