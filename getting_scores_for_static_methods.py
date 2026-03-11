from functools import lru_cache
from cachetools import cached
from cachetools.keys import hashkey
import math
from pathlib import Path
import os
import shutil
from typing import List, Tuple

import numpy as np
import scipy.io
from tqdm import tqdm
from multiprocessing import Pool

RDIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments"
ODIR = "/home/ubuntu/processed"
methods = ["mcp", "rlt", "wld"]
methods = ["mcp"]


def copy_files():
    RDIR = "/home/ubuntu/processed"
    for method in ["wld"]:
        for ds in ["fvusm", "mmcbnu", "polyu", "vera", "fv300"]:
            for ssplit in ["train", "test"]:
                rdir = os.path.join(RDIR, "unp_wld", ds, "0", ssplit)
                for class_name in tqdm(
                    os.listdir(rdir), desc=f"Copying files for {method} {ds} {ssplit}"
                ):
                    class_path = os.path.join(rdir, class_name)
                    if os.path.isdir(class_path):
                        for file_name in os.listdir(class_path):
                            if file_name.endswith(".mat"):
                                file_path = os.path.join(class_path, file_name)

                                new_file_name = f"{ssplit}_{file_name}"
                                new_file_path = os.path.join(
                                    ODIR, method, ds, class_name, new_file_name
                                )

                                # Ensure the destination directory exists
                                os.makedirs(
                                    os.path.dirname(new_file_path), exist_ok=True
                                )

                                # Move the file to the new location
                                shutil.copy(file_path, new_file_path)


@cached(cache={}, key=lambda x, cl, id, ds: hashkey(str(cl) + "_" + str(id) + "_" + ds))
def mean2(x, cl, id, ds):
    y = np.sum(x) / np.size(x)
    return x - y


def _corr2(a, b, cl1, cl2, id1, id2, ds):
    a = mean2(a, cl1, id1, ds)
    b = mean2(b, cl2, id2, ds)
    denominator = math.sqrt((a * a).sum() * (b * b).sum())
    if denominator == 0:
        return 0.0
    r = (a * b).sum() / denominator
    return r


def corr2(args):
    if args is None or len(args) != 7:
        print("Invalid arguments for corr2 function:", args)
        return None
    a, b, cl1, cl2, id1, id2, ds = args
    return _corr2(a, b, cl1, cl2, id1, id2, ds)


def load_class_data(class_path) -> Tuple[str, List[np.ndarray]]:
    files = [f for f in os.listdir(class_path) if f.endswith(".mat")]
    *_, class_name = Path(class_path).parts
    loaded_files = []
    for file in files:
        file_path = os.path.join(class_path, file)

        loaded_files.append(scipy.io.loadmat(file_path)["features"])
    return class_name, loaded_files


def genuine_generator(class_dict, ds):
    for class1 in class_dict:
        for id1, data1 in enumerate(class_dict[class1]):
            for id2, data2 in enumerate(class_dict[class1]):
                if id1 == id2:
                    continue
                yield data1, data2, class1, class1, id1, id2, ds
    return None


def imposter_generator(class_dict, ds):
    for class1 in class_dict:
        for class2 in class_dict:
            if class1 == class2:
                continue
            for id1, data1 in enumerate(class_dict[class1]):
                for id2, data2 in enumerate(class_dict[class2]):
                    yield data1, data2, class1, class2, id1, id2, ds
    return None


def get_scores():
    for method in ["wld"]:
        for ds in ["vera", "fv300", "fvusm", "mmcbnu", "polyu"]:
            class_dict = {}
            rdir = os.path.join(ODIR, method, ds)

            args = []
            for class_name in os.listdir(rdir):
                class_path = os.path.join(rdir, class_name)
                if os.path.isdir(class_path):
                    args.append(class_path)

            with Pool(38) as pool:
                results = list(
                    tqdm(
                        pool.imap(load_class_data, args),
                        total=len(args),
                        desc=f"Loading Data {method} {ds}",
                    )
                )

                for class_name, data in results:
                    class_dict[class_name] = data

                if not os.path.exists(os.path.join(f"./tmp/{method}/{ds}/genuine.txt")):
                    genuine_scores = []
                    comparator = genuine_generator(class_dict, ds)
                    results = list(
                        tqdm(
                            pool.imap(corr2, comparator, chunksize=10_000),
                        )
                    )

                    results = [score for score in results if score is not None]
                    genuine_scores.extend(results)
                    os.makedirs(f"./tmp/{method}/{ds}", exist_ok=True)
                    np.savetxt(
                        os.path.join(f"./tmp/{method}/{ds}/genuine.txt"),
                        genuine_scores,
                    )

                if not os.path.exists(
                    os.path.join(f"./tmp/{method}/{ds}/imposter.txt")
                ):
                    imposter_scores = []
                    comparator = imposter_generator(class_dict, ds)
                    results = list(
                        tqdm(
                            pool.imap(corr2, comparator, chunksize=10_000),
                        )
                    )

                    results = [score for score in results if score is not None]
                    imposter_scores.extend(results)
                    os.makedirs(f"./tmp/{method}/{ds}", exist_ok=True)
                    np.savetxt(
                        os.path.join(f"./tmp/{method}/{ds}/imposter.txt"),
                        imposter_scores,
                    )


# def copy_files():
#     for method in methods:
#         for ds in ["fv300", "fvusm", "mmcbnu", "polyu", "vera"]:
#             for classname in ["genuine", "imposter"]:
#                 fpath = os.path.join(
#                     "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/fingervein-datasetes/statistical_experiments/",
#                     f"{ds}_{method}_{classname}.txt",
#                 )
#                 if not os.path.exists(fpath):
#                     continue
#                 shutil.copy(fpath, f"./tmp/{method}/{ds}/{classname}.txt")


if __name__ == "__main__":
    copy_files()
    get_scores()
