from logging import getLogger
import itertools
from matplotlib import cm
from run_name_mappings import final_runs
import os
import re
from typing import Dict, Tuple
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import calculate_eer, set_seeds


def get_eers() -> Dict[str, Dict[str, Dict[int, Tuple[float, float, float, float]]]]:
    scores_dict = {}
    for dataset, data in final_runs.items():
        scores_dict[dataset] = {}
        for method, runs in data.items():
            scores_dict[dataset][method] = {}
            for seed, run_name in runs.items():
                fmr = np.load(f"tmp/{run_name}/{dataset}/far_scores.npy")
                fnmr = np.load(f"tmp/{run_name}/{dataset}/frr_scores.npy")
                tar1 = 100 - fnmr[np.argmin(np.abs(fmr - 1.0))]
                tar01 = 100 - fnmr[np.argmin(np.abs(fmr - 0.1))]
                tar001 = 100 - fnmr[np.argmin(np.abs(fmr - 0.01))]

                eval_log_file = f"tmp/{run_name}/eval_{dataset}.log"
                with open(eval_log_file, "r") as f:
                    lines = f.readlines()
                eer_line = lines[-1].strip()
                eer_match = re.search(r"EER:\s*([0-9.]+)", eer_line)
                if eer_match:
                    eer_value = float(eer_match.group(1))
                    scores_dict[dataset][method][seed] = (
                        eer_value,
                        tar1,
                        tar01,
                        tar001,
                    )
                else:
                    print(f"No EER found in line: {eer_line}")

        for method in ["mcp", "rlt", "wld"]:
            scores_dict[dataset][method] = {}

            fmr = np.load(f"tmp/{method}/{dataset}/far_scores.npy")
            fnmr = np.load(f"tmp/{method}/{dataset}/frr_scores.npy")
            tar1 = 100 - fnmr[np.argmin(np.abs(fmr - 1.0))]
            tar01 = 100 - fnmr[np.argmin(np.abs(fmr - 0.1))]
            tar001 = 100 - fnmr[np.argmin(np.abs(fmr - 0.01))]
            eval_log_file = f"tmp/{method}/eval_{dataset}.log"
            with open(eval_log_file, "r") as f:
                lines = f.readlines()

            eer_line = lines[-1].strip()
            eer_match = re.search(r"EER:\s*([0-9.]+)", eer_line)
            if eer_match:
                eer_value = float(eer_match.group(1))
                scores_dict[dataset][method][0] = (eer_value, tar1, tar01, tar001)
            else:
                print(f"No EER found in line: {eer_line}")

    for dataset, data in scores_dict.items():
        print(f"Dataset: {dataset}")
        for method, seeds in data.items():
            eers = []
            tar1s = []
            tar01s = []
            tar001s = []

            for seed, (eer, tar1, tar01, tar001) in seeds.items():
                eers.append(eer)
                tar1s.append(tar1)
                tar01s.append(tar01)
                tar001s.append(tar001)

            if len(eers) > 1:
                mean_eer = np.mean(eers)
                std_eer = np.std(eers)
                print(
                    f"& {method} & {mean_eer:.4f} ± {std_eer:.4f} & {np.mean(tar1s):.4f} ± {np.std(tar1s):.4f} & {np.mean(tar01s):.4f} ± {np.std(tar01s):.4f} & {np.mean(tar001s):.4f} ± {np.std(tar001s):.4f}"
                )
            else:
                print(
                    f"& {method} & {eers[0]:.4f} & {tar1s[0]:.4f} & {tar01s[0]:.4f} & {tar001s[0]:.4f}"
                )
        print()

    return scores_dict


def get_scores_dict() -> Dict[str, Dict[str, Dict[int, Tuple[NDArray, NDArray]]]]:
    scores_dict = {}
    for dataset, data in final_runs.items():
        scores_dict[dataset] = {}
        print("Loading dataset:", dataset)
        for method, runs in data.items():
            scores_dict[dataset][method] = {}
            print("\tLoading scores for method:", method)
            for seed, run_name in runs.items():
                # genuine_file = f"tmp/{run_name}/{dataset}/genuine_scores.npy"
                # imposter_file = f"tmp/{run_name}/{dataset}/imposter_scores.npy"
                #
                # if not os.path.exists(genuine_file) or not os.path.exists(
                #     imposter_file
                # ):
                #     print(
                #         f"Files {genuine_file} or {imposter_file} do not exist, skipping method {method}."
                #     )
                #     continue
                # genuine_scores = np.load(genuine_file).tolist()
                # imposter_scores = np.load(imposter_file).tolist()
                # if len(genuine_scores) > 2_000_000:
                #     genuine_scores = random.sample(genuine_scores, 2_000_000)
                # if len(imposter_scores) > 2_000_000:
                #     imposter_scores = random.sample(imposter_scores, 2_000_000)
                #
                # eer, far_scores, frr_scores, _ = calculate_eer(
                #     genuine_scores, imposter_scores
                # )

                far_score_file = f"tmp/{run_name}/{dataset}/far_scores.npy"
                if not os.path.exists(far_score_file):
                    print(
                        f"File {far_score_file} does not exist, skipping run {run_name}."
                    )
                    continue

                far_scores = np.load(far_score_file)

                frr_score_file = f"tmp/{run_name}/{dataset}/frr_scores.npy"
                if not os.path.exists(frr_score_file):
                    print(
                        f"File {frr_score_file} does not exist, skipping run {run_name}."
                    )
                    continue

                frr_scores = np.load(frr_score_file)

                if far_scores.shape[0] != 1:
                    far_scores = np.expand_dims(far_scores, axis=0)
                if frr_scores.shape[0] != 1:
                    frr_scores = np.expand_dims(frr_scores, axis=0)

                scores_dict[dataset][method][seed] = (far_scores, frr_scores)
            print(
                f"\tLoaded {len(scores_dict[dataset][method])} scores for method: {method}"
            )

        for method in ["mcp", "rlt", "wld"]:
            scores_dict[dataset][method] = {}
            # genuine_file = f"tmp/{method}/{dataset}/genuine.txt"
            # imposter_file = f"tmp/{method}/{dataset}/imposter.txt"
            # if not os.path.exists(genuine_file) or not os.path.exists(imposter_file):
            #     print(
            #         f"Files {genuine_file} or {imposter_file} do not exist, skipping method {method}."
            #     )
            #     continue
            # genuine_scores = np.loadtxt(genuine_file).tolist()
            # imposter_scores = np.loadtxt(imposter_file).tolist()

            # if len(genuine_scores) > 2_000_000:
            #     genuine_scores = random.sample(genuine_scores, 2_000_000)
            # if len(imposter_scores) > 2_000_000:
            #     imposter_scores = random.sample(imposter_scores, 2_000_000)
            #
            # eer, far_scores, frr_scores, _ = calculate_eer(
            #     genuine_scores, imposter_scores
            # )

            far_score_file = f"tmp/{method}/{dataset}/far_scores.npy"
            if not os.path.exists(far_score_file):
                print(
                    f"File {far_score_file} does not exist, skipping method {method}."
                )
                continue

            far_scores = np.load(far_score_file)

            frr_score_file = f"tmp/{method}/{dataset}/frr_scores.npy"
            if not os.path.exists(frr_score_file):
                print(
                    f"File {frr_score_file} does not exist, skipping method {method}."
                )
                continue
            frr_scores = np.load(frr_score_file)
            print(f"Loaded scores for {dataset} - {method}")
            scores_dict[dataset][method][0] = (far_scores, frr_scores)

    return scores_dict


def plot_det_curves_per_dataset(
    scores_dict: Dict[str, Dict[str, Dict[int, Tuple[NDArray, NDArray]]]],
):
    """
    Plot a gradient for each seed and a mean curve for each method per dataset.
    """

    rcParams.update({"font.size": 14})

    for dataset, methods in scores_dict.items():
        plt.figure(figsize=(10, 6))
        print(f"Plotting DET curves for dataset: {dataset}")

        for method, seeds in methods.items():
            print(f"Processing method: {method}")
            fars = []
            frrs = []
            for seed, (far_scores, frr_scores) in seeds.items():
                fars.append(far_scores)
                frrs.append(frr_scores)
            print(f"\tMethod {method} has {len(fars)} seeds with FAR and FRR scores.")
            if not fars or not frrs:
                continue
            if len(fars) != 1:
                fars = np.concatenate(fars, axis=0)
                frrs = np.concatenate(frrs, axis=0)

                print("After concat:", fars.shape, frrs.shape)
                mean_far = np.mean(fars, axis=0)
                mean_frr = np.mean(frrs, axis=0)

                std_far = np.std(fars, axis=0)
                std_frr = np.std(frrs, axis=0)
                print("After mean:", mean_far.shape, mean_frr.shape)
                plt.plot(
                    mean_far,
                    mean_frr,
                    label=f"{method if method != 'snakegraph2' else 'Proposed'} (mean)",
                    linewidth=2,
                )
                plt.fill_between(
                    mean_far,
                    mean_frr - std_frr,
                    mean_frr + std_frr,
                    alpha=0.2,
                )
            else:
                mean_far = fars[0]
                mean_frr = frrs[0]
                plt.plot(
                    mean_far,
                    mean_frr,
                    label=f"{method}",
                    linewidth=2,
                )

        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("False Acceptance Rate (FAR)")
        plt.ylabel("False Rejection Rate (FRR)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/det_curves_{dataset}.png")
        plt.close()


def plot_ablation():
    kernels_set = [
        [3, 3, 3],
        [5, 5, 5],
        [7, 7, 7],
        [9, 9, 9],
        [9, 7, 5],
        [7, 5, 3],
        [9, 5, 3],
        [9, 7, 3],
    ]
    eers = [
        12.335111777762416,
        12.445330657719536,
        11.835769737849018,
        11.837577589153444,
        11.060479203244075,
        13.06966010388987,
        11.2154698560545,
        10.823535945215658,
    ]
    for kernel_set, eer in zip(kernels_set, eers):
        print(f"{kernel_set} & {eer:.4f} \\\\")

    eers = [
        11.09448459829322,
        10.98850805391412,
        12.506120584898667,
        10.7128721265243,
        12.034655158028226,
        12.13837934213397,
        10.594548767961761,
        11.376957473836946,
        10.737880014890976,
        11.869909604527603,
        13.166805402062067,
        11.734164394156299,
        11.802598977019326,
        11.011694506895001,
        11.0244355147738,
        14.251009112914879,
    ]
    graphers = [2, 4, 6, 8]

    mesh = np.zeros((4, 4))
    for (g1, g2), eer in zip(itertools.product(graphers, graphers), eers):
        mesh[g1 // 2 - 1, g2 // 2 - 1] = eer

    # Plot a heat map for mesh
    # Add the numbers in each cell
    plt.figure(figsize=(8, 6))
    plt.imshow(mesh, cmap="hot", interpolation="nearest")
    plt.colorbar(label="EER (%)")
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            plt.text(
                j,
                i,
                f"{mesh[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mesh[i, j] < 12 else "black",
            )
            print(f"({i}, {j}) = {mesh[i, j]:.4f}")
    plt.xticks(np.arange(len(graphers)), [str(g) for g in graphers])
    plt.yticks(np.arange(len(graphers)), [str(g) for g in graphers])
    plt.xlabel("Number of second sequential GrapherBlocks")
    plt.ylabel("Number of first sequential GrapherBlocks")
    plt.tight_layout()
    plt.savefig("plots/backbone_numberofgraphers.png")


if __name__ == "__main__":
    # log = getLogger()
    # set_seeds(log, 2025)
    # print("Starting to populate scores dictionary...")
    # scores_dict = get_scores_dict()
    # print("Scores dictionary has been populated.")
    # # Use scores_dict for further processing or plotting.
    # plot_det_curves_per_dataset(scores_dict)
    # get_eers()
    plot_ablation()
