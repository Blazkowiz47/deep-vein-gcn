from logging import getLogger
import itertools
from run_name_mappings import final_runs
from scipy import stats
from sklearn.metrics import auc
import os
import re
from typing import Dict, Tuple
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import set_seeds
import seaborn as sns


def norminv(p: float | np.ndarray) -> float | np.ndarray:
    """
    Inverse CDF (quantile) of the standard normal distribution.
    Input: cumulative probability p in (0, 1)
    Output: z such that P(Z <= z) = p for Z ~ N(0, 1)
    """
    p_in = p
    p = np.asarray(p, dtype=float)

    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("p must be in the open interval (0, 1)")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p)

    mask_low = p < plow
    if np.any(mask_low):
        q = np.sqrt(-2.0 * np.log(p[mask_low]))
        x[mask_low] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    mask_mid = (~mask_low) & (p <= phigh)
    if np.any(mask_mid):
        q = p[mask_mid] - 0.5
        r = q * q
        x[mask_mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    mask_high = p > phigh
    if np.any(mask_high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
        x[mask_high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    if np.isscalar(p_in):
        return float(x)
    return x


def plot_det_curve_from_scores(
    far: NDArray,
    frr: NDArray,
    label: str = "Curve",
    min_pct: float = 0.01,
    max_pct: float = 50.0,
    figname: str = "det_curve.png",
) -> None:
    """
    Plot a single DET curve from FAR and FRR scores (in percent 0-100).

    - If arrays are 2D, they are treated as multiple seeds and the function plots
      the mean FRR with ±1 std shading. FAR is assumed identical across seeds; if not,
      FRR arrays are aligned to the first seed's FAR via 1D linear interpolation.
    - Axes are on the DET scale using the inverse normal CDF (norm.ppf), with
      ticks at specific percentages and labels shown as percents.

    Returns the matplotlib Axes used.
    """
    # Prepare axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize input shapes
    far_arr = np.asarray(far)
    frr_arr = np.asarray(frr)

    if far_arr.ndim == 1:
        far_list = [far_arr.reshape(-1)]
        frr_list = [frr_arr.reshape(-1)]
    elif far_arr.ndim == 2:
        far_list = [row.reshape(-1) for row in np.asarray(far_arr)]
        frr_list = [row.reshape(-1) for row in np.asarray(frr_arr)]
    else:
        raise ValueError("far and frr must be 1D or 2D arrays")

    # Canonical FAR grid and alignment
    x_far = far_list[0]
    aligned_frrs: list[np.ndarray] = []
    for seed_far, seed_frr in zip(far_list, frr_list):
        if seed_far.shape != x_far.shape or not np.allclose(
            seed_far, x_far, rtol=1e-6, atol=1e-9
        ):
            aligned = np.interp(x_far, seed_far, seed_frr)
            aligned_frrs.append(aligned)
        else:
            aligned_frrs.append(seed_frr)

    frr_stack = np.vstack(aligned_frrs)
    mean_frr = np.mean(frr_stack, axis=0)
    std_frr = np.std(frr_stack, axis=0) if frr_stack.shape[0] > 1 else None

    # DET transform using inverse normal CDF (norminv)
    eps = 1e-6
    x_pct = np.clip(x_far, min_pct, max_pct)
    y_pct = np.clip(mean_frr, min_pct, max_pct)
    x_det = norminv(np.clip(x_pct / 100.0, eps, 1 - eps))
    y_det = norminv(np.clip(y_pct / 100.0, eps, 1 - eps))

    # Plot mean line
    sns.lineplot(x=x_det, y=y_det, label=label, ax=ax, linewidth=2)

    # Std shading if available
    if std_frr is not None:
        lower_pct = np.clip(mean_frr - std_frr, min_pct, max_pct)
        upper_pct = np.clip(mean_frr + std_frr, min_pct, max_pct)
        lower_det = norminv(np.clip(lower_pct / 100.0, eps, 1 - eps))
        upper_det = norminv(np.clip(upper_pct / 100.0, eps, 1 - eps))
        ax.fill_between(x_det, lower_det, upper_det, alpha=0.2)

    # Axes styling: ticks at fixed percent levels, square aspect
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    ticks_labels = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]
    det_ticks = norminv(np.clip(np.array(ticks) / 100.0, 1e-6, 1 - 1e-6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(det_ticks)
    ax.set_yticks(det_ticks)
    ax.set_xticklabels(ticks_labels)
    ax.set_yticklabels(ticks_labels)

    det_lims = norminv(np.clip(np.array([min_pct, max_pct]) / 100.0, 1e-6, 1 - 1e-6))
    ax.set_xlim(det_lims[0], det_lims[1])
    ax.set_ylim(det_lims[0], det_lims[1])
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_xlabel("False Match Rate (FMR)")
    ax.set_ylabel("False Non-Match Rate (FNMR)")
    ax.legend()
    plt.savefig(figname)
    plt.close()

def plot_det_curves_per_dataset(
    scores_dict: Dict[str, Dict[str, Dict[int, Tuple[NDArray, NDArray]]]],
):
    """
    For each dataset and method, assume FAR is identical across seeds.
    Plot the mean FRR across seeds against FAR and fill ±1 std.
    """

    sns.set_theme(style="whitegrid")
    logger = getLogger()

    for dataset, methods in scores_dict.items():
        plt.figure(figsize=(3.5, 3.5))
        logger.info(f"Plotting DET curves for dataset: {dataset}")

        for method, seeds in methods.items():
            logger.debug(f"Processing method: {method}")
            far_list = []
            frr_list = []
            for seed, (far_scores, frr_scores) in seeds.items():
                far_arr = np.asarray(far_scores).reshape(-1)
                frr_arr = np.asarray(frr_scores).reshape(-1)
                far_list.append(far_arr)
                frr_list.append(frr_arr)

            if not far_list or not frr_list:
                continue

            # Use the first FAR as the canonical x-axis
            x_far = far_list[0]

            # Align all FRR arrays to the same FAR if needed
            aligned_frrs = []
            for idx, (far_arr, frr_arr) in enumerate(zip(far_list, frr_list)):
                if far_arr.shape != x_far.shape or not np.allclose(
                    far_arr, x_far, rtol=1e-6, atol=1e-9
                ):
                    print(f"Aligning seed {idx} for method {method}")
                    try:
                        aligned = np.interp(x_far, far_arr, frr_arr)
                        aligned_frrs.append(aligned)
                    except Exception:
                        logger.warning(
                            f"Could not align seed index {idx} for method {method}; skipping this seed."
                        )
                        continue
                else:
                    aligned_frrs.append(frr_arr)

            if len(aligned_frrs) == 0:
                continue

            frr_stack = np.vstack(aligned_frrs)
            mean_frr = np.mean(frr_stack, axis=0)
            std_frr = np.std(frr_stack, axis=0) if frr_stack.shape[0] > 1 else None

            # DET transform using inverse normal CDF (norminv)
            min_pct, max_pct = 0.01, 50.0
            eps = 1e-6
            x_pct = np.clip(x_far, min_pct, max_pct)
            y_pct = np.clip(mean_frr, min_pct, max_pct)

            x_det = norminv(np.clip(x_pct / 100.0, eps, 1 - eps))
            y_det = norminv(np.clip(y_pct / 100.0, eps, 1 - eps))

            # Plot with seaborn styling in DET space
            label = f"{method if method != 'snakegraph2' else 'Proposed'}"
            sns.lineplot(x=x_det, y=y_det, label=label, linewidth=2)
            if std_frr is not None:
                lower_pct = np.clip(mean_frr - std_frr, min_pct, max_pct)
                upper_pct = np.clip(mean_frr + std_frr, min_pct, max_pct)
                lower_det = norminv(np.clip(lower_pct / 100.0, eps, 1 - eps))
                upper_det = norminv(np.clip(upper_pct / 100.0, eps, 1 - eps))
                plt.fill_between(x_det, lower_det, upper_det, alpha=0.2)

        # DET axes with ticks at percentage levels
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
        ticks_labels = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        det_ticks = norminv(np.clip(np.array(ticks) / 100.0, 1e-6, 1 - 1e-6))
        ax.set_xticks(det_ticks)
        ax.set_yticks(det_ticks)
        ax.set_xticklabels(ticks_labels, fontsize=8)
        ax.set_yticklabels(ticks_labels, fontsize=8)
        det_lims = norminv(
            np.clip(np.array([min_pct, max_pct]) / 100.0, 1e-6, 1 - 1e-6)
        )
        x_det_lims = norminv(np.clip(np.array([min_pct, 48.0]) / 100.0, 1e-6, 1 - 1e-6))
        plt.xlim(x_det_lims[0], x_det_lims[1])
        plt.ylim(det_lims[0], det_lims[1])
        plt.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        plt.xlabel("False Match Rate (FMR) (%)", fontsize=10)
        plt.ylabel("False Non-Match Rate (FNMR) (%)", fontsize=10)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"plots/det_curves_{dataset}.png")
        plt.close()


def get_eers_auc_roc() -> Dict[
    str, Dict[str, Dict[int, Tuple[float, float, float, float]]]
]:
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
                auc_from_curve = auc(fmr / 100, (100 - fnmr) / 100) * 100

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
                        auc_from_curve,
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
            auc_from_curve = auc(fmr / 100, (100 - fnmr) / 100) * 100
            eval_log_file = f"tmp/{method}/eval_{dataset}.log"
            with open(eval_log_file, "r") as f:
                lines = f.readlines()

            eer_line = lines[-1].strip()
            eer_match = re.search(r"EER:\s*([0-9.]+)", eer_line)
            if eer_match:
                eer_value = float(eer_match.group(1))
                scores_dict[dataset][method][0] = (
                    eer_value,
                    tar1,
                    tar01,
                    tar001,
                    auc_from_curve,
                )
            else:
                print(f"No EER found in line: {eer_line}")

    for dataset, data in scores_dict.items():
        print(f"Dataset: {dataset}")
        for method, seeds in data.items():
            eers = []
            tar1s = []
            tar01s = []
            tar001s = []
            auc_from_curves = []

            for seed, (eer, tar1, tar01, tar001, auc_from_curve) in seeds.items():
                eers.append(eer)
                tar1s.append(tar1)
                tar01s.append(tar01)
                tar001s.append(tar001)
                auc_from_curves.append(auc_from_curve)

            if len(eers) > 1:
                mean_eer = np.mean(eers)
                # std_eer = np.std(eers)
                mean_auc = np.mean(auc_from_curves)
                # std_auc = np.std(auc_from_curves)
                ci_eer = stats.t.interval(
                    0.95, len(eers) - 1, loc=mean_eer, scale=stats.sem(eers)
                )
                ci_auc = stats.t.interval(
                    0.95, len(eers) - 1, loc=mean_auc, scale=stats.sem(auc_from_curves)
                )
                # print(ci_eer, ci_auc)
                print(
                    f"& {method} & {mean_auc:.4f}  ({ci_auc[0]:.4f} - {ci_auc[1]:.4f}) & {mean_eer:.4f} ({ci_eer[0]:.4f} - {ci_eer[1]:.4f}) "
                )
            else:
                print(
                    f"& {method} & {auc_from_curve:.4f} & {eers[0]:.4f} & {eers[0]:.4f} "
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
                # plot_det_curve_from_scores(
                #     far_scores, frr_scores, figname="plots/test.png", max_pct=100.0
                # )
                # exit()

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


def plot_roc_curves_per_dataset(
    scores_dict: Dict[str, Dict[str, Dict[int, Tuple[NDArray, NDArray]]]],
):
    """
    For each dataset and method, assume FAR is identical across seeds.
    Plot the mean FRR across seeds against FAR and fill ±1 std.
    """

    sns.set_theme(style="whitegrid")
    logger = getLogger()

    for dataset, methods in scores_dict.items():
        plt.figure(figsize=(6, 6))
        logger.info(f"Plotting ROC curves for dataset: {dataset}")

        for method, seeds in methods.items():
            logger.debug(f"Processing method: {method}")
            far_list = []
            frr_list = []
            for seed, (far_scores, frr_scores) in seeds.items():
                far_arr = np.asarray(far_scores).reshape(-1)
                frr_arr = np.asarray(frr_scores).reshape(-1)
                far_list.append(far_arr)
                frr_list.append(frr_arr)

            if not far_list or not frr_list:
                continue

            # Use the first FAR as the canonical x-axis
            x_far = far_list[0]

            # Align all FRR arrays to the same FAR if needed
            aligned_frrs = []
            for idx, (far_arr, frr_arr) in enumerate(zip(far_list, frr_list)):
                if far_arr.shape != x_far.shape or not np.allclose(
                    far_arr, x_far, rtol=1e-6, atol=1e-9
                ):
                    print(f"Aligning seed {idx} for method {method}")
                    try:
                        aligned = np.interp(x_far, far_arr, frr_arr)
                        aligned_frrs.append(aligned)
                    except Exception:
                        logger.warning(
                            f"Could not align seed index {idx} for method {method}; skipping this seed."
                        )
                        continue
                else:
                    aligned_frrs.append(frr_arr)

            if len(aligned_frrs) == 0:
                continue

            frr_stack = np.vstack(aligned_frrs)
            mean_frr = np.mean(frr_stack, axis=0)
            std_frr = np.std(frr_stack, axis=0) if frr_stack.shape[0] > 1 else None

            label = f"{method if method != 'snakegraph2' else 'Proposed'}"
            sns.lineplot(x=x_far, y=100 - mean_frr, label=label, linewidth=2)
            if std_frr is not None:
                lower_frr = np.clip(100 - (mean_frr + std_frr), 0, 100)
                upper_frr = np.clip(100 - (mean_frr - std_frr), 0, 100)
                plt.fill_between(x_far, lower_frr, upper_frr, alpha=0.2)

        # DET axes with ticks at percentage levels
        # ticks = [0.01, 0.1, 1, 2, 5, 10, 20, 40, 80, 100]
        # ticks_labels = ["0.01", "0.1", "1", "2", "5", "10", "20", "40", "80", "100"]
        ax = plt.gca()
        # ax.set_aspect("equal", adjustable="box")
        ax.set_xscale("log")
        # ax.set_yticks(range(0, 101, 10))
        # ax.set_xticks(ticks)
        # ax.set_yticklabels([str(t) for t in range(0, 101, 10)])
        # ax.set_xticklabels(ticks_labels)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        plt.xlabel("False Match Rate (FMR) (%)", fontsize=10)
        plt.ylabel("True Match Rate (TMR) (%)", fontsize=10)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"plots/roc_curves_{dataset}.png")
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
    plt.imshow(mesh, cmap="viridis", interpolation="nearest")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("EER (%)", fontsize=14)
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            plt.text(
                j,
                i,
                f"{mesh[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mesh[i, j] < 12 else "black",
                fontsize=14,
                fontweight="bold",
            )
            print(f"({i}, {j}) = {mesh[i, j]:.4f}")
    plt.xticks(np.arange(len(graphers)), [str(g) for g in graphers], fontsize=12)
    plt.yticks(np.arange(len(graphers)), [str(g) for g in graphers], fontsize=12)
    plt.xlabel("Number of second sequential GrapherBlocks", fontsize=14)
    plt.ylabel("Number of first sequential GrapherBlocks", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/backbone_numberofgraphers.png")


if __name__ == "__main__":
    log = getLogger()
    set_seeds(log, 2025)
    print("Starting to populate scores dictionary...")
    scores_dict = get_scores_dict()
    # print("Scores dictionary has been populated.")
    # # Use scores_dict for further processing or plotting.
    # plot_det_curves_per_dataset(scores_dict)
    # plot_roc_curves_per_dataset(scores_dict)
    # get_eers_auc_roc()
    plot_ablation()
