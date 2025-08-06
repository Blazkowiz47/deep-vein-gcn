import itertools
import numpy as np
import os
import yaml
import argparse
from test import parallel_driver
from utils import initialise_dirs, logger, set_seeds, calculate_eer

final_runs = {
    "vera": {
        "arcvein": {
            0: "arcvein_leaveoneout_06_03_25_10_35_2_223",
            1: "arcvein_leaveoneout_06_03_25_10_35_2_225",
            2: "arcvein_leaveoneout_06_03_25_10_35_2_228",
            3: "arcvein_leaveoneout_06_03_25_10_35_2_235",
        },
        "lgfin": {
            0: "lgfin_leaveoneout_06_03_25_13_04_2_236",
            1: "lgfin_leaveoneout_06_03_25_14_46_2_255",
            2: "lgfin_leaveoneout_06_03_25_16_30_2_210",
            3: "lgfin_leaveoneout_06_03_25_18_13_2_247",
        },
        "fv-vit": {
            0: "fvit_leaveoneout_07_03_25_11_11_2_201",
            1: "fvit_leaveoneout_07_03_25_11_11_2_203",
            2: "fvit_leaveoneout_07_03_25_11_11_2_207",
            3: "fvit_leaveoneout_07_03_25_11_11_2_213",
        },
        "veinAttNet": {
            0: "leaveoutds_veinAttNet_vera_seed_0",
            1: "leaveoutds_veinAttNet_vera_seed_1",
            2: "leaveoutds_veinAttNet_vera_seed_2",
            3: "leaveoutds_veinAttNet_vera_seed_3",
        },
        "snakegraph2": {
            0: "dscgrapher_leaveoneout_24_04_25_02_57_2_253",
            1: "dscgrapher_leaveoneout_22_04_25_09_59_2_225",
            2: "dscgrapher_leaveoneout_22_04_25_16_23_2_230",
            3: "dscgrapher_leaveoneout_22_04_25_22_47_2_249",
        },
    },
    "polyu": {
        "arcvein": {
            0: "arcvein_leaveoneout_06_03_25_11_40_2_248",
            1: "arcvein_leaveoneout_06_03_25_11_40_2_249",
            2: "arcvein_leaveoneout_06_03_25_11_40_2_255",
            3: "arcvein_leaveoneout_06_03_25_11_41_2_200",
        },
        "lgfin": {
            0: "lgfin_leaveoneout_07_03_25_13_35_2_248",
            1: "lgfin_leaveoneout_07_03_25_13_35_2_249",
            2: "lgfin_leaveoneout_07_03_25_19_18_2_248",
            3: "lgfin_leaveoneout_07_03_25_19_18_2_249",
        },
        "fv-vit": {
            0: "fvit_leaveoneout_07_03_25_12_54_2_207",
            1: "fvit_leaveoneout_07_03_25_12_54_2_210",
            2: "fvit_leaveoneout_07_03_25_12_54_2_214",
            3: "fvit_leaveoneout_07_03_25_12_54_2_220",
        },
        "veinAttNet": {
            0: "leaveoutds_veinAttNet_polyu_seed_0",
            1: "leaveoutds_veinAttNet_polyu_seed_1",
            2: "leaveoutds_veinAttNet_polyu_seed_2",
            3: "leaveoutds_veinAttNet_polyu_seed_3",
        },
        "snakegraph2": {
            0: "dscgrapher_leaveoneout_23_04_25_20_21_2_231",
            1: "dscgrapher_leaveoneout_21_04_25_07_39_2_217",
            2: "dscgrapher_leaveoneout_21_04_25_14_48_2_203",
            3: "dscgrapher_leaveoneout_21_04_25_21_09_2_255",
        },
    },
    "fvusm": {
        "arcvein": {
            0: "arcvein_leaveoneout_06_03_25_14_25_2_240",
            1: "arcvein_leaveoneout_06_03_25_14_25_2_242",
            2: "arcvein_leaveoneout_06_03_25_14_25_2_246",
            3: "arcvein_leaveoneout_06_03_25_14_25_2_252",
        },
        "lgfin": {
            1: "lgfin_leaveoneout_06_03_25_17_05_2_224",
            2: "lgfin_leaveoneout_06_03_25_17_05_2_227",
            3: "lgfin_leaveoneout_06_03_25_17_05_2_233",
        },
        "fv-vit": {
            0: "fvit_leaveoneout_07_03_25_14_47_2_226",
            1: "fvit_leaveoneout_07_03_25_14_47_2_228",
            2: "fvit_leaveoneout_07_03_25_14_47_2_232",
            3: "fvit_leaveoneout_07_03_25_14_47_2_238",
        },
        "veinAttNet": {
            0: "leaveoutds_veinAttNet_fvusm_seed_0",
            1: "leaveoutds_veinAttNet_fvusm_seed_1",
            2: "leaveoutds_veinAttNet_fvusm_seed_2",
            3: "leaveoutds_veinAttNet_fvusm_seed_3",
        },
        "snakegraph2": {
            0: "dscgrapher_leaveoneout_18_04_25_11_49_2_227",
            1: "dscgrapher_leaveoneout_18_04_25_18_13_2_209",
            2: "dscgrapher_leaveoneout_19_04_25_02_38_2_246",
            3: "dscgrapher_leaveoneout_19_04_25_10_40_2_239",
        },
    },
    "mmcbnu": {
        "arcvein": {
            0: "arcvein_leaveoneout_06_03_25_13_26_2_229",
            1: "arcvein_leaveoneout_06_03_25_13_26_2_230",
            2: "arcvein_leaveoneout_06_03_25_13_26_2_234",
            3: "arcvein_leaveoneout_06_03_25_13_26_2_241",
        },
        "lgfin": {
            0: "lgfin_leaveoneout_07_03_25_05_34_2_201",
            1: "lgfin_leaveoneout_07_03_25_05_34_2_202",
            2: "lgfin_leaveoneout_07_03_25_05_34_2_206",
            3: "lgfin_leaveoneout_07_03_25_05_34_2_212",
        },
        "fv-vit": {
            0: "fvit_leaveoneout_07_03_25_14_18_2_225",
            1: "fvit_leaveoneout_07_03_25_14_18_2_227",
            2: "fvit_leaveoneout_07_03_25_14_18_2_230",
            3: "fvit_leaveoneout_07_03_25_14_18_2_236",
        },
        "veinAttNet": {
            0: "leaveoutds_veinAttNet_mmcbnu_seed_0",
            1: "leaveoutds_veinAttNet_mmcbnu_seed_1",
            2: "leaveoutds_veinAttNet_mmcbnu_seed_2",
            3: "leaveoutds_veinAttNet_mmcbnu_seed_3",
        },
        "snakegraph2": {
            0: "dscgrapher_leaveoneout_23_04_25_14_48_2_203",
            1: "dscgrapher_leaveoneout_19_04_25_21_42_2_259",
            2: "dscgrapher_leaveoneout_20_04_25_08_01_2_244",
            3: "dscgrapher_leaveoneout_20_04_25_13_46_2_236",
        },
    },
     "fv300": {
        "arcvein": {
            0: "arcvein_leaveoneout_06_03_25_15_22_2_228",
            1: "arcvein_leaveoneout_06_03_25_15_22_2_229",
            2: "arcvein_leaveoneout_06_03_25_15_22_2_233",
            3: "arcvein_leaveoneout_06_03_25_15_22_2_240",
        },
        "lgfin": {
            0: "lgfin_leaveoneout_06_03_25_15_53_2_211",
            1: "lgfin_leaveoneout_06_03_25_15_53_2_212",
            2: "lgfin_leaveoneout_06_03_25_15_53_2_217",
            3: "lgfin_leaveoneout_06_03_25_15_53_2_223",
        },
        "fv-vit": {
            0: "fvit_leaveoneout_07_03_25_15_13_2_222",
            1: "fvit_leaveoneout_07_03_25_15_13_2_224",
            2: "fvit_leaveoneout_07_03_25_15_13_2_228",
            3: "fvit_leaveoneout_07_03_25_15_13_2_234",
        },
        "veinAttNet": {
            0: "leaveoutds_veinAttNet_fv300_seed_0",
            1: "leaveoutds_veinAttNet_fv300_seed_1",
            2: "leaveoutds_veinAttNet_fv300_seed_2",
            3: "leaveoutds_veinAttNet_fv300_seed_3",
        },
        "snakegraph2": {
            0: "dscgrapher_leaveoneout_24_04_25_09_59_2_251",
            1: "dscgrapher_leaveoneout_23_04_25_07_20_2_231",
            2: "dscgrapher_leaveoneout_23_04_25_09_27_2_255",
            3: "dscgrapher_leaveoneout_23_04_25_11_31_2_247",
        },
    },
}


stem_ablation = {
    (3, 3, 3): "dscgrapher_leaveoneout_04_04_25_11_21_2_236",
    (5, 5, 5): "dscgrapher_leaveoneout_04_04_25_14_58_2_255",
    (7, 7, 7): "dscgrapher_leaveoneout_04_04_25_19_22_2_231",
    (9, 9, 9): "dscgrapher_leaveoneout_05_04_25_00_07_2_209",
    (9, 7, 5): "dscgrapher_leaveoneout_05_04_25_05_46_2_242",
    (7, 5, 3): "dscgrapher_leaveoneout_05_04_25_10_56_2_246",
    (9, 5, 3): "dscgrapher_leaveoneout_05_04_25_15_15_2_257",
    (9, 7, 3): "dscgrapher_leaveoneout_05_04_25_20_09_2_210",
}


def get_backbone_ablation():
    backbone_ablation = {}
    graphers = [2, 4, 6, 8]
    runnames = [
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
    for i, (g1, g2) in enumerate(itertools.product(graphers, graphers)):
        backbone_ablation[(g1, g2)] = runnames[i]
    return backbone_ablation


# backbone_ablation = get_backbone_ablation()
def get_config_file(model):
    if model == "snakegraph2":
        return "./configs/dscgrapher2.yaml"
    if model == "arcvein":
        return "./configs/arcvein.yaml"
    if model == "lgfin":
        return "./configs/lgim.yaml"
    if model == "fv-vit":
        return "./configs/fvit.yaml"
    if model == "veinAttNet":
        return "./configs/veinattnet.yaml"
    raise NotImplementedError(f"Config file for {model} not implemented.")


def final_runs_eval():
    for dataset, methods in final_runs.items():
        if dataset == "fv300":
            continue

        for method, seeds in methods.items():
            with open(get_config_file(method), "r") as fp:
                config = yaml.safe_load(fp)
            for seed, run_name in seeds.items():
                if method == "veinAttNet":
                    run_name = f"{dataset}/{seed}_{run_name}"
                    continue

                print(f"Ongoing: {method} {seed}: {run_name} on {dataset}")
                config["leaveoutds"] = dataset
                args = argparse.Namespace(
                    config=get_config_file(method),
                    checkpoint=run_name,
                    dataset=dataset,
                    logger_level="ERROR",
                    continue_model=None,
                )
                eer = parallel_driver(args, config)
                print(f"Dataset: {dataset} Method: {method} Seed: {seed} EER: {eer}")
                torch.cuda.empty_cache()

    # for method in ["mcp", "rlt", "wld"]:
    #     for dataset in ["vera", "polyu", "fvusm", "mmcbnu", "fv300"]:
    #         initialise_dirs(method)
    #         logfile = rf"tmp/{method}/eval_{dataset}.log"
    #         log = logger.get_logger(method, logfile, "ERROR")
    #
    #         genuine_scores = (
    #             np.loadtxt(f"tmp/{method}/{dataset}/genuine.txt").reshape(-1).tolist()
    #         )
    #         imposter_scores = (
    #             np.loadtxt(f"tmp/{method}/{dataset}/imposter.txt").reshape(-1).tolist()
    #         )
    #         eer, far, frr, _ = calculate_eer(genuine_scores, imposter_scores)
    #
    #         log.error(f"{method} Dataset: {dataset} EER: {eer}")
    #         np.save(f"tmp/{method}/{dataset}/far_scores.npy", far)
    #         np.save(f"tmp/{method}/{dataset}/frr_scores.npy", frr)
    return


if __name__ == "__main__":
    import torch

    # Clear cuda cache
    torch.cuda.empty_cache()
    final_runs_eval()
