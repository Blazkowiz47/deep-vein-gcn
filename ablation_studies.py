import argparse
import itertools
import json
import logging
from logging import INFO, getLogger
from typing import Any, Dict

import yaml

from test import parallel_driver
from train import main

logging.basicConfig(level=INFO)
log = getLogger()


def ablate_stem(config: Dict[str, Any]) -> None:
    print(config["stem"])

    eers = [  # noqa: F841
        12.335111777762416,
        12.445330657719536,
        11.835769737849018,
        11.837577589153444,
        11.060479203244075,
        13.06966010388987,
        11.2154698560545,
        10.823535945215658,
    ]
    for kernel_set in [
        [3, 3, 3],
        [5, 5, 5],
        [7, 7, 7],
        [9, 9, 9],
        [9, 7, 5],
        [7, 5, 3],
        [9, 5, 3],
        [9, 7, 3],
    ]:
        config["stem"]["kernels"] = kernel_set
        # model = get_model(config["model"], config, log).cuda()
        # log.info(f"{model(torch.rand(3, 3, 224, 224).cuda()).shape}")
        args = argparse.Namespace(
            config="./configs/dscgrapher2.yaml",
            seed=0,
            leave="fvusm",
            wandb=True,
            dataset="leaveoneout",
            model_name=None,
            logger_level="INFO",
            continue_model=None,
        )
        main(args, config)


def ablate_backbone(config: Dict[str, Any]) -> None:
    print(config["stem"])
    graphers = [2, 4, 6, 8]
    continue_till = True
    for g1, g2 in itertools.product(graphers, graphers):
        config["backbone"]["block0"]["graphers"] = g1
        config["backbone"]["block1"]["graphers"] = g2
        if g1 == 4 and g2 == 2:
            continue_till = False
        if continue_till:
            continue
        # model = get_model(config["model"], config, log).cuda()
        # log.info(f"{model(torch.rand(3, 3, 224, 224).cuda()).shape}")
        args = argparse.Namespace(
            config="./configs/dscgrapher2.yaml",
            seed=0,
            leave="fvusm",
            wandb=True,
            dataset="leaveoneout",
            model_name=None,
            logger_level="INFO",
            continue_model=None,
        )
        main(args, config)


def final_runs(config):
    run_maps = {}
    ckpts = {
        "fvusm": {
            0: "dscgrapher_leaveoneout_18_04_25_11_49_2_227",
            1: "dscgrapher_leaveoneout_18_04_25_18_13_2_209",
            2: "dscgrapher_leaveoneout_19_04_25_02_38_2_246",
            3: "dscgrapher_leaveoneout_19_04_25_10_40_2_239",
        },
        "mmcbnu": {
            0: "dscgrapher_leaveoneout_23_04_25_14_48_2_203",
            1: "dscgrapher_leaveoneout_19_04_25_21_42_2_259",
            2: "dscgrapher_leaveoneout_20_04_25_08_01_2_244",
            3: "dscgrapher_leaveoneout_20_04_25_13_46_2_236",
        },
        "polyu": {
            0: "dscgrapher_leaveoneout_23_04_25_20_21_2_231",
            1: "dscgrapher_leaveoneout_21_04_25_07_39_2_217",
            2: "dscgrapher_leaveoneout_21_04_25_14_48_2_203",
            3: "dscgrapher_leaveoneout_21_04_25_21_09_2_255",
        },
        "vera": {
            0: "dscgrapher_leaveoneout_24_04_25_02_57_2_253",
            1: "dscgrapher_leaveoneout_22_04_25_09_59_2_225",
            2: "dscgrapher_leaveoneout_22_04_25_16_23_2_230",
            3: "dscgrapher_leaveoneout_22_04_25_22_47_2_249",
        },
        "fv300": {
            0: "dscgrapher_leaveoneout_24_04_25_09_59_2_251",
            1: "dscgrapher_leaveoneout_23_04_25_07_20_2_231",
            2: "dscgrapher_leaveoneout_23_04_25_09_27_2_255",
            3: "dscgrapher_leaveoneout_23_04_25_11_31_2_247",
        },
    }
    for dataset in ckpts:
        for seed, run_name in ckpts[dataset].items():
            # args = argparse.Namespace(
            #     config="./configs/dscgrapher2.yaml",
            #     seed=seed,
            #     leave=dataset,
            #     wandb=True,
            #     dataset="leaveoneout",
            #     model_name=None,
            #     logger_level="INFO",
            #     continue_model=None,
            # )
            # run_name = main(args, config)
            # ckpts[dataset][seed] = run_name

            args = argparse.Namespace(
                config="./configs/dscgrapher2.yaml",
                checkpoint=run_name,
                dataset=dataset,
                logger_level="ERROR",
                continue_model=None,
            )
            eer = parallel_driver(args, config)
            print(f"Dataset: {dataset} seed: {seed} eer: {eer}")

            if dataset not in run_maps:
                run_maps[dataset] = {}
            run_maps[dataset][seed] = eer
            break
        break
    return
    with open("final_results.json", "w+") as fp:
        json.dump(run_maps, fp)


def fetch_results(config):
    run_maps = {}
    ckpts = {
        "fvusm": {
            0: "dscgrapher_leaveoneout_18_04_25_11_49_2_227",
            1: "dscgrapher_leaveoneout_18_04_25_18_13_2_209",
            2: "dscgrapher_leaveoneout_19_04_25_02_38_2_246",
            3: "dscgrapher_leaveoneout_19_04_25_10_40_2_239",
        },
        "mmcbnu": {
            0: "dscgrapher_leaveoneout_23_04_25_14_48_2_203",
            1: "dscgrapher_leaveoneout_19_04_25_21_42_2_259",
            2: "dscgrapher_leaveoneout_20_04_25_08_01_2_244",
            3: "dscgrapher_leaveoneout_20_04_25_13_46_2_236",
        },
        "polyu": {
            0: "dscgrapher_leaveoneout_23_04_25_20_21_2_231",
            1: "dscgrapher_leaveoneout_21_04_25_07_39_2_217",
            2: "dscgrapher_leaveoneout_21_04_25_14_48_2_203",
            3: "dscgrapher_leaveoneout_21_04_25_21_09_2_255",
        },
        "vera": {
            0: "dscgrapher_leaveoneout_24_04_25_02_57_2_253",
            1: "dscgrapher_leaveoneout_22_04_25_09_59_2_225",
            2: "dscgrapher_leaveoneout_22_04_25_16_23_2_230",
            3: "dscgrapher_leaveoneout_22_04_25_22_47_2_249",
        },
        "fv300": {
            0: "dscgrapher_leaveoneout_24_04_25_09_59_2_251",
            1: "dscgrapher_leaveoneout_23_04_25_07_20_2_231",
            2: "dscgrapher_leaveoneout_23_04_25_09_27_2_255",
            3: "dscgrapher_leaveoneout_23_04_25_11_31_2_247",
        },
    }
    for dataset in ckpts:
        for seed, run_name in ckpts[dataset].items():
            # model = get_model(config["model"], config, log).cuda()
            # log.info(f"{model(torch.rand(3, 3, 224, 224).cuda()).shape}")
            # args = argparse.Namespace(
            #     config="./configs/dscgrapher2.yaml",
            #     seed=seed,
            #     leave=dataset,
            #     wandb=True,
            #     dataset="leaveoneout",
            #     model_name=None,
            #     logger_level="INFO",
            #     continue_model=None,
            # )
            # run_name = main(args, config)

            args = argparse.Namespace(
                config="./configs/dscgrapher2.yaml",
                checkpoint=run_name,
                dataset=dataset,
                logger_level="ERROR",
                continue_model=None,
            )
            eer = parallel_driver(args, config)

            if dataset not in run_maps:
                run_maps[dataset] = {}
            run_maps[dataset][seed] = eer

    with open("final_results.json", "w+") as fp:
        json.dump(run_maps, fp)


def ablate_loss(config):
    runs = {}
    print(config["loss"])
    for loss in ["mse", "crossentropy", "focalloss"]:
        config["loss"] = loss
        args = argparse.Namespace(
            config="./configs/dscgrapher2.yaml",
            seed=0,
            leave="fvusm",
            wandb=True,
            dataset="leaveoneout",
            model_name=None,
            logger_level="INFO",
            continue_model=None,
        )
        run_name = main(args, config)
        runs[loss] = run_name

    with open("ablation_results_loss.json", "w+") as fp:
        json.dump(runs, fp)


def driver() -> None:
    with open("./configs/dscgrapher2.yaml", "r") as fp:
        config = yaml.safe_load(fp)
    # ablate_stem(config)
    # ablate_backbone(config)
    ablate_loss(config)
    # fetch_results(config)
    # final_runs(config)


def test():
    with open("./configs/dscgrapher2.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # Stem ablation:
    ckpts = [
        "dscgrapher_leaveoneout_04_04_25_11_21_2_236",
        "dscgrapher_leaveoneout_04_04_25_14_58_2_255",
        "dscgrapher_leaveoneout_04_04_25_19_22_2_231",
        "dscgrapher_leaveoneout_05_04_25_00_07_2_209",
        "dscgrapher_leaveoneout_05_04_25_05_46_2_242",
        "dscgrapher_leaveoneout_05_04_25_10_56_2_246",
        "dscgrapher_leaveoneout_05_04_25_15_15_2_257",
        "dscgrapher_leaveoneout_05_04_25_20_09_2_210",
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

    # Backbone ablation:
    ckpts = [
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
    eers = [  # noqa: F841
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

    for kernel_set, ckpt in zip(
        [
            [3, 3, 3],
            [5, 5, 5],
            [7, 7, 7],
            [9, 9, 9],
            [9, 7, 5],
            [7, 5, 3],
            [9, 5, 3],
            [9, 7, 3],
        ],
        ckpts,
    ):
        config["stem"]["kernels"] = kernel_set

    graphers = [2, 4, 6, 8]
    for (g1, g2), ckpt in zip(itertools.product(graphers, graphers), ckpts):
        config["backbone"]["block0"]["graphers"] = g1
        config["backbone"]["block1"]["graphers"] = g2
        # model = get_model(config["model"], config, log).cuda()
        # log.info(f"{model(torch.rand(3, 3, 224, 224).cuda()).shape}")
        args = argparse.Namespace(
            config="./configs/dscgrapher2.yaml",
            checkpoint=ckpt,
            dataset="fvusm",
            logger_level="ERROR",
            continue_model=None,
        )
        parallel_driver(args, config)


if __name__ == "__main__":
    driver()
    # test()
