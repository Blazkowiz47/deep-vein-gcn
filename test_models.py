import logging
from logging import getLogger, DEBUG, INFO
import yaml

from models import get_model
from cdatasets import get_dataset
from utils import compute_eer_mp
import torch
import math

logging.basicConfig(level=INFO)
log = getLogger()


def cross_entropy(x, y):
    loss = 0
    for preds, lbl in zip(x, y):
        deno = sum([math.exp(p) for p in preds])
        num = math.exp(preds[lbl])
        loss += -math.log(num / deno)
    return loss / len(x)


def main():
    config_file = "./configs/dscgrapher2.yaml"

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = get_model(config["model"], config, log).cuda()
    model.load_state_dict(
        torch.load(
            "./tmp/dscgrapher_leaveoneout_29_03_25_12_22_2_224/checkpoints/best_model.pt",
            weights_only=True,
        )
    )
    # log.info(str(model))
    # print(model(torch.rand(3, 3, 224, 224).cuda()).shape)

    wrapper = get_dataset("fvusm", config, log, partition_split=0)
    eer, genscores, impscores = compute_eer_mp(model, wrapper, workers=4, device="cuda")
    print(f"EER: {eer}")


if __name__ == "__main__":
    main()
