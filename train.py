"""
Main training file.
calls the train pipeline with configs.
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# Incase you use wandb uncomment following line
import wandb
from cdatasets import get_dataset
from losses import get_loss
from models import get_model
from utils import get_run_name, initialise_dirs, logger, set_seeds

parser = argparse.ArgumentParser(
    description="Training Config",
    add_help=True,
)

parser.add_argument(
    "-c",
    "--config",
    # default="./configs/dscgrapher.yaml",
    # default="./configs/deepvein.yaml",
    default="./configs/arcvein.yaml",
    type=str,
    help="Train config file.",
)

parser.add_argument(
    "--seed",
    type=str,
    required=False,
    help="Train config file.",
)

parser.add_argument(
    "--logger-level",
    type=str,
    default="INFO",
    help="Logger level",
)

parser.add_argument(
    "-d",
    "--dataset",
    default="leaveoneout",
    type=str,
    help="""
    Give a single dataset name or multiple datasets to chain together.
    eg: -d fv300
    """,
)

parser.add_argument(
    "-ckpt",
    "--continue-model",
    type=str,
    default=None,
    help="Load initial weights from partially/pretrained model.",
)

parser.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="Model name to save the model.",
)

parser.add_argument(
    "--wandb",
    action="store_true",
    help="Use wandb for logging.",
)

parser.add_argument(
    "--leave",
    type=str,
    required=False,
    help="Name of the dataset to leave out.",
)
# You can add any additional arguments if you need here.


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    if args.seed:
        config["stat_seed"] = args.seed
    if args.leave:
        config["leaveoutds"] = args.leave

    model = config["model"]
    model_name: str = args.model_name or get_run_name(model, args.dataset)
    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/train.log"
    ckptdir = rf"tmp/{model_name}/checkpoints"
    log = logger.get_logger(model_name, logfile, args.logger_level)
    log.info(f"Training started for: {model_name}.")
    log.info("Config:")

    # Uncomment following line if you use wandb
    wandb_run_name = None
    if args.wandb:
        wandb_run_name = model_name

    if wandb_run_name:
        wandb.init(
            # set the wandb project where this run will be logged
            project="fingervein",
            name=wandb_run_name,
            config={
                **config,
                "dataset": args.dataset,
            },
        )

    set_seeds(log, config["seed"])
    epochs = config["epochs"]
    validate_after_epochs = config["validate_after_epochs"]

    device = config["device"]  # You can change this to cpu.

    wrapper = get_dataset(args.dataset, config, log)

    trainds = wrapper.get_split("train")
    validationds = wrapper.get_split("validation",batch_size=8)
    if config["num_classes"] != wrapper.num_classes:
        config["num_classes"] = wrapper.num_classes

    log.info(str(config))
    model = get_model(model, config, log).to(device)
    log.info("Model:")
    log.info(str(model))

    if args.continue_model:
        model.load_state_dict(torch.load(args.continue_model))

    criterion = get_loss(config["loss"], config, log).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    params.extend([p for p in criterion.parameters() if p.requires_grad])

    optimizer = AdamW(
        params,
        lr=config["lr"],
        weight_decay=0.05,
    )

    scheduler = CosineAnnealingLR(optimizer, epochs, 1e-3)
    best_validation_loss = np.inf

    for epoch in range(epochs):
        model.train()
        criterion.train()
        train_losses = []
        step1_train_losses = []
        step2_train_losses = []
        wandblog = {}
        pbar = tqdm(trainds, desc=f"Epoch {epoch + 1}")
        i = 0
        for image, label in pbar:
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            preds = model(image)
            loss1, loss2 = criterion(
                preds, label, freeze_centroids=epoch > config["freeze_centroids"]
            )
            step_loss = loss1 + loss2
            step_loss.backward()
            optimizer.step()

            train_losses.append(step_loss.detach().cpu().item())
            step1_train_losses.append(loss1.detach().cpu().item())
            step2_train_losses.append(loss2.detach().cpu().item())
            pbar.set_postfix({"loss": np.mean(train_losses)})
            pbar.update(1)

            i += 1
            if i == 10:
                continue

        pbar.close()
        log.info(f"Average train step loss: {np.mean(train_losses)}")
        wandblog = {
            "train_loss": np.mean(train_losses),
            "Step1_loss": np.mean(step1_train_losses),
            "Step2_loss": np.mean(step2_train_losses),
        }

        torch.save(
            model.state_dict(),
            os.path.join(ckptdir, f"epoch_{epoch}.pt"),
        )
        if not epoch % validate_after_epochs:
            validation_losses = []
            step1_losses = []
            step2_losses = []
            model.eval()
            criterion.eval()
            pbar = tqdm(validationds, desc="Validation")
            for image, label in pbar:
                image, label = image.to(device), label.to(device)
                preds = model(image)
                loss1, loss2 = criterion(preds, label)
                step_loss = loss1 + loss2
                validation_losses.append(step_loss.detach().cpu().item())
                step1_losses.append(loss1.detach().cpu().item())
                step2_losses.append(loss2.detach().cpu().item())
                pbar.set_postfix({"loss": np.mean(validation_losses)})

            pbar.close()
            validation_loss = np.mean(validation_losses)
            log.info(f"Average validation step loss: {validation_loss}")
            wandblog = {
                **wandblog,
                "validation_loss": np.mean(validation_losses),
                "Step1_loss": np.mean(step1_losses),
                "Step2_loss": np.mean(step2_losses),
            }
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(ckptdir, "best_model.pt"),
                )
        scheduler.step()

        if wandb_run_name:
            wandb.log(wandblog)
    # Uncomment following line if you use wandb
    if wandb_run_name:
        wandb.finish()

    log.info(f"Training completed for: {model_name}.")
    if not len(os.listdir(ckptdir)):
        os.system(f"rm -rf tmp/{model_name}")


if __name__ == "__main__":
    main()
