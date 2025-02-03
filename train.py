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
    "-m",
    "--model",
    default="deepvein",
    type=str,
    help="Model name.",
)

parser.add_argument(
    "-c",
    "--config",
    default="configs/base.yaml",
    type=str,
    help="Train config file.",
)

parser.add_argument(
    "-d",
    "--dataset",
    default="fv300",
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
    "--logger-level",
    type=str,
    default="INFO",
    help="Logger level",
)

# You can add any additional arguments if you need here.


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    model_name: str = args.model_name or get_run_name(args.model, args.dataset)
    initialise_dirs(model_name)
    logfile = rf"tmp/{model_name}/train.log"
    ckptdir = rf"tmp/{model_name}/checkpoints"
    log = logger.get_logger(model_name, logfile, args.logger_level)

    # Uncomment following line if you use wandb
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

    model = get_model(args.model, config, log).to(device)
    log.info(str(model))
    wrapper = get_dataset(args.dataset, config, log)

    trainds = wrapper.get_split("train")
    validationds = wrapper.get_split("validation")

    if args.continue_model:
        model.load_state_dict(torch.load(args.continue_model))

    criterion = get_loss("proposed", config, log)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=0.05,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs, 1e-4)
    best_validation_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_losses = []
        pbar = tqdm(total=len(trainds), desc=f"Epoch {epoch + 1}")
        for image, label in trainds:
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            preds = model(image)
            step_loss = criterion(preds, label)
            step_loss.backward()
            optimizer.step()

            train_losses.append(step_loss.detach().cpu().item())
            pbar.set_postfix({"loss": np.mean(train_losses)})
            pbar.update(1)

        pbar.close()
        log.info(f"Average train step loss: {np.mean(train_losses)}")
        wandb.log({"train_loss": np.mean(train_losses)})

        torch.save(
            model.state_dict(),
            os.path.join(ckptdir, f"epoch_{epoch}.pt"),
        )
        if not epoch % validate_after_epochs:
            validation_losses = []
            model.eval()
            pbar = tqdm(total=len(validationds), desc="Validation")
            for image, label in validationds:
                image, label = image.to(device), label.to(device)
                preds = model(image)
                step_loss = criterion(preds, label)
                validation_losses.append(step_loss.detach().cpu().item())
                pbar.set_postfix({"loss": np.mean(validation_losses)})
                pbar.update(1)

            pbar.close()
            validation_loss = np.mean(validation_losses)
            log.info(f"Average validation step loss: {validation_loss}")
            wandb.log({"validation_loss": np.mean(validation_losses)})
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(ckptdir, "best_model.pt"),
                )
        scheduler.step()
    # Uncomment following line if you use wandb
    if wandb_run_name:
        wandb.finish()

    if not len(os.listdir(ckptdir)):
        os.system(f"rm -rf tmp/{model_name}")


if __name__ == "__main__":
    main()
