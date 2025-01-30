import yaml
from models import get_model
import torch
from utils import get_logger

with open("./configs/base.yaml", "r") as fp:
    config = yaml.safe_load(fp)

logger = get_logger("test","", "DEBUG")


model = get_model("deepvein", config, logger)
x = torch.randn(6, 3, 224, 224).float()
y = model(x)
print("Output of model:", y.shape)
