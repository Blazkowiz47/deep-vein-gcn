from logging import getLogger
import torch
from torch.nn import Linear, MSELoss, Module


class Mse(Module):
    def __init__(self, config, log, **kwargs):
        super(Mse, self).__init__()
        self.log = log
        self.name = "Mse"
        self.num_classes = config["num_classes"]
        self.fc = Linear(config["embedding_size"], self.num_classes)
        self.mse = MSELoss()

    def forward(self, embds, labels, **kwargs):
        preds = self.fc(embds)
        return self.mse(preds, labels), torch.tensor(0.0).to(preds.device), preds


if __name__ == "__main__":
    config = {}
    log = getLogger("Mse")
    log.setLevel("DEBUG")
    loss = Mse(config, log)
    embds = torch.randn(10, 301)
    labels = torch.zeros(10, 301)
    for i in range(10):
        labels[i, i] = 1

    x1, x2 = loss(embds, labels)
    print(x1.shape, x2.shape)
    print(x1, x2)
    print("--------------------------------")
