from logging import getLogger
import torch
from torch.nn import NLLLoss, Linear, Module


class NllLoss(Module):
    def __init__(self, config, log, **kwargs):
        super(NllLoss, self).__init__()
        self.log = log
        self.name = "NllLoss"
        self.num_classes = config["num_classes"]
        self.fc = Linear(config["embedding_size"], self.num_classes)
        self.nll_loss = NLLLoss()

    def forward(self, embds, labels, **kwargs):
        logits = self.fc(embds)
        labels = torch.argmax(labels, dim=1)
        return (
            self.nll_loss(logits, labels),
            torch.tensor(0.0).to(embds.device),
            logits,
        )


if __name__ == "__main__":
    config = {"num_classes": 301, "embedding_size": 301}
    log = getLogger("NllLoss")
    log.setLevel("DEBUG")
    loss = NllLoss(config, log)
    embds = torch.randn(10, 301)
    labels = torch.zeros(10, 301)
    for i in range(10):
        labels[i, i] = 1

    x1, x2 = loss(labels, labels)
    print(x1.shape, x2.shape)
    print(x1, x2)

    logits = torch.tensor(
        [[10.0, 0.0, -10.0, 0], [10.0, 0.0, -10.0, 0]]
    )  # High confidence for class 0
    targets = torch.tensor([0, 3])  # Ground truth is class 0

    # Compute loss
    loss = NllLoss()(logits, targets)
    print(loss.item())  # Output: 0.0 (or very close)
