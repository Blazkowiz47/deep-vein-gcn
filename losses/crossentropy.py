from logging import getLogger
import torch
from torch.nn import CrossEntropyLoss, Module


class CrossEntropy(Module):
    def __init__(self, config, log, **kwargs):
        super(CrossEntropy, self).__init__()
        self.log = log
        self.name = "CrossEntropy"
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, preds, labels, **kwargs):
        labels = torch.argmax(labels, dim=1)
        return self.cross_entropy(preds, labels), torch.tensor(0.0).to(embds.device)


if __name__ == "__main__":
    config = {}
    log = getLogger("CrossEntropy")
    log.setLevel("DEBUG")
    loss = CrossEntropy(config, log)
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
    loss = CrossEntropyLoss()(logits, targets)
    print(loss.item())  # Output: 0.0 (or very close)
