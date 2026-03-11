import inspect
from logging import Logger
from typing import Any, Dict

from torch.nn import Module


def get_loss(loss: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    if loss == "proposed":
        from losses.proposed import Proposed

        log.warning(f"Proposed\n{inspect.getsource(Proposed)}")
        return Proposed(config, log, **kwargs)

    if loss == "arcveinloss":
        from losses.arcveinloss import ArcCosineLoss

        log.warning(f"ArcCosineLoss\n{inspect.getsource(ArcCosineLoss)}")
        return ArcCosineLoss(config, log, **kwargs)

    if loss == "crossentropy":
        from losses.crossentropy import CrossEntropy

        log.warning(f"CrossEntropyLoss\n{inspect.getsource(CrossEntropy)}")
        return CrossEntropy(config, log, **kwargs)

    if loss == "mse":
        from losses.mse import Mse

        log.warning(f"Mse\n{inspect.getsource(Mse)}")
        return Mse(config, log, **kwargs)

    if loss == "nll":
        from losses.nllloss import NllLoss

        log.warning(f"NllLoss\n{inspect.getsource(NllLoss)}")
        return NllLoss(config, log, **kwargs)

    if loss == "focalloss":
        from losses.focalloss import FocalLoss

        log.warning(f"FocalLoss\n{inspect.getsource(FocalLoss)}")
        return FocalLoss(config, log, **kwargs)
    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Loss: {loss} not present")
