import inspect
from logging import Logger
from typing import Any, Dict

from torch.nn import Module


def get_loss(loss: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    if loss == "proposed":
        from losses.proposed import Proposed

        log.warning(f"Proposed\n{inspect.getsource(Proposed)}")
        return Proposed(config, log, **kwargs)
    if loss == "crossentropy":
        from torch.nn import CrossEntropyLoss

        log.warning(f"CrossEntropyLoss\n{inspect.getsource(CrossEntropyLoss)}")
        return CrossEntropyLoss()

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Loss: {loss} not present")
