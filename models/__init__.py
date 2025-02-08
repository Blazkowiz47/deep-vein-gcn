import inspect
from logging import Logger
from typing import Any, Dict

from torch.nn import Module


def get_model(model: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    if model == "deepvein":
        from models.deepvein import DeepVein

        log.warning(f"DeepVein\n{inspect.getsource(DeepVein)}")
        return DeepVein(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Model: {model} not present")
