from logging import Logger
from typing import Any, Dict
from torch.nn import Module


def get_model(model: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    if model == "dscgrapher":
        from models.dscgrapher import Dscgrapher

        return Dscgrapher(config, log, **kwargs)

    if model == "deepvein":
        from models.deepvein import Deepvein

        return Deepvein(config, log, **kwargs)

    if model == "fvit":
        from models.fvit import Fvit

        return Fvit(config, log, **kwargs)

    if model == "veinAttNet":
        from models.veinAttNet import VeinAttNet

        return VeinAttNet(config, log, **kwargs)

    if model == "lgfin":
        from models.lgfin import Lgfin

        return Lgfin(config, log, **kwargs)

    if model == "arcvein":
        from models.arcvein import Arcvein

        return Arcvein(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Model: {model} not present")
