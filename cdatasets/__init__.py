from logging import Logger
from typing import Any, Dict
from utils import Wrapper


def get_dataset(dataset: str, config: Dict[str, Any], log: Logger, **kwargs) -> Wrapper:
    if dataset == "leaveoneout":
        from cdatasets.leaveoneout import LeaveoneoutWrapper

        return LeaveoneoutWrapper(config, log, **kwargs)

    if dataset == "balanced_leaveoneout":
        from cdatasets.leaveoneout import BalancedLeaveoneoutWrapper

        return BalancedLeaveoneoutWrapper(config, log, **kwargs)

    if dataset == "leaveoneout2":
        from cdatasets.leaveoneout2 import Leaveoneout2Wrapper

        return Leaveoneout2Wrapper(config, log, **kwargs)

    if dataset == "intra":
        from cdatasets.intra import IntraWrapper

        return IntraWrapper(config, log, **kwargs)

    if dataset == "balanced_intra":
        from cdatasets.intra import BalancedIntraWrapper

        return BalancedIntraWrapper(config, log, **kwargs)

    if dataset == "mmcbnu":
        from cdatasets.mmcbnu import MmcbnuWrapper

        return MmcbnuWrapper(config, log, **kwargs)

    if dataset == "fv300":
        from cdatasets.fv300 import Fv300Wrapper

        return Fv300Wrapper(config, log, **kwargs)

    if dataset == "polyu":
        from cdatasets.polyu import PolyuWrapper

        return PolyuWrapper(config, log, **kwargs)

    if dataset == "vera":
        from cdatasets.vera import VeraWrapper

        return VeraWrapper(config, log, **kwargs)

    if dataset == "fvusm":
        from cdatasets.fvusm import FvusmWrapper

        return FvusmWrapper(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new dataset.
    raise NotImplementedError(f"Dataset: {dataset} not present")
