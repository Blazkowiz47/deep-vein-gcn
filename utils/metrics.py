from logging import Logger, getLogger
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any, List, Optional, Tuple, Union
from functools import partial

import numpy as np


def _inner_worker(
    args: Tuple[int, List[Tuple[int, float]]],
    genuine: np.ndarray,
    imposter: np.ndarray,
) -> Tuple[List[int], List[float], List[float]]:
    pos, thresholds = args
    ids, far, frr = [], [], []
    for id, threshold in thresholds:
        fr = np.where(genuine <= threshold)[0].shape[0]
        fa = np.where(imposter >= threshold)[0].shape[0]
        ids.append(id)
        far.append(fa * 100 / imposter.shape[0])
        frr.append(fr * 100 / genuine.shape[0])

    return ids, far, frr


def chunkify(lst: List[Any], n: int) -> List[Tuple[int, List[Any]]]:
    return [(i, lst[i::n]) for i in range(n)]


def calculate_eer(
    genuine_scores_list: List[Union[float, int]],
    imposter_scores_list: List[Union[float, int]],
    bins: int = 10_001,
    num_workers: int = 38,
    log: Optional[Logger] = None,
) -> Tuple[float, Any, Any, Any]:
    """

    Calculates Equal Error Rate (eer).

    Remember: Genuine scores provided must be greater in value compared to

    imposter.

    Can be used to calculate D-EER, by replacing imposter scores to morph

    scores.

    Parameters

    ----------------------------------------------------------------------

    genuine : List[float] | NDArray

        The list of genuine scores.

    imposter : List[float] | NDArray

        The list of imposter scores.

    Returns

    ----------------------------------------------------------------------

    eer : float

        Equal Error Rate (eer) calculated from given genuine and imposter

        scores.
    far: array
    frr: array
    thresholds: array

    Example

    ----------------------------------------------------------------------

    import common_metrics

    genuine_scores = ... # genuine is a 1D numpy array or List of float

    imposter_scores = ... # imposter is a 1D numpy array or List of float

    eer = common_metrics.eer(

        genuine_scores,

        imposter_scores,

        bins=10_001,

    )

    ----------------------------------------------------------------------

    """
    if log is None:
        log = getLogger(__name__)

    genuine = np.squeeze(np.array(genuine_scores_list))

    imposter = np.squeeze(np.array(imposter_scores_list))

    far = np.ones(bins)

    frr = np.ones(bins)

    mi = min(np.min(imposter), np.min(genuine))

    mx = max(np.max(genuine), np.max(imposter))

    thresholds = np.linspace(mi, mx, bins)

    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    # based on the thresholds using multiprocessing for efficiency

    inner_worker = partial(_inner_worker, genuine=genuine, imposter=imposter)

    with Pool(num_workers) as p:
        chunkified_thresholds = chunkify(
            list(enumerate(thresholds.tolist())), num_workers
        )
        results = list(
            p.map(
                inner_worker,
                chunkified_thresholds,
            )
        )

        log.error("Calculated FAR and FRR")

        for ids, frs, fas in results:
            for id, fr, fa in zip(ids, frs, fas):
                frr[id] = fr
                far[id] = fa

    di = np.argmin(np.abs(far - frr))

    eer = (far[di] + frr[di]) / 2

    return eer, far, frr, thresholds
