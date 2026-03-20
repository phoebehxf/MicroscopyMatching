# Modified from Trackastra (https://github.com/weigertlab/trackastra)

import logging

import dask.array as da
import numpy as np
import torch
from typing import Optional, Union

logger = logging.getLogger(__name__)


def blockwise_sum(
    A: torch.Tensor, timepoints: torch.Tensor, dim: int = 0, reduce: str = "sum"
):
    if not A.shape[dim] == len(timepoints):
        raise ValueError(
            f"Dimension {dim} of A ({A.shape[dim]}) must match length of timepoints"
            f" ({len(timepoints)})"
        )

    A = A.transpose(dim, 0)

    if len(timepoints) == 0:
        logger.warning("Empty timepoints in block_sum. Returning zero tensor.")
        return A
    # -1 is the filling value for padded/invalid timepoints
    min_t = timepoints[timepoints >= 0]
    if len(min_t) == 0:
        logger.warning("All timepoints are -1 in block_sum. Returning zero tensor.")
        return A

    min_t = min_t.min()
    # after that, valid timepoints start with 1 (padding timepoints will be mapped to 0)
    ts = torch.clamp(timepoints - min_t + 1, min=0)
    index = ts.unsqueeze(1).expand(-1, len(ts))
    blocks = ts.max().long() + 1
    out = torch.zeros((blocks, A.shape[1]), device=A.device, dtype=A.dtype)
    out = torch.scatter_reduce(out, 0, index, A, reduce=reduce)
    B = out[ts]
    B = B.transpose(0, dim)

    return B


def blockwise_causal_norm(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    mode: str = "quiet_softmax",
    mask_invalid: torch.BoolTensor = None,
    eps: float = 1e-6,
):
    """Normalization over the causal dimension of A.

    For each block of constant timepoints, normalize the corresponding block of A
    such that the sum over the causal dimension is 1.

    Args:
        A (torch.Tensor): input tensor
        timepoints (torch.Tensor): timepoints for each element in the causal dimension
        mode: normalization mode.
            `linear`: Simple linear normalization.
            `softmax`: Apply exp to A before normalization.
            `quiet_softmax`: Apply exp to A before normalization, and add 1 to the denominator of each row/column.
        mask_invalid: Values that should not influence the normalization.
        eps (float, optional): epsilon for numerical stability.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    A = A.clone()

    if mode in ("softmax", "quiet_softmax"):
        # Subtract max for numerical stability
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning


        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = -torch.inf
        # TODO set to min, then to 0 after exp

        # Blockwise max
        with torch.no_grad():
            ma0 = blockwise_sum(A, timepoints, dim=0, reduce="amax")
            ma1 = blockwise_sum(A, timepoints, dim=1, reduce="amax")

        u0 = torch.exp(A - ma0)
        u1 = torch.exp(A - ma1)
    elif mode == "linear":
        A = torch.sigmoid(A)
        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = 0

        u0, u1 = A, A
        ma0 = ma1 = 0
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    u0_sum = blockwise_sum(u0, timepoints, dim=0) + eps
    u1_sum = blockwise_sum(u1, timepoints, dim=1) + eps

    if mode == "quiet_softmax":
        # Add 1 to the denominator of the softmax. With this, the softmax outputs can be all 0, if the logits are all negative.
        # If the logits are positive, the softmax outputs will sum to 1.
        # Trick: With maximum subtraction, this is equivalent to adding 1 to the denominator
        u0_sum += torch.exp(-ma0)
        u1_sum += torch.exp(-ma1)

    mask0 = timepoints.unsqueeze(0) > timepoints.unsqueeze(1)
    # mask1 = timepoints.unsqueeze(0) < timepoints.unsqueeze(1)
    # Entries with t1 == t2 are always masked out in final loss
    mask1 = ~mask0

    # blockwise diagonal will be normalized along dim=0
    res = mask0 * u0 / u0_sum + mask1 * u1 / u1_sum
    res = torch.clamp(res, 0, 1)
    return res




def normalize(x: Union[np.ndarray, da.Array], subsample: Optional[int] = 4):
    """Percentile normalize the image.

    If subsample is not None, calculate the percentile values over a subsampled image (last two axis)
    which is way faster for large images.
    """
    x = x.astype(np.float32)
    if subsample is not None and all(s > 64 * subsample for s in x.shape[-2:]):
        y = x[..., ::subsample, ::subsample]
    else:
        y = x

    mi, ma = np.percentile(y, (1, 99.8)).astype(np.float32)
    x -= mi
    x /= ma - mi + 1e-8
    return x

def normalize_01(x: Union[np.ndarray, da.Array], subsample: Optional[int] = 4):
    """Percentile normalize the image.

    If subsample is not None, calculate the percentile values over a subsampled image (last two axis)
    which is way faster for large images.
    """
    x = x.astype(np.float32)
    if subsample is not None and all(s > 64 * subsample for s in x.shape[-2:]):
        y = x[..., ::subsample, ::subsample]
    else:
        y = x

    # mi, ma = np.percentile(y, (1, 99.8)).astype(np.float32)
    mi = x.min()
    ma = x.max()
    x -= mi
    x /= ma - mi + 1e-8
    return x

