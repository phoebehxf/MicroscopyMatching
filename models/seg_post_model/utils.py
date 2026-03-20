"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging
import io
from tqdm import tqdm, trange
import cv2
from scipy.ndimage import find_objects
import numpy as np
import fastremap
import fill_voids
from models.seg_post_model import metrics


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)



# def masks_to_outlines(masks):
#     """Get outlines of masks as a 0-1 array.

#     Args:
#         masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.

#     Returns:
#         outlines (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are outlines.
#     """
#     if masks.ndim > 3 or masks.ndim < 2:
#         raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
#                          masks.ndim)
#     outlines = np.zeros(masks.shape, bool)

#     if masks.ndim == 3:
#         for i in range(masks.shape[0]):
#             outlines[i] = masks_to_outlines(masks[i])
#         return outlines
#     else:
#         slices = find_objects(masks.astype(int))
#         for i, si in enumerate(slices):
#             if si is not None:
#                 sr, sc = si
#                 mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
#                 contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                             cv2.CHAIN_APPROX_NONE)
#                 pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
#                 vr, vc = pvr + sr.start, pvc + sc.start
#                 outlines[vr, vc] = 1
#         return outlines


def stitch3D(masks, stitch_threshold=0.25):
    """
    Stitch 2D masks into a 3D volume using a stitch_threshold on IOU.

    Args:
        masks (list or ndarray): List of 2D masks.
        stitch_threshold (float, optional): Threshold value for stitching. Defaults to 0.25.

    Returns:
        list: List of stitched 3D masks.
    """
    mmax = masks[0].max()
    empty = 0
    for i in trange(len(masks) - 1):
        iou = metrics._intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, masks.dtype)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, masks.dtype)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks


# def diameters(masks):
#     """
#     Calculate the diameters of the objects in the given masks.

#     Parameters:
#     masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

#     Returns:
#         tuple: A tuple containing the median diameter and an array of diameters for each object.

#     Examples:
#     >>> masks = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
#     >>> diameters(masks)
#     (1.0, array([1.41421356, 1.0, 1.0]))
#     """
#     uniq, counts = fastremap.unique(masks.astype("int32"), return_counts=True)
#     counts = counts[1:]
#     md = np.median(counts**0.5)
#     if np.isnan(md):
#         md = 0
#     md /= (np.pi**0.5) / 2
#     return md, counts**0.5


# def radius_distribution(masks, bins):
#     """
#     Calculate the radius distribution of masks.

#     Args:
#         masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)
#         bins (int): Number of bins for the histogram.

#     Returns:
#         A tuple containing a normalized histogram of radii, median radius, array of radii.

#     """
#     unique, counts = np.unique(masks, return_counts=True)
#     counts = counts[unique != 0]
#     nb, _ = np.histogram((counts**0.5) * 0.5, bins)
#     nb = nb.astype(np.float32)
#     if nb.sum() > 0:
#         nb = nb / nb.sum()
#     md = np.median(counts**0.5) * 0.5
#     if np.isnan(md):
#         md = 0
#     md /= (np.pi**0.5) / 2
#     return nb, md, (counts**0.5) / 2


# def size_distribution(masks):
#     """
#     Calculates the size distribution of masks.

#     Args:
#         masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

#     Returns:
#         float: The ratio of the 25th percentile of mask sizes to the 75th percentile of mask sizes.
#     """
#     counts = np.unique(masks, return_counts=True)[1][1:]
#     return np.percentile(counts, 25) / np.percentile(counts, 75)


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ Fills holes in masks (2D/3D) and discards masks smaller than min_size.

    This function fills holes in each mask using fill_voids.fill.
    It also removes masks that are smaller than the specified min_size.

    Parameters:
    masks (ndarray): Int, 2D or 3D array of labelled masks.
        0 represents no mask, while positive integers represent mask labels.
        The size can be [Ly x Lx] or [Lz x Ly x Lx].
    min_size (int, optional): Minimum number of pixels per mask.
        Masks smaller than min_size will be removed.
        Set to -1 to turn off this functionality. Default is 15.

    Returns:
        ndarray: Int, 2D or 3D array of masks with holes filled and small masks removed.
            0 represents no mask, while positive integers represent mask labels.
            The size is [Ly x Lx] or [Lz x Ly x Lx].
    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)

    # Filter small masks
    if min_size > 0:
        counts = fastremap.unique(masks, return_counts=True)[1][1:]
        masks = fastremap.mask(masks, np.nonzero(counts < min_size)[0] + 1)
        fastremap.renumber(masks, in_place=True)
        
    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            msk = fill_voids.fill(msk)
            masks[slc][msk] = (j + 1)
            j += 1

    if min_size > 0:
        counts = fastremap.unique(masks, return_counts=True)[1][1:]
        masks = fastremap.mask(masks, np.nonzero(counts < min_size)[0] + 1)
        fastremap.renumber(masks, in_place=True)
    
    return masks
