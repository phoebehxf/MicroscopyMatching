"""Regionprops features and its augmentations.
WindowedRegionFeatures (WRFeatures) is a class that holds regionprops features for a windowed track region.
Modified from Trackastra (https://github.com/weigertlab/trackastra)
"""

import itertools
import logging
from collections import OrderedDict
from collections.abc import Iterable #, Sequence
from functools import reduce
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from edt import edt
from skimage.measure import regionprops, regionprops_table
from tqdm import tqdm
from typing import Tuple, Optional, Sequence, Union, List
import typing
import torch
logger = logging.getLogger(__name__)

_PROPERTIES = {
    "regionprops": (
        "area",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        "inertia_tensor",
    ),
    "regionprops2": (
        "equivalent_diameter_area",
        "intensity_mean",
        "inertia_tensor",
        "border_dist",
    ),
}


def _border_dist_fast(mask: np.ndarray, cutoff: float = 5):
    cutoff = int(cutoff)
    border = np.ones(mask.shape, dtype=np.float32)
    ndim = len(mask.shape)

    for axis, size in enumerate(mask.shape):
        # Create fade values for the band [0, cutoff)
        band_vals = np.arange(cutoff, dtype=np.float32) / cutoff

        # Build slices for the low border
        low_slices = [slice(None)] * ndim
        low_slices[axis] = slice(0, cutoff)
        border_low = border[tuple(low_slices)]
        border_low_vals = np.minimum(
            border_low, band_vals[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(low_slices)] = border_low_vals

        # Build slices for the high border
        high_slices = [slice(None)] * ndim
        high_slices[axis] = slice(size - cutoff, size)
        band_vals_rev = band_vals[::-1]
        border_high = border[tuple(high_slices)]
        border_high_vals = np.minimum(
            border_high, band_vals_rev[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(high_slices)] = border_high_vals

    dist = 1 - border
    return tuple(r.intensity_max for r in regionprops(mask, intensity_image=dist))


class WRFeatures:
    """regionprops features for a windowed track region."""

    def __init__(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        timepoints: np.ndarray,
        features: typing.OrderedDict[str, np.ndarray],
    ):
        self.ndim = coords.shape[-1]
        if self.ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        self.coords = coords
        self.labels = labels
        self.features = features.copy()
        self.timepoints = timepoints

    def __repr__(self):
        s = (
            f"WindowRegionFeatures(ndim={self.ndim}, nregions={len(self.labels)},"
            f" ntimepoints={len(np.unique(self.timepoints))})\n\n"
        )
        for k, v in self.features.items():
            s += f"{k:>20} -> {v.shape}\n"
        return s

    @property
    def features_stacked(self):
        return np.concatenate([v for k, v in self.features.items()], axis=-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if key in self.features:
            return self.features[key]
        else:
            raise KeyError(f"Key {key} not found in features")

    @classmethod
    def concat(cls, feats: Sequence["WRFeatures"]) -> "WRFeatures":
        """Concatenate multiple WRFeatures into a single one."""
        if len(feats) == 0:
            raise ValueError("Cannot concatenate empty list of features")
        return reduce(lambda x, y: x + y, feats)

    def __add__(self, other: "WRFeatures") -> "WRFeatures":
        """Concatenate two WRFeatures."""
        if self.ndim != other.ndim:
            raise ValueError("Cannot concatenate features of different dimensions")
        if self.features.keys() != other.features.keys():
            raise ValueError("Cannot concatenate features with different properties")

        coords = np.concatenate([self.coords, other.coords], axis=0)
        labels = np.concatenate([self.labels, other.labels], axis=0)
        timepoints = np.concatenate([self.timepoints, other.timepoints], axis=0)

        features = OrderedDict(
            (k, np.concatenate([v, other.features[k]], axis=0))
            for k, v in self.features.items()
        )

        return WRFeatures(
            coords=coords, labels=labels, timepoints=timepoints, features=features
        )

    @classmethod
    def from_mask_img(
        cls,
        mask: np.ndarray,
        img: np.ndarray,
        properties="regionprops2",
        t_start: int = 0,
    ):
        img = np.asarray(img)
        mask = np.asarray(mask)

        _ntime, ndim = mask.shape[0], mask.ndim - 1
        if ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        properties = tuple(_PROPERTIES[properties])
        if "label" in properties or "centroid" in properties:
            raise ValueError(
                f"label and centroid should not be in properties {properties}"
            )

        if "border_dist" in properties:
            use_border_dist = True
            # remove border_dist from properties
            properties = tuple(p for p in properties if p != "border_dist")
        else:
            use_border_dist = False

        df_properties = ("label", "centroid", *properties)
        dfs = []
        for i, (y, x) in enumerate(zip(mask, img)):
            _df = pd.DataFrame(
                regionprops_table(y, intensity_image=x, properties=df_properties)
            )
            _df["timepoint"] = i + t_start

            if use_border_dist:
                _df["border_dist"] = _border_dist_fast(y)

            dfs.append(_df)
        df = pd.concat(dfs)

        if use_border_dist:
            properties = (*properties, "border_dist")

        timepoints = df["timepoint"].values.astype(np.int32)
        labels = df["label"].values.astype(np.int32)
        coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)

        features = OrderedDict(
            (
                p,
                np.stack(
                    [
                        df[c].values.astype(np.float32)
                        for c in df.columns
                        if c.startswith(p)
                    ],
                    axis=-1,
                ),
            )
            for p in properties
        )

        return cls(
            coords=coords, labels=labels, timepoints=timepoints, features=features
        )


def get_features(
    detections: np.ndarray,
    imgs: Optional[np.ndarray] = None,
    features: Literal["none", "wrfeat"] = "wrfeat",
    ndim: int = 2,
    n_workers=0,
    progbar_class=tqdm,
) -> List[WRFeatures]:
    detections = _check_dimensions(detections, ndim)
    imgs = _check_dimensions(imgs, ndim)
    logger.info(f"Extracting features from {len(detections)} detections")
    if n_workers > 0:
        logger.info(f"Using {n_workers} processes for feature extraction")
        features = joblib.Parallel(n_jobs=n_workers, backend="loky")(
            joblib.delayed(WRFeatures.from_mask_img)(
                # New axis for time component
                mask=mask[np.newaxis, ...].copy(),
                img=img[np.newaxis, ...].copy(),
                t_start=t,
            )
            for t, (mask, img) in progbar_class(
                enumerate(zip(detections, imgs)),
                total=len(imgs),
                desc="Extracting features",
            )
        )
    else:
        logger.info("Using single process for feature extraction")
        features = tuple(
            WRFeatures.from_mask_img(
                mask=mask[np.newaxis, ...],
                img=img[np.newaxis, ...],
                t_start=t,
            )
            for t, (mask, img) in progbar_class(
                enumerate(zip(detections, imgs)),
                total=len(imgs),
                desc="Extracting features",
            )
        )

    return features


def _check_dimensions(x: np.ndarray, ndim: int):
    if ndim == 2 and not x.ndim == 3:
        raise ValueError(f"Expected 2D data, got {x.ndim - 1}D data")
    elif ndim == 3:
        # if ndim=3 and data is two dimensional, it will be cast to 3D
        if x.ndim == 3:
            x = np.expand_dims(x, axis=1)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D data, got {x.ndim - 1}D data")
    return x


def build_windows_sd(
    features: List[WRFeatures], imgs_enc, imgs_stable, boxes, imgs, masks, window_size: int, progbar_class=tqdm
) -> List[dict]:
    windows = []
    for t1, t2 in progbar_class(
        zip(range(0, len(features)), range(window_size, len(features) + 1)),
        total=len(features) - window_size + 1,
        desc="Building windows",
    ):
        feat = WRFeatures.concat(features[t1:t2])

        labels = feat.labels
        timepoints = feat.timepoints
        coords = feat.coords

        if len(feat) == 0:
            coords = np.zeros((0, feat.ndim), dtype=int)

        w = dict(
            coords=coords,
            t1=t1,
            labels=labels,
            timepoints=timepoints,
            features=feat.features_stacked,
            img_enc=imgs_enc[t1:t2],
            image_stable=imgs_stable[t1:t2],
            boxes=boxes,
            img=imgs[t1:t2],
            mask=masks[t1:t2],
            coords_t=torch.tensor(coords, dtype=torch.float32),
            labels_t=torch.tensor(labels, dtype=torch.int32),
            timepoints_t=torch.tensor(timepoints, dtype=torch.int64),
            features_t=torch.tensor(feat.features_stacked, dtype=torch.float32),
            img_t=torch.tensor(imgs[t1:t2], dtype=torch.float32),
            mask_t=torch.tensor(masks[t1:t2], dtype=torch.int32),
        )
        windows.append(w)

    logger.debug(f"Built {len(windows)} track windows.\n")
    return windows

