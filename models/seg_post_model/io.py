"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os
import numpy as np
import cv2
import tifffile
import logging
from tqdm import tqdm
import re

try:
    import nd2
    ND2 = True
except:
    ND2 = False

try:
    import nrrd
    NRRD = True
except:
    NRRD = False


io_logger = logging.getLogger(__name__)

def load_dax(filename):
    ### modified from ZhuangLab github:
    ### https://github.com/ZhuangLab/storm-analysis/blob/71ae493cbd17ddb97938d0ae2032d97a0eaa76b2/storm_analysis/sa_library/datareader.py#L156

    inf_filename = os.path.splitext(filename)[0] + ".inf"
    if not os.path.exists(inf_filename):
        io_logger.critical(
            f"ERROR: no inf file found for dax file {filename}, cannot load dax without it"
        )
        return None

    ### get metadata
    image_height, image_width = None, None
    # extract the movie information from the associated inf file
    size_re = re.compile(r"frame dimensions = ([\d]+) x ([\d]+)")
    length_re = re.compile(r"number of frames = ([\d]+)")
    endian_re = re.compile(r" (big|little) endian")

    with open(inf_filename, "r") as inf_file:
        lines = inf_file.read().split("\n")
        for line in lines:
            m = size_re.match(line)
            if m:
                image_height = int(m.group(2))
                image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    bigendian = 1
                else:
                    bigendian = 0
    # set defaults, warn the user that they couldn"t be determined from the inf file.
    if not image_height:
        io_logger.warning("could not determine dax image size, assuming 256x256")
        image_height = 256
        image_width = 256

    ### load image
    img = np.memmap(filename, dtype="uint16",
                    shape=(number_frames, image_height, image_width))
    if bigendian:
        img = img.byteswap()
    img = np.array(img)

    return img


def imread(filename):
    """
    Read in an image file with tif or image file type supported by cv2.

    Args:
        filename (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data as a NumPy array.

    Raises:
        None

    Raises an error if the image file format is not supported.

    Examples:
        >>> img = imread("image.tif")
    """
    # ensure that extension check is not case sensitive
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff" or ext == ".flex":
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]["shape"]
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f"reading tiff with {ltif} planes")
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i, page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)
        return img
    elif ext == ".dax":
        img = load_dax(filename)
        return img
    elif ext == ".nd2":
        if not ND2:
            io_logger.critical("ERROR: need to 'pip install nd2' to load in .nd2 file")
            return None
    elif ext == ".nrrd":
        if not NRRD:
            io_logger.critical(
                "ERROR: need to 'pip install pynrrd' to load in .nrrd file")
            return None
        else:
            img, metadata = nrrd.read(filename)
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            return img
    elif ext != ".npy":
        try:
            img = cv2.imread(filename, -1)  #cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2, 1, 0]]
            return img
        except Exception as e:
            io_logger.critical("ERROR: could not read file, %s" % e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat["masks"]
            return masks
        except Exception as e:
            io_logger.critical("ERROR: could not read masks from file, %s" % e)
            return None


def imsave(filename, arr):
    """
    Saves an image array to a file.

    Args:
        filename (str): The name of the file to save the image to.
        arr (numpy.ndarray): The image array to be saved.

    Returns:
        None
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff":
        tifffile.imwrite(filename, data=arr, compression="zlib")
    else:
        if len(arr.shape) > 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)

