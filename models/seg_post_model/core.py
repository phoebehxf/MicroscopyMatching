"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging
import numpy as np
from tqdm import trange
from . import transforms, utils

import torch

TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)


def use_gpu(gpu_number=0, use_torch=True):
    """ 
    Check if GPU is available for use.

    Args:
        gpu_number (int): The index of the GPU to be used. Default is 0.
        use_torch (bool): Whether to use PyTorch for GPU check. Default is True.

    Returns:
        bool: True if GPU is available, False otherwise.

    Raises:
        ValueError: If use_torch is False, as cellpose only runs with PyTorch now.
    """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError("cellpose only runs with PyTorch now")


def _use_gpu_torch(gpu_number=0):
    """
    Checks if CUDA or MPS is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA or MPS is available and working, False otherwise.
    """
    try:
        device = torch.device("cuda:" + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info("** TORCH CUDA version installed and working. **")
        return True
    except:
        pass
    try:
        device = torch.device('mps:' + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info('** TORCH MPS version installed and working. **')
        return True
    except:
        core_logger.info('Neither TORCH CUDA nor MPS version not installed/working.')
        return False


def assign_device(use_torch=True, gpu=False, device=0):
    """
    Assigns the device (CPU or GPU or mps) to be used for computation.

    Args:
        use_torch (bool, optional): Whether to use torch for GPU detection. Defaults to True.
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        device (int or str, optional): The device index or name to be used. Defaults to 0.

    Returns:
        torch.device, bool (True if GPU is used, False otherwise)
    """

    if isinstance(device, str):
        if device != "mps" or not(gpu and torch.backends.mps.is_available()):
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        try:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device}')
                core_logger.info(">>>> using GPU (CUDA)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                core_logger.info(">>>> using GPU (MPS)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
    else:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu = False
        cpu = True
    
    if cpu:
        device = torch.device("cpu")
        core_logger.info(">>>> using CPU")
        gpu = False
    return device, gpu


def _forward(net, x, feat=None):
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob) and style features.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).to(net.device, dtype=net.dtype)
    else:
        X = x
    if feat is not None:
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat).to(net.device, dtype=net.dtype)
    net.eval()
    with torch.no_grad():
        y, style = net(X, feat=feat)[:2]
    del X
    y = y.detach().cpu().to(torch.float32).numpy()
    style = style.detach().cpu().to(torch.float32).numpy()
    return y, style


def run_net(net, imgi, feat=None, batch_size=8, augment=False, tile_overlap=0.1, bsize=224,
            rsz=None):
    """ 
    Run network on stack of images.
    
    (faster if augment is False)

    Args:
        net (class): cellpose network (model.net)
        imgi (np.ndarray): The input image or stack of images of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability. 
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    # run network
    Lz, Ly0, Lx0, nchan = imgi.shape 
    if rsz is not None:
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0 # 512, 512
    
    ly, lx = bsize, bsize # 256, 256
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize)) # 8
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2 # 528, 528
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])
    
    if augment:
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize)) # 3
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize)) # 3
    
    
    # run multiple slices at the same time
    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles) # number of imgs to run in the same batch, 1
    niter = int(np.ceil(Lz / nimgs)) # 1
    ziterator = (trange(niter, file=tqdm_out, mininterval=30) 
                    if niter > 10 or Lz > 1 else range(niter))
    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32") # 9, 3, 256, 256
        if feat is not None:
            FEATa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32") # 9, 256
        else: 
            FEATa = None
        for i, b in enumerate(inds):
            # pad image for net so Ly and Lx are divisible by 4
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2,0,1), pads, mode="constant") # 3, 528, 528

            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment,
                tile_overlap=tile_overlap) # IMG: 3, 3, 3, 256, 256
            IMGa[i * ntiles : (i+1) * ntiles] = np.reshape(IMG, 
                                            (ny * nx, nchan, ly, lx))
            if feat is not None:
                featb = transforms.resize_image(feat[b], rsz=rsz) if rsz is not None else feat[b].copy()
                featb = np.pad(featb.transpose(2,0,1), pads, mode="constant")
                FEAT, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                    featb, bsize=bsize, augment=augment,
                    tile_overlap=tile_overlap)          
                FEATa[i * ntiles : (i+1) * ntiles] = np.reshape(FEAT,
                                            (ny * nx, nchan, ly, lx))
                
        # run network
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya0, stylea0 = _forward(net, IMGa[bslc], feat=FEATa[bslc] if FEATa is not None else None)
            if j == 0:
                nout = ya0.shape[1]
                ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
                stylea = np.zeros((IMGa.shape[0], 256), "float32")
            ya[bslc] = ya0
            stylea[bslc] = stylea0

        # average tiles
        for i, b in enumerate(inds):
            if i==0 and k==0:
                yf = np.zeros((Lz, nout, Ly, Lx), "float32")
                styles = np.zeros((Lz, 256), "float32")
            y = ya[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, 3, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, :imgb.shape[-2], :imgb.shape[-1]]
            stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
            stylei /= (stylei**2).sum()**0.5
            styles[b] = stylei
    # slices from padding
    yf = yf[:, :, ypad1 : Ly-ypad2, xpad1 : Lx-xpad2]
    yf = yf.transpose(0,2,3,1)   
    return yf, np.array(styles)
