import logging
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log

def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time
