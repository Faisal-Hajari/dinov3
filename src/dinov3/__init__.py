# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
try:
    from importlib.metadata import version as _v
    __version__ = _v("dinov3")
except Exception:
    __version__ = "0.1.5"

from . import checkpointer
from . import configs
from . import data
from . import distributed
from . import env
from . import eval
from . import fsdp
from . import hub
from . import layers
from . import logging
from . import loss
from . import models
from . import run
from . import thirdparty
from . import train
from . import utils
from . import build_models
