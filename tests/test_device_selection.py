import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.distributed.basic import get_device, has_mps


def test_get_device_safe():
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_has_mps_safe():
    flag = has_mps()
    assert isinstance(flag, bool)
    if platform.system() != "Darwin":
        assert flag is False
