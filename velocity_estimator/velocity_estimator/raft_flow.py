"""
Isolated RAFT loading + optical-flow computation.
=====================================================
RAFT's own code (RAFT/core/*.py) uses flat, script-style imports
(`from utils.utils import ...`, `from update import ...`, ...) that only
resolve if RAFT/core itself is on sys.path. The original scripts in this
repo did `sys.path.insert(0, "RAFT/core")` at import time, which caused two
problems once this project is embedded into a larger application (e.g. a
ROS node):

  1. "RAFT/core" is a relative path, resolved against the process's current
     working directory - it broke the moment the importing code ran from
     anywhere else (which is normal for a ROS node started via roslaunch).
  2. It makes RAFT/core/utils permanently importable as the top-level
     module `utils` (and `raft`, `update`, `corr`, `extractor` likewise) for
     the rest of the process. That collides with any other module named
     `utils` elsewhere in the host application - close to guaranteed in a
     ROS workspace - and once Python has cached the wrong `utils` in
     sys.modules, every later `import utils` anywhere in the process gets
     RAFT's version instead, regardless of sys.path.

load_raft() below loads RAFT exactly once (subsequent calls are cached),
using an absolute path, and restores sys.path/sys.modules to their prior
state again afterwards, so importing this module has no lasting side
effect on the rest of the process.
"""

import argparse
import importlib
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
RAFT_CORE_DIR = REPO_ROOT / "RAFT" / "core"
DEFAULT_RAFT_WEIGHTS = REPO_ROOT / "RAFT" / "models" / "raft-kitti.pth"

_COLLIDING_MODULE_NAMES = ("raft", "update", "extractor", "corr", "utils", "utils.utils")

_load_lock = threading.Lock()
_raft_symbols_cache = {}  # str(raft_core_dir) -> cached (RAFT_cls, InputPadder_cls)


def _load_raft_symbols(raft_core_dir: Optional[str] = None):
    """Import RAFT's `RAFT` model class and `InputPadder`, without leaking
    RAFT/core's generically-named modules into sys.modules/sys.path.

    raft_core_dir: override for where RAFT/core lives - only needed when
    this module is deployed somewhere that doesn't have RAFT/ as its own
    sibling directory (e.g. copied into a ROS package with a different
    layout). Defaults to RAFT_CORE_DIR (this repo's own RAFT/core)."""
    core_dir = Path(raft_core_dir) if raft_core_dir else RAFT_CORE_DIR
    cache_key = str(core_dir)
    if cache_key in _raft_symbols_cache:
        return _raft_symbols_cache[cache_key]

    with _load_lock:
        if cache_key in _raft_symbols_cache:
            return _raft_symbols_cache[cache_key]

        if not core_dir.is_dir():
            raise FileNotFoundError(
                f"RAFT core directory not found at {core_dir}. "
                "Did you clone the RAFT submodule (git submodule update --init)?"
            )

        saved_modules = {name: sys.modules.get(name) for name in _COLLIDING_MODULE_NAMES}
        sys.path.insert(0, str(core_dir))
        try:
            for name in _COLLIDING_MODULE_NAMES:
                sys.modules.pop(name, None)
            raft_module = importlib.import_module("raft")
            utils_module = importlib.import_module("utils.utils")
            RAFT_cls = raft_module.RAFT
            InputPadder_cls = utils_module.InputPadder
        finally:
            sys.path.remove(str(core_dir))
            for name, mod in saved_modules.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        _raft_symbols_cache[cache_key] = (RAFT_cls, InputPadder_cls)
        return _raft_symbols_cache[cache_key]


class RaftFlowEstimator:
    """Loads RAFT once and turns pairs of consecutive RGB frames into the
    RGB-encoded optical-flow image the paper's speed estimator expects."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        RAFT_cls, InputPadder_cls = _load_raft_symbols()
        self._InputPadder = InputPadder_cls
        self.device = device

        weights_path = Path(weights_path) if weights_path else DEFAULT_RAFT_WEIGHTS
        if not weights_path.is_file():
            raise FileNotFoundError(f"RAFT weights not found at {weights_path}")

        # muss argparse.Namespace sein, nicht z.B. SimpleNamespace: RAFT prüft
        # intern via `'dropout' not in self.args`, was Namespace.__contains__
        # braucht (das SimpleNamespace nicht implementiert).
        raft_args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
        model = RAFT_cls(raft_args)
        state_dict = torch.load(weights_path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def compute_flow(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """img1_rgb, img2_rgb: consecutive (H, W, 3) uint8 RGB frames.
        Returns the RAW (H, W, 2) float32 flow field (dx, dy) - no color
        encoding. Used by compute_flow_rgb() below, and directly by callers
        that want the raw flow (e.g. efficientnet_baseline/)."""
        # np.ascontiguousarray: callers may hand in views with negative
        # strides (e.g. a manual bgr[:, :, ::-1] channel flip) - torch.from_numpy
        # rejects those, so normalize regardless of what the caller's frame
        # source looks like.
        image1 = torch.from_numpy(np.ascontiguousarray(img1_rgb)).permute(2, 0, 1).float()[None].to(self.device)
        image2 = torch.from_numpy(np.ascontiguousarray(img2_rgb)).permute(2, 0, 1).float()[None].to(self.device)

        padder = self._InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)
        return flow_up[0].permute(1, 2, 0).cpu().numpy()

    def compute_flow_rgb(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """Returns the (H, W, 3) uint8 HSV->RGB flow visualization used as
        the paper-architecture speed estimators' input (see compute_flow()
        for the raw, unencoded flow)."""
        flow = self.compute_flow(img1_rgb, img2_rgb)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
