"""Pretrained Foundation & Clinical Model Hub for Ocuviitals.

Automatically downloads, caches locally in ./weights/, and loads:
1. MediaPipe Face Landmarker (face_landmarker.task) + OpenCV haarcascade_eye.xml.
2. Pretrained Vision Foundation Backbone via timm (ConvNeXt / Swin / DINOv2)
   with graceful fallback to torchvision MobileNetV3-Small.
3. Pretrained Super-Resolution model (FSRCNN_x2.pb via cv2.dnn_superres)
   with graceful fallback to Lanczos4 + Unsharp Masking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import urllib.request
import cv2

def is_lite_mode() -> bool:
    return bool(os.environ.get("RENDER") or os.environ.get("OCUVIITALS_LITE") == "1")


_HAS_MP_TASKS = False
mp = None
mp_tasks = None
mp_vision = None


WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FSRCNN_X2_URL = (
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/"
    "models/FSRCNN_x2.pb"
)


def _download_file(url: str, dest: Path, timeout: int = 15) -> bool:
    """Download a file to dest if not already cached. Returns True if available."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Ocuviitals/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return dest.exists() and dest.stat().st_size > 1024
    except Exception:
        if dest.exists() and dest.stat().st_size < 1024:
            dest.unlink(missing_ok=True)
        return False


@dataclass
class PretrainedModelRegistry:
    """Holds loaded pretrained models and their human-readable status labels."""

    face_landmarker: Any = None
    haar_eye_cascade: Optional[cv2.CascadeClassifier] = None
    vision_backbone: Any = None
    vision_feature_dim: int = 576
    sr_model: Any = None
    status_badges: Dict[str, str] = field(default_factory=dict)

    def summary_list(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.status_badges.items()]


def load_face_landmarker(weights_dir: Path = WEIGHTS_DIR) -> Tuple[Any, str]:
    """Load Google's official pretrained MediaPipe Face Landmarker task (skipped in 512MB Lite mode)."""
    if is_lite_mode():
        return None, "OpenCV Haar + Anatomical Iris (Render Low-RAM)"

    try:
        from mediapipe.tasks import python as mp_tasks  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    except Exception:
        return None, "Unavailable (mediapipe.tasks missing)"

    task_path = weights_dir / "face_landmarker.task"
    if not _download_file(FACE_LANDMARKER_URL, task_path, timeout=20):
        return None, "Offline (face_landmarker.task not cached)"

    try:
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(task_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return landmarker, "MediaPipe FaceLandmarker v2 (478 landmarks)"
    except Exception:
        return None, "Unavailable (FaceLandmarker init failed)"


def load_haar_eye_cascade() -> Tuple[Optional[cv2.CascadeClassifier], str]:
    """Load OpenCV's pretrained haarcascade_eye.xml."""
    haar_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
    if not haar_path.exists():
        return None, "Unavailable"
    cascade = cv2.CascadeClassifier(str(haar_path))
    if cascade.empty():
        return None, "Unavailable"
    return cascade, "OpenCV Haar Eye Cascade"


def load_vision_backbone() -> Tuple[Any, int, str]:
    """Load pretrained foundation vision backbone, skipping heavy torch on 512MB Render."""
    if is_lite_mode():
        return None, 576, "Render-Lite Spatial Colorimetric + Vascular Engine (Low-RAM)"

    import socket
    import torch
    import torch.nn as nn
    from torchvision import models

    try:
        import timm  # type: ignore
        has_timm = True
    except Exception:
        timm = None
        has_timm = False

    candidates = [
        ("swin_tiny_patch4_window7_224", "Swin-Tiny Pretrained (timm)"),
        ("convnext_small.fb_in22k_ft_in1k", "ConvNeXt-Small IN22k Pretrained (timm)"),
        ("vit_small_patch14_dinov2.lvd142m", "DINOv2 ViT-Small Pretrained (timm)"),
    ]
    if has_timm:
        # Avoid 23s+ huggingface_hub retry backoff when offline
        online = False
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=1.5):
                online = True
        except OSError:
            online = False

        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if not online:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            for model_name, label in candidates:
                try:
                    m = timm.create_model(model_name, pretrained=True, num_classes=0)
                    m.eval()
                    feat_dim = int(getattr(m, "num_features", 768))
                    return m, feat_dim, label
                except Exception:
                    continue
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

    # Graceful fallback to torchvision pretrained MobileNetV3-Small
    try:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        label = "MobileNetV3-Small Pretrained (torchvision)"
    except Exception:
        m = models.mobilenet_v3_small(weights=None)
        label = "MobileNetV3-Small (Offline Init)"

    class _MobilenetBackbone(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(self.features(x)).flatten(1)

    return _MobilenetBackbone(m), 576, label


def load_super_resolution(weights_dir: Path = WEIGHTS_DIR) -> Tuple[Any, str]:
    """Load pretrained FSRCNN_x2 super-resolution via cv2.dnn_superres, with Lanczos fallback."""
    has_dnn_sr = hasattr(cv2, "dnn_superres")
    if has_dnn_sr:
        sr_path = weights_dir / "FSRCNN_x2.pb"
        if _download_file(FSRCNN_X2_URL, sr_path, timeout=12):
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(str(sr_path))
                sr.setModel("fsrcnn", 2)
                return sr, "FSRCNN-x2 Pretrained (OpenCV DNN)"
            except Exception:
                pass
    return None, "Lanczos4 + Unsharp Mask (CPU Fallback)"


_CACHED_REGISTRY: Optional[PretrainedModelRegistry] = None


def get_model_registry(force_reload: bool = False) -> PretrainedModelRegistry:
    """Return cached PretrainedModelRegistry with all foundation models loaded."""
    global _CACHED_REGISTRY
    if _CACHED_REGISTRY is not None and not force_reload:
        return _CACHED_REGISTRY

    landmarker, lm_label = load_face_landmarker()
    haar, haar_label = load_haar_eye_cascade()
    backbone, feat_dim, bb_label = load_vision_backbone()
    sr_model, sr_label = load_super_resolution()

    badges = {
        "Landmarks": lm_label,
        "Eye Cascade": haar_label,
        "Vision Backbone": bb_label,
        "Super-Resolution": sr_label,
    }
    _CACHED_REGISTRY = PretrainedModelRegistry(
        face_landmarker=landmarker,
        haar_eye_cascade=haar,
        vision_backbone=backbone,
        vision_feature_dim=feat_dim,
        sr_model=sr_model,
        status_badges=badges,
    )
    return _CACHED_REGISTRY
