"""Automatic Medical Eye Quality Enhancement Pipeline for Ocuviitals.

4-stage clinical enhancement pipeline:
1. Anatomical Auto-Crop (15% margin preserving sclera, iris, pupil, lower conjunctiva).
2. Gray-World White Balance + LAB CLAHE (clipLimit=2.5, tileGridSize=(8,8)).
3. Pretrained Super-Resolution (FSRCNN-x2) or Lanczos4 + Unsharp Masking to 512x512.
4. Returns raw vs. enhanced comparison images for Streamlit side-by-side display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from src.pretrained_loader import PretrainedModelRegistry, get_model_registry


@dataclass
class EnhancedEyeResult:
    raw_crop_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    enhanced_bgr: np.ndarray
    sr_method: str
    crop_box: Tuple[int, int, int, int]


def gray_world_white_balance(bgr_img: np.ndarray) -> np.ndarray:
    """Normalize ambient/flash color cast via Gray-World assumption."""
    img = bgr_img.astype(np.float32)
    avg_b = float(np.mean(img[:, :, 0])) + 1e-6
    avg_g = float(np.mean(img[:, :, 1])) + 1e-6
    avg_r = float(np.mean(img[:, :, 2])) + 1e-6
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    # Gentle 70% correction so true clinical pallor/icterus hues are preserved
    scale_b = 0.3 + 0.7 * (avg_gray / avg_b)
    scale_g = 0.3 + 0.7 * (avg_gray / avg_g)
    scale_r = 0.3 + 0.7 * (avg_gray / avg_r)

    img[:, :, 0] *= np.clip(scale_b, 0.75, 1.30)
    img[:, :, 1] *= np.clip(scale_g, 0.75, 1.30)
    img[:, :, 2] *= np.clip(scale_r, 0.75, 1.30)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_lab_clahe(bgr_img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """Apply CLAHE(clipLimit=2.5, tileGridSize=(8,8)) on the L-channel in LAB space."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def enhance_eye_image(
    frame_bgr: np.ndarray,
    eye_bbox: Optional[Tuple[int, int, int, int]] = None,
    target_size: Tuple[int, int] = (512, 512),
    registry: Optional[PretrainedModelRegistry] = None,
) -> EnhancedEyeResult:
    """Run the 4-stage clinical enhancement pipeline on a validated eye frame."""
    reg = registry or get_model_registry()
    h, w = frame_bgr.shape[:2]

    # Stage 1: Anatomical Auto-Crop with 15% margin
    if eye_bbox is not None:
        x, y, bw, bh = eye_bbox
        mx = int(0.15 * bw)
        my = int(0.15 * bh)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(w, x + bw + mx)
        y1 = min(h, y + bh + my)
        crop_bgr = frame_bgr[y0:y1, x0:x1]
        crop_box = (x0, y0, x1 - x0, y1 - y0)
    else:
        crop_bgr = frame_bgr.copy()
        crop_box = (0, 0, w, h)

    if crop_bgr.size == 0:
        crop_bgr = frame_bgr.copy()

    raw_resized = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    raw_crop_rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)

    # Stage 2: Gray-World White Balance + LAB CLAHE
    wb_bgr = gray_world_white_balance(crop_bgr)
    clahe_bgr = apply_lab_clahe(wb_bgr, clip_limit=2.5)

    # Stage 3: Pretrained Super-Resolution (FSRCNN-x2) or Bilateral + Lanczos + Unsharp Mask
    if reg.sr_model is not None and max(clahe_bgr.shape[:2]) < 512:
        try:
            upscaled = reg.sr_model.upsample(clahe_bgr)
            enhanced_bgr = cv2.resize(upscaled, target_size, interpolation=cv2.INTER_AREA)
            sr_method = reg.status_badges.get("Super-Resolution", "FSRCNN-x2 Pretrained")
        except Exception:
            reg.sr_model = None
            enhanced_bgr = None
            sr_method = "Lanczos4 + Unsharp Mask (Fallback)"
    else:
        enhanced_bgr = None
        sr_method = reg.status_badges.get("Super-Resolution", "Lanczos4 + Unsharp Mask")

    if enhanced_bgr is None:
        denoised = cv2.bilateralFilter(clahe_bgr, d=5, sigmaColor=35, sigmaSpace=35)
        upscaled = cv2.resize(denoised, target_size, interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.4)
        enhanced_bgr = cv2.addWeighted(upscaled, 1.35, gaussian, -0.35, 0)

    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return EnhancedEyeResult(
        raw_crop_rgb=raw_crop_rgb,
        enhanced_rgb=enhanced_rgb,
        enhanced_bgr=enhanced_bgr,
        sr_method=sr_method,
        crop_box=crop_box,
    )
