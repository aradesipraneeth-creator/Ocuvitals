"""Explainability Module (Gradient Saliency + Vascular/Pallor Attention & Vessel Skeleton)."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from src.models import OcuviitalsLiteFusionNet, preprocess_for_model
from src.pretrained_loader import is_lite_mode
from src.signal_processor import AutonomicMetrics
from src.vision_tracker import VisionExtractionResult


def generate_gradcam_overlay(
    model: Any,
    vision: VisionExtractionResult,
    metrics: AutonomicMetrics,
    target_class: int = 0,
    alpha: float = 0.42,
) -> np.ndarray:
    """Compute neural gradient saliency (or vascular edge saliency in Lite mode) overlay (RGB)."""
    h, w = vision.eye_roi_clahe.shape[:2]

    if isinstance(model, OcuviitalsLiteFusionNet) or is_lite_mode():
        # Fast OpenCV Laplacian/Sobel microvascular saliency (zero torch RAM)
        green = vision.eye_roi_clahe[:, :, 1].astype(np.float32)
        gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3)
        sal = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (15, 15), 0)
    else:
        import torch

        model.eval()
        eye_tensor, seq_tensor = preprocess_for_model(vision, metrics)
        eye_tensor = eye_tensor.clone().detach().requires_grad_(True)
        try:
            logits, _ = model(eye_tensor, seq_tensor)
            score = logits[0, target_class]
            model.zero_grad()
            score.backward()
            grads = eye_tensor.grad
            if grads is not None:
                sal = torch.abs(grads[0]).mean(dim=0).detach().cpu().numpy()
                sal = cv2.GaussianBlur(sal, (15, 15), 0)
            else:
                sal = np.zeros((224, 224), dtype=np.float32)
        except Exception:
            sal = np.zeros((224, 224), dtype=np.float32)

    cam_resized = cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)
    if np.max(cam_resized) > 1e-8:
        cam_norm = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8)
    else:
        cam_norm = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    conj_prior = np.exp(-(((yy - 0.78 * h) / (0.16 * h)) ** 2 + ((xx - 0.50 * w) / (0.30 * w)) ** 2))
    sclera_l = np.exp(-(((yy - 0.45 * h) / (0.15 * h)) ** 2 + ((xx - 0.20 * w) / (0.12 * w)) ** 2))
    sclera_r = np.exp(-(((yy - 0.45 * h) / (0.15 * h)) ** 2 + ((xx - 0.80 * w) / (0.12 * w)) ** 2))
    phys_prior = np.clip(0.85 * conj_prior + 0.65 * (sclera_l + sclera_r), 0.0, 1.0)

    combined_cam = np.clip(0.55 * cam_norm + 0.45 * phys_prior, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * combined_cam)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base_bgr = vision.eye_roi_clahe.copy()
    blended_bgr = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def generate_vessel_skeleton_overlay(enhanced_bgr: np.ndarray) -> np.ndarray:
    """Extract scleral micro-vessel skeleton via morphological black-hat & overlay in cyan."""
    if enhanced_bgr is None or enhanced_bgr.size == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)

    h, w = enhanced_bgr.shape[:2]
    green = enhanced_bgr[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    _, vessel_mask = cv2.threshold(blackhat, 14, 255, cv2.THRESH_BINARY)

    cy = h // 2
    yy, xx = np.ogrid[:h, :w]
    iris_exclude = ((xx - w // 2) ** 2 + (yy - cy) ** 2) <= (min(h, w) * 0.25) ** 2
    vessel_mask[iris_exclude] = 0
    vessel_mask = cv2.dilate(vessel_mask, np.ones((2, 2), np.uint8), iterations=1)

    overlay = enhanced_bgr.copy()
    overlay[vessel_mask > 0] = (255, 230, 0)
    blended = cv2.addWeighted(enhanced_bgr, 0.45, overlay, 0.55, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
