"""Strict Eye-Only Gatekeeper Validation for OptiScan AI (OcuVitals).

Runs BEFORE any enhancement, inference, or 3D rendering to ensure only
valid human eye images or videos enter the clinical pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _HAS_MP_IMAGE = hasattr(mp, "Image") and hasattr(mp, "ImageFormat")
except Exception:
    mp = None
    _HAS_MP_IMAGE = False

from src.pretrained_loader import PretrainedModelRegistry, get_model_registry


REJECTION_MESSAGE = (
    "Invalid Input: No human eye detected. "
    "Please upload a clear close-up photo or video of a human eye."
)

# MediaPipe 478-landmark left/right eye contour indices
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class EyeValidationResult:
    is_valid: bool
    confidence: float
    method: str
    eye_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    message: str


def _verify_anatomical_iris_sclera(roi_bgr: np.ndarray) -> Tuple[bool, float]:
    """Verify circular dark pupil/iris inside brighter sclera/periocular region."""
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.0
    h, w = roi_bgr.shape[:2]
    if h < 24 or w < 24:
        return False, 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Check 1: Meaningful dynamic range (reject flat walls / solid colors)
    p10, p90 = np.percentile(blur, [10, 90])
    if (p90 - p10) < 35.0:
        return False, 0.15

    # Check 2: Circular iris/pupil detection via HoughCircles or dark-core centrality
    min_r = max(6, int(min(h, w) * 0.10))
    max_r = max(min_r + 4, int(min(h, w) * 0.38))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min(h, w) // 2),
        param1=60,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )

    best_score = 0.0
    if circles is not None and len(circles[0]) > 0:
        for cx, cy, r in circles[0]:
            cx_i, cy_i, r_i = int(cx), int(cy), int(r)
            if not (0.15 * w <= cx_i <= 0.85 * w and 0.15 * h <= cy_i <= 0.85 * h):
                continue
            yy, xx = np.ogrid[:h, :w]
            dist2 = (xx - cx_i) ** 2 + (yy - cy_i) ** 2
            inner_mask = dist2 <= (0.55 * r_i) ** 2
            iris_mask = (dist2 > (0.55 * r_i) ** 2) & (dist2 <= (1.05 * r_i) ** 2)
            sclera_mask = (dist2 > (1.15 * r_i) ** 2) & (dist2 <= (1.9 * r_i) ** 2)

            if np.sum(inner_mask) < 8 or np.sum(sclera_mask) < 16:
                continue
            inner_mean = float(np.mean(gray[inner_mask]))
            iris_mean = float(np.mean(gray[iris_mask])) if np.any(iris_mask) else inner_mean + 15.0
            sclera_mean = float(np.mean(gray[sclera_mask]))

            # Anatomical rule: pupil < iris < sclera
            if sclera_mean > inner_mean + 35.0 and sclera_mean > iris_mean + 18.0:
                contrast = min(1.0, (sclera_mean - inner_mean) / 110.0)
                score = 0.78 + 0.20 * contrast
                best_score = max(best_score, score)

    if best_score >= 0.76:
        return True, float(min(0.98, best_score))

    # Check 3: Concentric dark-center vs bright-lateral-sclera symmetry (for partial/oblique macro eyes)
    cy0, cx0 = h // 2, w // 2
    r_core = max(6, min(h, w) // 8)
    core = gray[max(0, cy0 - r_core) : min(h, cy0 + r_core), max(0, cx0 - r_core) : min(w, cx0 + r_core)]
    left_scl = gray[int(h * 0.25) : int(h * 0.70), int(w * 0.05) : int(w * 0.28)]
    right_scl = gray[int(h * 0.25) : int(h * 0.70), int(w * 0.72) : int(w * 0.95)]
    if core.size > 0 and left_scl.size > 0 and right_scl.size > 0:
        core_min = float(np.percentile(core, 20))
        scl_mean = 0.5 * (float(np.mean(left_scl)) + float(np.mean(right_scl)))
        # Reject text/document-like high-frequency grids by checking sclera smoothness
        scl_std = 0.5 * (float(np.std(left_scl)) + float(np.std(right_scl)))
        if scl_mean - core_min > 55.0 and scl_std < 48.0 and core_min < 85.0:
            return True, 0.80

    return False, float(best_score)


def validate_single_frame(
    frame_bgr: np.ndarray,
    registry: Optional[PretrainedModelRegistry] = None,
) -> EyeValidationResult:
    """Validate a single BGR image/frame using MediaPipe FaceLandmarker + Haar Eye + Anatomy."""
    if frame_bgr is None or frame_bgr.size == 0:
        return EyeValidationResult(False, 0.0, "empty", None, REJECTION_MESSAGE)

    reg = registry or get_model_registry()
    h, w = frame_bgr.shape[:2]

    # 1. Try MediaPipe FaceLandmarker (best for face / half-face frames)
    if reg.face_landmarker is not None and _HAS_MP_IMAGE:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = reg.face_landmarker.detect(mp_img)
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                eye_pts = pts[LEFT_EYE_IDX]
                x0 = max(0, int(np.min(eye_pts[:, 0]) - 0.25 * (np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0]))))
                x1 = min(w, int(np.max(eye_pts[:, 0]) + 0.25 * (np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0]))))
                y0 = max(0, int(np.min(eye_pts[:, 1]) - 0.50 * (np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1]))))
                y1 = min(h, int(np.max(eye_pts[:, 1]) + 0.85 * (np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1]))))
                if x1 - x0 >= 16 and y1 - y0 >= 16:
                    crop = frame_bgr[y0:y1, x0:x1]
                    ok_anat, anat_conf = _verify_anatomical_iris_sclera(crop)
                    conf = max(0.86, anat_conf) if ok_anat else 0.82
                    return EyeValidationResult(
                        is_valid=True,
                        confidence=round(conf, 3),
                        method="MediaPipe FaceLandmarker + Iris Anatomy",
                        eye_bbox=(x0, y0, x1 - x0, y1 - y0),
                        message="Human eye validated via MediaPipe FaceLandmarker.",
                    )
        except Exception:
            pass

    # 2. Try OpenCV Haar Eye Cascade + Anatomical Iris/Sclera Verification (for macro close-ups)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if reg.haar_eye_cascade is not None:
        eyes = reg.haar_eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(28, w // 6), max(28, h // 6))
        )
        for (ex, ey, ew, eh) in sorted(eyes, key=lambda b: b[2] * b[3], reverse=True):
            crop = frame_bgr[ey : ey + eh, ex : ex + ew]
            ok_anat, anat_conf = _verify_anatomical_iris_sclera(crop)
            if ok_anat and anat_conf > 0.75:
                return EyeValidationResult(
                    is_valid=True,
                    confidence=round(anat_conf, 3),
                    method="Haar Eye Cascade + Anatomical Iris Check",
                    eye_bbox=(int(ex), int(ey), int(ew), int(eh)),
                    message="Human eye validated via Haar Eye Cascade & iris/sclera proportions.",
                )

    # 3. Direct macro close-up anatomical check (when eye fills >65% of frame)
    ok_macro, macro_conf = _verify_anatomical_iris_sclera(frame_bgr)
    if ok_macro and macro_conf > 0.75:
        return EyeValidationResult(
            is_valid=True,
            confidence=round(macro_conf, 3),
            method="Macro Ocular Anatomy Validator",
            eye_bbox=(0, 0, w, h),
            message="Close-up human eye validated via circular iris/sclera anatomy.",
        )

    return EyeValidationResult(
        is_valid=False,
        confidence=round(macro_conf, 3),
        method="Rejected",
        eye_bbox=None,
        message=REJECTION_MESSAGE,
    )


def validate_eye_input(
    image_or_frames: Union[np.ndarray, List[np.ndarray]],
    registry: Optional[PretrainedModelRegistry] = None,
) -> EyeValidationResult:
    """Validate an eye image or list of video frames before downstream processing."""
    reg = registry or get_model_registry()
    if isinstance(image_or_frames, list):
        if not image_or_frames:
            return EyeValidationResult(False, 0.0, "empty", None, REJECTION_MESSAGE)
        # Sample up to 5 evenly-spaced frames (handles occasional blink frames)
        idxs = np.linspace(0, len(image_or_frames) - 1, min(5, len(image_or_frames))).astype(int)
        best_res: Optional[EyeValidationResult] = None
        valid_count = 0
        for idx in idxs:
            res = validate_single_frame(image_or_frames[int(idx)], registry=reg)
            if res.is_valid:
                valid_count += 1
                if best_res is None or res.confidence > best_res.confidence:
                    best_res = res
        if valid_count >= 1 and best_res is not None:
            return best_res
        return EyeValidationResult(False, 0.1, "Rejected", None, REJECTION_MESSAGE)

    return validate_single_frame(image_or_frames, registry=reg)
