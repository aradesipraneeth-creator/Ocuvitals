"""Vision & ROI Tracker for Ocuviitals.

Uses OpenCV and MediaPipe FaceMesh (with refine_landmarks=True) to extract:
1. Normalized pupil-to-iris diameter ratio time-series.
2. Green-channel mean intensity from the scleral/periocular ROI (rPPG).
3. High-resolution CLAHE-enhanced cropped eye/sclera/lower-eyelid ROI.

Includes graceful blink/missing-face interpolation and a close-up eye fallback
when working with zoomed-in eye videos or environments without MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _HAS_MP_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
except Exception:
    mp = None
    _HAS_MP_SOLUTIONS = False


# MediaPipe FaceMesh landmark indices (with refine_landmarks=True)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Vertical/horizontal landmarks for Eye Aspect Ratio (EAR) blink detection
LEFT_EAR_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EAR_IDX = (362, 385, 387, 263, 373, 380)


@dataclass
class VisionExtractionResult:
    """Container for frame-by-frame extracted ocular & rPPG signals."""

    fps: float
    timestamps: np.ndarray
    pupil_ratio: np.ndarray
    rppg_green: np.ndarray
    rppg_rgb: np.ndarray
    eye_roi_clahe: np.ndarray
    eye_roi_raw: np.ndarray
    conjunctiva_roi: np.ndarray
    sclera_roi: np.ndarray
    blink_mask: np.ndarray
    valid_frames: int
    total_frames: int
    tracking_method: str = "mediapipe"
    notes: List[str] = field(default_factory=list)


class EyeVisionTracker:
    """Extracts pupil dynamics, scleral rPPG, and CLAHE-enhanced eye ROIs."""

    def __init__(self, ear_blink_threshold: float = 0.18, roi_size: Tuple[int, int] = (224, 224)) -> None:
        self.ear_blink_threshold = ear_blink_threshold
        self.roi_size = roi_size
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self._face_mesh = None
        if _HAS_MP_SOLUTIONS:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def apply_clahe_bgr(self, bgr_img: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L-channel in LAB space to preserve color balance."""
        if bgr_img is None or bgr_img.size == 0:
            return np.zeros((self.roi_size[1], self.roi_size[0], 3), dtype=np.uint8)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_eq = self._clahe.apply(l_chan)
        merged = cv2.merge((l_eq, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _compute_ear(pts: np.ndarray, idx: Tuple[int, int, int, int, int, int]) -> float:
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
        vert1 = np.linalg.norm(p2 - p6)
        vert2 = np.linalg.norm(p3 - p5)
        horiz = np.linalg.norm(p1 - p4) + 1e-6
        return float((vert1 + vert2) / (2.0 * horiz))

    @staticmethod
    def _estimate_pupil_from_iris_patch(gray_patch: np.ndarray) -> float:
        """Estimate pupil-to-iris diameter ratio inside a cropped iris/eye patch."""
        if gray_patch.size == 0:
            return 0.40
        h, w = gray_patch.shape[:2]
        # Focus on central circular region where pupil resides
        cy, cx = h // 2, w // 2
        radius = max(4, min(h, w) // 2)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        vals = gray_patch[mask]
        if vals.size < 16:
            return 0.40
        # Pupil is significantly darker than surrounding iris
        thresh = np.percentile(vals, 24)
        pupil_pixels = np.sum((gray_patch <= thresh) & mask)
        iris_area = float(np.sum(mask)) + 1e-6
        area_ratio = pupil_pixels / iris_area
        diam_ratio = float(np.sqrt(np.clip(area_ratio, 0.04, 0.65)))
        return float(np.clip(diam_ratio, 0.20, 0.72))

    def _extract_with_mediapipe(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, bool]]:
        """Return (pupil_ratio, rppg_green, eye_crop, conjunctiva_crop, sclera_crop, is_blink)."""
        if self._face_mesh is None:
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        ear_l = self._compute_ear(pts, LEFT_EAR_IDX)
        ear_r = self._compute_ear(pts, RIGHT_EAR_IDX)
        ear = 0.5 * (ear_l + ear_r)
        is_blink = ear < self.ear_blink_threshold

        # Choose left eye contour + lower eyelid margin for clinical ROI
        eye_pts = pts[LEFT_EYE_CONTOUR]
        x_min = max(0, int(np.min(eye_pts[:, 0]) - 0.28 * (np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0]))))
        x_max = min(w, int(np.max(eye_pts[:, 0]) + 0.28 * (np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0]))))
        y_min = max(0, int(np.min(eye_pts[:, 1]) - 0.55 * (np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1]))))
        y_max = min(h, int(np.max(eye_pts[:, 1]) + 0.95 * (np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1]))))

        if x_max - x_min < 12 or y_max - y_min < 12:
            return None

        eye_crop = frame_bgr[y_min:y_max, x_min:x_max]

        # Iris bounding box for pupil ratio
        iris_pts = pts[LEFT_IRIS]
        ix_min = max(0, int(np.min(iris_pts[:, 0])))
        ix_max = min(w, int(np.max(iris_pts[:, 0])))
        iy_min = max(0, int(np.min(iris_pts[:, 1])))
        iy_max = min(h, int(np.max(iris_pts[:, 1])))
        iris_patch = cv2.cvtColor(frame_bgr[iy_min:iy_max, ix_min:ix_max], cv2.COLOR_BGR2GRAY) if (ix_max > ix_min and iy_max > iy_min) else np.array([])
        pupil_ratio = self._estimate_pupil_from_iris_patch(iris_patch)

        # Scleral & periocular ROI for green-channel rPPG
        eh, ew = eye_crop.shape[:2]
        sclera_crop = eye_crop[int(eh * 0.25) : int(eh * 0.58), int(ew * 0.12) : int(ew * 0.88)]
        conj_crop = eye_crop[int(eh * 0.58) : int(eh * 0.90), int(ew * 0.20) : int(ew * 0.80)]

        if sclera_crop.size == 0:
            sclera_crop = eye_crop
        if conj_crop.size == 0:
            conj_crop = eye_crop

        rppg_green = float(np.mean(sclera_crop[:, :, 1]))
        rppg_rgb = np.mean(sclera_crop[:, :, ::-1], axis=(0, 1)).astype(np.float64)
        return pupil_ratio, rppg_green, rppg_rgb, eye_crop, conj_crop, sclera_crop, is_blink

    def _extract_closeup_fallback(
        self, frame_bgr: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Fallback for close-up eye videos/images where full face is not visible."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect dark pupil/iris center near middle of frame
        cx0, cy0 = w // 2, int(h * 0.46)
        search_r = max(16, min(h, w) // 3)
        sx0, sx1 = max(0, cx0 - search_r), min(w, cx0 + search_r)
        sy0, sy1 = max(0, cy0 - search_r), min(h, cy0 + search_r)
        center_patch = gray[sy0:sy1, sx0:sx1]

        # Blink heuristic: if central region has very high uniform brightness (eyelid skin), treat as blink
        p10 = np.percentile(center_patch, 10) if center_patch.size else 120.0
        p90 = np.percentile(center_patch, 90) if center_patch.size else 120.0
        is_blink = (p90 - p10) < 22.0

        # Estimate pupil radius using dark-threshold contour inside central iris zone
        pupil_ratio = 0.40
        if not is_blink and center_patch.size > 64:
            blur = cv2.GaussianBlur(center_patch, (7, 7), 0)
            # Iris is darker than sclera; pupil is darkest core
            iris_thresh = np.percentile(blur, 38)
            pupil_thresh = np.percentile(blur, 14)
            iris_area = float(np.sum(blur <= iris_thresh)) + 1e-6
            pupil_area = float(np.sum(blur <= pupil_thresh))
            pupil_ratio = float(np.clip(np.sqrt(pupil_area / iris_area), 0.22, 0.72))

        # Scleral & conjunctival sub-regions
        sclera_left = frame_bgr[int(h * 0.30) : int(h * 0.60), int(w * 0.08) : int(w * 0.30)]
        sclera_right = frame_bgr[int(h * 0.30) : int(h * 0.60), int(w * 0.70) : int(w * 0.92)]
        sclera_crop = np.concatenate([sclera_left, sclera_right], axis=1) if sclera_left.size and sclera_right.size else frame_bgr
        conj_crop = frame_bgr[int(h * 0.66) : int(h * 0.92), int(w * 0.22) : int(w * 0.78)]
        if conj_crop.size == 0:
            conj_crop = frame_bgr

        # Green channel mean & full RGB mean across scleral/periocular vascular zone
        rppg_green = float(0.6 * np.mean(sclera_crop[:, :, 1]) + 0.4 * np.mean(conj_crop[:, :, 1]))
        scl_rgb = np.mean(sclera_crop[:, :, ::-1], axis=(0, 1))
        conj_rgb = np.mean(conj_crop[:, :, ::-1], axis=(0, 1))
        rppg_rgb = (0.6 * scl_rgb + 0.4 * conj_rgb).astype(np.float64)
        return pupil_ratio, rppg_green, rppg_rgb, frame_bgr, conj_crop, sclera_crop, is_blink

    @staticmethod
    def _interpolate_invalid(values: np.ndarray, invalid_mask: np.ndarray, default_val: float) -> np.ndarray:
        """Linearly interpolate over blink or missed-detection frames."""
        out = values.astype(np.float64).copy()
        valid_idx = np.where(~invalid_mask)[0]
        if valid_idx.size == 0:
            return np.full_like(out, default_val, dtype=np.float64)
        if valid_idx.size == len(out):
            return out
        invalid_idx = np.where(invalid_mask)[0]
        out[invalid_idx] = np.interp(invalid_idx, valid_idx, out[valid_idx])
        return out

    def process_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> VisionExtractionResult:
        """Process a list of BGR frames into aligned ocular signals and ROIs."""
        if not frames:
            raise ValueError("No frames provided to EyeVisionTracker.")

        fps = float(fps) if fps and fps > 1.0 else 30.0
        n = len(frames)
        pupil_raw = np.zeros(n, dtype=np.float64)
        rppg_raw = np.zeros(n, dtype=np.float64)
        rppg_rgb_raw = np.zeros((n, 3), dtype=np.float64)
        blink_mask = np.zeros(n, dtype=bool)
        missing_mask = np.zeros(n, dtype=bool)

        best_eye_crop = None
        best_conj_crop = None
        best_sclera_crop = None
        mp_hits = 0

        for idx, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                missing_mask[idx] = True
                continue

            mp_out = self._extract_with_mediapipe(frame)
            if mp_out is not None:
                mp_hits += 1
                pr, rg, rgb_v, eye_c, conj_c, scl_c, is_blink = mp_out
            else:
                pr, rg, rgb_v, eye_c, conj_c, scl_c, is_blink = self._extract_closeup_fallback(frame)

            pupil_raw[idx] = pr
            rppg_raw[idx] = rg
            rppg_rgb_raw[idx] = rgb_v
            blink_mask[idx] = is_blink

            if not is_blink and best_eye_crop is None:
                best_eye_crop = eye_c
                best_conj_crop = conj_c
                best_sclera_crop = scl_c

        if best_eye_crop is None:
            mid = frames[n // 2]
            best_eye_crop = mid
            h, w = mid.shape[:2]
            best_conj_crop = mid[int(h * 0.65) : int(h * 0.92), int(w * 0.2) : int(w * 0.8)]
            best_sclera_crop = mid[int(h * 0.30) : int(h * 0.60), int(w * 0.1) : int(w * 0.35)]

        invalid = blink_mask | missing_mask
        pupil_clean = self._interpolate_invalid(pupil_raw, invalid, default_val=0.40)
        rppg_clean = self._interpolate_invalid(rppg_raw, invalid, default_val=128.0)
        rppg_rgb_clean = np.column_stack(
            [self._interpolate_invalid(rppg_rgb_raw[:, c], invalid, default_val=128.0) for c in range(3)]
        )

        eye_resized = cv2.resize(best_eye_crop, self.roi_size, interpolation=cv2.INTER_CUBIC)
        eye_clahe = self.apply_clahe_bgr(eye_resized)

        timestamps = np.arange(n, dtype=np.float64) / fps
        method = "mediapipe_facemesh" if mp_hits > max(1, n // 4) else "closeup_ocular_roi"
        notes = []
        if np.any(blink_mask):
            notes.append(f"Interpolated {int(np.sum(blink_mask))} blink frame(s) cleanly.")
        if method != "mediapipe_facemesh":
            notes.append("Used close-up ocular ROI tracker (ideal for zoomed eye capture).")

        return VisionExtractionResult(
            fps=fps,
            timestamps=timestamps,
            pupil_ratio=pupil_clean,
            rppg_green=rppg_clean,
            rppg_rgb=rppg_rgb_clean,
            eye_roi_clahe=eye_clahe,
            eye_roi_raw=eye_resized,
            conjunctiva_roi=best_conj_crop,
            sclera_roi=best_sclera_crop,
            blink_mask=blink_mask,
            valid_frames=int(n - np.sum(invalid)),
            total_frames=n,
            tracking_method=method,
            notes=notes,
        )

    def process_video(self, video_path: str, max_frames: int = 600) -> VisionExtractionResult:
        """Read a video file with OpenCV and extract ocular signals."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1.0:
            fps = 30.0

        frames: List[np.ndarray] = []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("Video contained zero readable frames.")
        return self.process_frames(frames, fps=fps)

    def process_image(self, image_bgr: np.ndarray) -> VisionExtractionResult:
        """Process a single uploaded eye image by creating a 3-second static ROI buffer."""
        frames = [image_bgr.copy() for _ in range(90)]
        res = self.process_frames(frames, fps=30.0)
        res.notes.append("Static image mode: vascular & pallor screening active; temporal HRV/rPPG requires video.")
        return res
