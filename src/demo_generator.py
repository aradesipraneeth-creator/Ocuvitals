"""Synthetic Realistic Eye Video & Signal Generator for OptiScan AI.

Generates realistic synthetic ocular videos (with sclera, iris, dynamic pupil
hippus/PLR, lower palpebral conjunctiva, blood vessels, natural blinks, and
green-channel cardiac rPPG pulse modulation) for instant offline testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np


@dataclass(frozen=True)
class ClinicalDemoProfile:
    name: str
    description: str
    heart_rate_bpm: float
    rhythm_irregularity: float  # 0.0 = regular, 0.25 = irregular
    pupil_base_ratio: float
    pupil_hippus_amp: float
    conjunctiva_bgr: tuple  # (B, G, R)
    sclera_bgr: tuple       # (B, G, R)
    vessel_count: int


DEMO_PROFILES: Dict[str, ClinicalDemoProfile] = {
    "Healthy Baseline": ClinicalDemoProfile(
        name="Healthy Baseline",
        description="Normal 70 BPM regular sinus rhythm, low autonomic stress, healthy pink conjunctiva, clear sclera.",
        heart_rate_bpm=70.0,
        rhythm_irregularity=0.02,
        pupil_base_ratio=0.38,
        pupil_hippus_amp=0.015,
        conjunctiva_bgr=(125, 118, 212),  # Healthy vascular pink-red
        sclera_bgr=(238, 240, 244),       # Clear white sclera
        vessel_count=4,
    ),
    "High Stress & Irregular Rhythm": ClinicalDemoProfile(
        name="High Stress & Irregular Rhythm",
        description="Elevated 104 BPM with beat-to-beat RR irregularity, pronounced pupil hippus, and mild vascular strain.",
        heart_rate_bpm=104.0,
        rhythm_irregularity=0.22,
        pupil_base_ratio=0.52,
        pupil_hippus_amp=0.065,
        conjunctiva_bgr=(132, 126, 204),
        sclera_bgr=(230, 232, 242),
        vessel_count=12,
    ),
    "Anemia / Conjunctival Pallor Flag": ClinicalDemoProfile(
        name="Anemia / Conjunctival Pallor Flag",
        description="Pale lower palpebral conjunctiva (reduced erythema) with compensatory mild tachycardia (92 BPM).",
        heart_rate_bpm=92.0,
        rhythm_irregularity=0.04,
        pupil_base_ratio=0.43,
        pupil_hippus_amp=0.028,
        conjunctiva_bgr=(182, 184, 202),  # Pale / washed-out conjunctiva
        sclera_bgr=(240, 242, 245),
        vessel_count=3,
    ),
    "Vascular & Lipid/Icterus Strain": ClinicalDemoProfile(
        name="Vascular & Lipid/Icterus Strain",
        description="Prominent scleral micro-vessels, subtle yellow-lipid scleral tint, and elevated autonomic strain.",
        heart_rate_bpm=88.0,
        rhythm_irregularity=0.08,
        pupil_base_ratio=0.46,
        pupil_hippus_amp=0.040,
        conjunctiva_bgr=(130, 120, 210),
        sclera_bgr=(188, 224, 236),       # Subtle yellowish icterus tint
        vessel_count=18,
    ),
}


def generate_synthetic_eye_frames(
    profile_name: str = "Healthy Baseline",
    duration_sec: float = 10.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate a list of realistic BGR eye frames for the chosen clinical profile."""
    profile = DEMO_PROFILES.get(profile_name, DEMO_PROFILES["Healthy Baseline"])
    rng = np.random.default_rng(seed)

    n_frames = int(duration_sec * fps)
    t = np.arange(n_frames, dtype=np.float64) / fps

    # Generate cardiac phase with optional RR irregularity (HRV / arrhythmia)
    base_freq = profile.heart_rate_bpm / 60.0
    phase = np.zeros(n_frames, dtype=np.float64)
    inst_freq = base_freq + profile.rhythm_irregularity * base_freq * (
        0.7 * np.sin(2 * np.pi * 0.35 * t) + 0.5 * np.sin(2 * np.pi * 0.85 * t)
    )
    for i in range(1, n_frames):
        phase[i] = phase[i - 1] + 2.0 * np.pi * inst_freq[i] / fps
    # Asymmetric PPG-like cardiac pulse (systolic peak + dicrotic notch)
    pulse_wave = 0.65 * np.sin(phase) + 0.35 * np.sin(2.0 * phase - 0.6)

    # Pupil ratio trajectory (hippus + optional Pupillary Light Reflex flash dip at t=4.0s)
    plr_dip = 0.08 * np.exp(-((t - 4.4) ** 2) / (2 * 0.45 ** 2))
    pupil_series = (
        profile.pupil_base_ratio
        + profile.pupil_hippus_amp * (0.65 * np.sin(2 * np.pi * 0.25 * t) + 0.35 * np.cos(2 * np.pi * 0.42 * t))
        - plr_dip
    )
    pupil_series = np.clip(pupil_series, 0.22, 0.68)

    # Pre-generate static vessel coordinates on sclera
    vessels = []
    cx, cy = width // 2, int(height * 0.45)
    iris_r = int(min(width, height) * 0.24)
    for _ in range(profile.vessel_count):
        side = -1 if rng.random() < 0.5 else 1
        y0 = int(cy + rng.integers(-28, 28))
        x0 = int(cx + side * rng.integers(iris_r + 6, iris_r + 48))
        x1 = int(x0 + side * rng.integers(14, 42))
        y1 = int(y0 + rng.integers(-16, 16))
        vessels.append(((x0, y0), (x1, y1)))

    # Blink windows (2 brief blinks in 10s)
    blink_frames = set()
    for b_center in [int(n_frames * 0.32), int(n_frames * 0.74)]:
        for k in range(b_center - 2, b_center + 3):
            if 0 <= k < n_frames:
                blink_frames.add(k)

    # Base skin / periocular background
    frames: List[np.ndarray] = []
    yy, xx = np.mgrid[0:height, 0:width]

    for i in range(n_frames):
        # Skin background
        frame = np.full((height, width, 3), (156, 174, 212), dtype=np.float32)

        if i in blink_frames:
            # Closed eyelid frame
            cv2.ellipse(frame, (cx, cy), (int(width * 0.40), int(height * 0.28)), 0, 0, 360, (148, 165, 205), -1)
            cv2.line(frame, (int(cx - width * 0.36), cy + 6), (int(cx + width * 0.36), cy + 6), (95, 105, 130), 2)
            noise = rng.normal(0, 1.5, frame.shape).astype(np.float32)
            frames.append(np.clip(frame + noise, 0, 255).astype(np.uint8))
            continue

        # Subtle green-channel rPPG pulse modulation on sclera & conjunctiva
        g_mod = float(4.2 * pulse_wave[i])
        scl_col = (
            profile.sclera_bgr[0],
            np.clip(profile.sclera_bgr[1] + g_mod, 0, 255),
            profile.sclera_bgr[2],
        )
        conj_col = (
            profile.conjunctiva_bgr[0],
            np.clip(profile.conjunctiva_bgr[1] + 0.8 * g_mod, 0, 255),
            profile.conjunctiva_bgr[2],
        )

        # Sclera ellipse
        cv2.ellipse(frame, (cx, cy), (int(width * 0.38), int(height * 0.24)), 0, 0, 360, scl_col, -1)

        # Scleral blood vessels
        for pt1, pt2 in vessels:
            cv2.line(frame, pt1, pt2, (78, 72, 185), 1, lineType=cv2.LINE_AA)

        # Iris & Pupil
        cv2.circle(frame, (cx, cy), iris_r, (74, 96, 128), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(iris_r * 0.88), (62, 82, 114), 2, lineType=cv2.LINE_AA)
        pupil_r = max(6, int(iris_r * pupil_series[i]))
        cv2.circle(frame, (cx, cy), pupil_r, (18, 18, 22), -1, lineType=cv2.LINE_AA)
        # Corneal specular highlight
        cv2.circle(frame, (cx - int(iris_r * 0.32), cy - int(iris_r * 0.32)), max(3, iris_r // 6), (250, 252, 255), -1)

        # Lower palpebral conjunctiva band
        cv2.ellipse(
            frame,
            (cx, int(height * 0.72)),
            (int(width * 0.32), int(height * 0.12)),
            0,
            0,
            180,
            conj_col,
            -1,
        )

        # Soft vignette & sensor noise
        noise = rng.normal(0, 1.2, frame.shape).astype(np.float32)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


def ensure_sample_asset(
    output_dir: str | Path = "assets",
    profile_name: str = "Healthy Baseline",
) -> Path:
    """Create a sample synthetic eye video in assets/ if not already present."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_slug = profile_name.lower().replace(" ", "_").replace("/", "_")
    video_path = out_path / f"sample_{safe_slug}.avi"

    if not video_path.exists():
        frames = generate_synthetic_eye_frames(profile_name=profile_name, duration_sec=8.0, fps=30.0)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    return video_path
