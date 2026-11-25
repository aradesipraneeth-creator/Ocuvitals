"""End-to-end verification script for Ocuviitals."""

from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_generator import DEMO_PROFILES, ensure_sample_asset, generate_synthetic_eye_frames
from src.explainability import generate_gradcam_overlay, generate_vessel_skeleton_overlay
from src.eye_3d_viewer import build_3d_eye_digital_twin
from src.eye_validator import REJECTION_MESSAGE, validate_eye_input
from src.image_enhancer import enhance_eye_image
from src.models import OcuviitalsFusionNet, predict_systemic_risks
from src.pretrained_loader import get_model_registry
from src.report_and_history import HistoryStore, build_multilingual_summary, generate_pdf_triage_report
from src.signal_processor import analyze_signals
from src.vision_tracker import EyeVisionTracker


def main() -> None:
    print("=== Ocuviitals End-to-End Self-Test ===")

    # 1. Pretrained Foundation & Clinical Model Hub
    reg = get_model_registry()
    print("[OK] Active Pretrained Models:")
    for k, v in reg.status_badges.items():
        print(f"     - {k}: {v}")

    model = OcuviitalsFusionNet(registry=reg, prefer_pretrained=True)

    # 2. Strict Eye-Only Gatekeeper: Non-Eye Rejection Tests
    non_eye_cases = {
        "Blank Wall": np.full((240, 320, 3), 210, dtype=np.uint8),
        "Random Noise": np.random.default_rng(7).integers(0, 255, (240, 320, 3), dtype=np.uint8),
        "Document Grid": np.full((240, 320, 3), 245, dtype=np.uint8),
        "Landscape Gradient": np.tile(np.linspace(40, 220, 240, dtype=np.uint8)[:, None, None], (1, 320, 3)),
    }
    # Add text-like horizontal bars to Document Grid
    for y in range(20, 220, 18):
        cv2.line(non_eye_cases["Document Grid"], (20, y), (300, y), (30, 30, 30), 2)

    for label, img in non_eye_cases.items():
        vres = validate_eye_input(img, registry=reg)
        assert not vres.is_valid, f"Expected rejection for {label}, got valid={vres.is_valid} ({vres.method})"
        assert vres.message == REJECTION_MESSAGE
        print(f"[OK] Gatekeeper rejected non-eye '{label}' (conf={vres.confidence:.2f})")

    # 3. Valid Eye Pipeline: Gatekeeper -> Enhancement -> Vision/Signal -> Fusion -> 3D Twin
    sample_path = ensure_sample_asset(ROOT / "assets", "Healthy Baseline")
    assert sample_path.exists()

    tracker = EyeVisionTracker()
    try:
        for name in DEMO_PROFILES:
            frames = generate_synthetic_eye_frames(profile_name=name, duration_sec=8.0, fps=30.0)
            val = validate_eye_input(frames, registry=reg)
            assert val.is_valid and val.confidence > 0.75, f"Gatekeeper failed on '{name}': {val}"

            enh = enhance_eye_image(frames[len(frames) // 2], eye_bbox=val.eye_bbox, registry=reg)
            assert enh.raw_crop_rgb.shape == (512, 512, 3)
            assert enh.enhanced_rgb.shape == (512, 512, 3)

            vis = tracker.process_frames(frames, fps=30.0)
            vis.eye_roi_clahe = cv2.resize(enh.enhanced_bgr, (224, 224), interpolation=cv2.INTER_AREA)
            met = analyze_signals(vis.pupil_ratio, vis.rppg_green, fps=vis.fps, rppg_rgb=vis.rppg_rgb)
            risk = predict_systemic_risks(model, vis, met)
            cam = generate_gradcam_overlay(model, vis, met)
            vessel_rgb = generate_vessel_skeleton_overlay(enh.enhanced_bgr)
            twin_fig = build_3d_eye_digital_twin(vis, met, risk)

            assert cam.shape == (224, 224, 3)
            assert vessel_rgb.shape == (512, 512, 3)
            assert len(twin_fig.data) >= 6, "Expected 3D eyeball, vessels, iris, pupil, cornea, and hover pins"

            pdf_bytes = generate_pdf_triage_report(enh.raw_crop_rgb, enh.enhanced_rgb, vessel_rgb, met, risk)
            assert pdf_bytes.startswith(b"%PDF"), "Invalid PDF generated"
            for lang in ["English", "हिन्दी (Hindi)", "Español (Spanish)"]:
                assert len(build_multilingual_summary(met, risk, lang=lang)) > 30

            store = HistoryStore(ROOT / "assets" / "test_history.json")
            hist = store.append_scan(name, met, risk)
            deltas = store.compute_baseline_deltas(hist, hist[-1])

            print(
                f"[OK] Profile '{name}': Gatekeeper={val.confidence:.2f} | SQI={met.sqi_score:.0f}% | "
                f"PLR={met.plr_constriction_pct:.1f}% | HR={met.heart_rate_bpm} BPM ({met.rhythm_flag}) | "
                f"Stress={met.stress_index} | DeltaHR={deltas['hr_delta']:+.1f} | Priority={risk.triage_priority}"
            )
    finally:
        tracker.close()

    print("=== ALL UPGRADED SELF-TESTS PASSED ===")


if __name__ == "__main__":
    main()
