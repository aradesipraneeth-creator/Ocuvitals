"""Ocuviitals — Full Clinical Triage & 3D Digital Twin Dashboard.

Pipeline:
Input -> Strict Eye-Only Gatekeeper -> Pretrained 4-Stage Enhancement ->
Motion-Robust POS rPPG + SQI + PLR -> Pretrained Vision + BiLSTM Fusion ->
Interactive 3D Eye Twin + Vessel Skeleton + Baseline History + PDF Export.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.demo_generator import DEMO_PROFILES, generate_synthetic_eye_frames
from src.report_and_history import SUPPORTED_LANGUAGES


st.set_page_config(
    page_title="Ocuviitals | Zero-Hardware Ocular Triage",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0ea5e9 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(255,255,255,0.14);
    }
    .main-header h1 { margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
    .main-header p { margin: 0.35rem 0 0 0; color: #cbd5e1; font-size: 0.95rem; }
    .disclaimer-banner {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        color: #78350f;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .kpi-label { font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { font-size: 1.65rem; font-weight: 700; color: #0f172a; margin: 0.2rem 0; }
    .kpi-sub { font-size: 0.82rem; color: #475569; }
    .model-badge {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        font-size: 0.8rem;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_hub_and_model():
    from src.models import OcuviitalsFusionNet
    from src.pretrained_loader import get_model_registry

    registry = get_model_registry()
    model = OcuviitalsFusionNet(registry=registry, prefer_pretrained=True)
    return registry, model


def _finalize_pipeline(
    vision_res,
    enhanced_res,
    val_res,
    model,
    scan_label: str,
):
    import cv2
    from src.explainability import generate_gradcam_overlay, generate_vessel_skeleton_overlay
    from src.eye_3d_viewer import build_3d_eye_digital_twin
    from src.models import predict_systemic_risks
    from src.report_and_history import HistoryStore, generate_pdf_triage_report
    from src.signal_processor import analyze_signals

    vision_res.eye_roi_clahe = cv2.resize(enhanced_res.enhanced_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    metrics = analyze_signals(
        vision_res.pupil_ratio,
        vision_res.rppg_green,
        fps=vision_res.fps,
        rppg_rgb=vision_res.rppg_rgb,
    )
    risk = predict_systemic_risks(model, vision_res, metrics)
    gradcam_rgb = generate_gradcam_overlay(model, vision_res, metrics, target_class=0)
    vessel_rgb = generate_vessel_skeleton_overlay(enhanced_res.enhanced_bgr)
    twin_fig = build_3d_eye_digital_twin(vision_res, metrics, risk)

    store = HistoryStore("assets/scan_history.json")
    history = store.append_scan(scan_label, metrics, risk)
    deltas = store.compute_baseline_deltas(history, history[-1])
    pdf_bytes = generate_pdf_triage_report(
        enhanced_res.raw_crop_rgb,
        enhanced_res.enhanced_rgb,
        vessel_rgb,
        metrics,
        risk,
    )

    return {
        "rejected": False,
        "validation": val_res,
        "enhanced": enhanced_res,
        "vision": vision_res,
        "metrics": metrics,
        "risk": risk,
        "gradcam_rgb": gradcam_rgb,
        "vessel_rgb": vessel_rgb,
        "twin_fig": twin_fig,
        "history": history,
        "deltas": deltas,
        "pdf_bytes": pdf_bytes,
    }


def run_gated_pipeline_on_frames(frames: list[np.ndarray], fps: float, scan_label: str = "Ocular Scan"):
    from src.eye_validator import validate_eye_input
    from src.image_enhancer import enhance_eye_image
    from src.vision_tracker import EyeVisionTracker

    registry, model = load_hub_and_model()
    val_res = validate_eye_input(frames, registry=registry)
    if not val_res.is_valid:
        return {"rejected": True, "validation": val_res}

    mid_frame = frames[len(frames) // 2]
    enhanced_res = enhance_eye_image(mid_frame, eye_bbox=val_res.eye_bbox, registry=registry)

    tracker = EyeVisionTracker()
    try:
        vision_res = tracker.process_frames(frames, fps=fps)
    finally:
        tracker.close()

    return _finalize_pipeline(vision_res, enhanced_res, val_res, model, scan_label)


def run_gated_pipeline_on_image(image_bgr: np.ndarray, scan_label: str = "Uploaded Eye Image"):
    from src.eye_validator import validate_eye_input
    from src.image_enhancer import enhance_eye_image
    from src.vision_tracker import EyeVisionTracker

    registry, model = load_hub_and_model()
    val_res = validate_eye_input(image_bgr, registry=registry)
    if not val_res.is_valid:
        return {"rejected": True, "validation": val_res}

    enhanced_res = enhance_eye_image(image_bgr, eye_bbox=val_res.eye_bbox, registry=registry)
    tracker = EyeVisionTracker()
    try:
        vision_res = tracker.process_image(image_bgr)
    finally:
        tracker.close()

    return _finalize_pipeline(vision_res, enhanced_res, val_res, model, scan_label)


def capture_webcam_10s(duration_sec: int = 10, target_fps: float = 20.0, flash_plr: bool = True):
    """Capture a 10-second eye scan from a local webcam (with DirectShow fallback on Windows)."""
    import cv2

    cap = None
    backend_candidates = (
        [(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY), (1, cv2.CAP_DSHOW), (1, cv2.CAP_ANY)]
        if os.name == "nt"
        else [(0, cv2.CAP_ANY), (1, cv2.CAP_ANY)]
    )
    for cam_idx, api_pref in backend_candidates:
        try:
            candidate = cv2.VideoCapture(cam_idx, api_pref)
            if candidate is not None and candidate.isOpened():
                ok_test, _ = candidate.read()
                if ok_test:
                    cap = candidate
                    break
                candidate.release()
        except Exception:
            pass

    if cap is None:
        return (
            None,
            0.0,
            "No server-side webcam accessible (normal on cloud servers like Render, or if another app is using the camera). "
            "Use the **Browser Live Camera** tab above—it accesses your device camera directly!",
        )

    frames = []
    preview_slot = st.empty()
    flash_slot = st.empty()
    progress_bar = st.progress(0.0, text="Initializing 10-second ocular scan...")
    start = time.time()
    frame_interval = 1.0 / target_fps

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration_sec:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(frame)

            if flash_plr and 4.0 <= elapsed <= 5.5:
                flash_slot.markdown(
                    '<div style="background:#ffffff;height:180px;border-radius:12px;border:3px solid #38bdf8;'
                    'display:flex;align-items:center;justify-content:center;font-weight:700;color:#0f172a;">'
                    '☀️ LIGHT STIMULUS ACTIVE — KEEP LOOKING AT CAMERA</div>',
                    unsafe_allow_html=True,
                )
            else:
                flash_slot.empty()

            if len(frames) % 3 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(rgb, caption=f"Live Scan ({elapsed:.1f}s / {duration_sec}s)", width=360)
            progress_bar.progress(min(1.0, elapsed / duration_sec), text=f"Recording POS-rPPG & Pupil Reflex... {elapsed:.1f}s / {duration_sec}s")
            time.sleep(max(0.0, frame_interval - 0.005))
    finally:
        cap.release()
        progress_bar.empty()
        preview_slot.empty()
        flash_slot.empty()

    if len(frames) < 15:
        return None, 0.0, "Insufficient frames captured from webcam."
    actual_fps = len(frames) / max(time.time() - start, 1.0)
    return frames, actual_fps, ""


def render_results(res: dict, language: str):
    import plotly.graph_objects as go
    from src.eye_validator import REJECTION_MESSAGE
    from src.report_and_history import build_multilingual_summary

    if res.get("rejected"):
        st.error(f"🚫 {REJECTION_MESSAGE}")
        st.caption(f"Gatekeeper confidence: {res['validation'].confidence:.2f} | Downstream health metrics and 3D rendering blocked.")
        return

    val_res = res["validation"]
    enhanced = res["enhanced"]
    vision_res = res["vision"]
    metrics = res["metrics"]
    risk = res["risk"]
    gradcam_rgb = res["gradcam_rgb"]
    vessel_rgb = res["vessel_rgb"]
    twin_fig = res["twin_fig"]
    history = res["history"]
    deltas = res["deltas"]

    col_ok, col_pdf = st.columns([3, 1])
    with col_ok:
        st.success(
            f"✅ Eye Gatekeeper Passed ({val_res.method} · Conf: {val_res.confidence:.2f}) | "
            f"Signal Confidence: **{risk.signal_confidence_pct:.0f}%** (POS-rPPG SQI: {metrics.sqi_score:.0f}%)"
        )
    with col_pdf:
        st.download_button(
            label="📄 Download Triage Report (PDF)",
            data=res["pdf_bytes"],
            file_name="Ocuviitals_Clinical_Triage_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # KPI Summary Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Autonomic Stress &amp; PLR</div>
                <div class="kpi-value">{metrics.stress_index:.0f} / 100 ({metrics.stress_level})</div>
                <div class="kpi-sub">Baseline Δ: {deltas['stress_delta']:+.1f} · PLR Constriction: {metrics.plr_constriction_pct:.1f}% ({metrics.plr_recovery_sec:.2f}s rec)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Heart Rate &amp; Rhythm (POS-rPPG)</div>
                <div class="kpi-value">{metrics.heart_rate_bpm:.0f} BPM · {metrics.rhythm_flag}</div>
                <div class="kpi-sub">Baseline Δ: {deltas['hr_delta']:+.1f} BPM · RMSSD: {metrics.rmssd_ms:.1f} ms · SDNN: {metrics.sdnn_ms:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Overall Triage Priority</div>
                <div class="kpi-value" style="color:{risk.triage_color};">{risk.triage_priority}</div>
                <div class="kpi-sub">Signal Confidence: {risk.signal_confidence_pct:.0f}% · Frames: {vision_res.valid_frames}/{vision_res.total_frames}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Multilingual Accessibility Triage Summary + Voice Readout
    summary_text = build_multilingual_summary(metrics, risk, lang=language)
    with st.expander(f"🗣️ Primary-Care Triage Summary ({language})", expanded=True):
        st.write(summary_text)
        if st.button("🔊 Read Triage Summary Aloud"):
            escaped = json.dumps(summary_text)
            tts_lang = json.dumps(SUPPORTED_LANGUAGES.get(language, "en-IN"))
            components.html(
                f"""
                <script>
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance({escaped});
                u.lang = {tts_lang};
                window.speechSynthesis.speak(u);
                </script>
                """,
                height=0,
            )

    # Interactive 3D Eye Digital Twin + 3-Panel Clinical Imaging
    st.markdown("### Interactive 3D Eye Digital Twin & Vascular Explainability")
    col_3d, col_enh = st.columns([1.2, 1.05])

    with col_3d:
        st.markdown("**Rotatable 3D Ocular Digital Twin (Hover Pins for Biomarkers)**")
        st.plotly_chart(twin_fig, use_container_width=True)

    with col_enh:
        st.markdown("**Raw Upload vs. Pretrained AI-Enhanced vs. Vessel Skeleton**")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.image(enhanced.raw_crop_rgb, caption="1. Raw Crop", use_container_width=True)
        with e2:
            st.image(enhanced.enhanced_rgb, caption="2. Enhanced 512²", use_container_width=True)
        with e3:
            st.image(vessel_rgb, caption="3. Vessel Skeleton", use_container_width=True)
        st.image(gradcam_rgb, caption="Pretrained Swin-Tiny Saliency + Vascular/Pallor Attention", use_container_width=True)

    # Systemic Disease Risk Flags
    st.markdown("### Systemic Disease Risk Flags (Scleral-Calibrated Multimodal Fusion)")
    r1, r2, r3 = st.columns(3)
    flags = [
        ("Anemia / Conjunctival Pallor", risk.anemia_pallor_prob, risk.rationale["anemia"], r1),
        ("Vascular / Hypertension Strain", risk.vascular_strain_prob, risk.rationale["vascular"], r2),
        ("Lipid / Icterus Indicators", risk.lipid_icterus_prob, risk.rationale["lipid"], r3),
    ]
    for title, prob, desc, col in flags:
        with col:
            pct = int(round(prob * 100))
            badge = "🔴 Elevated" if pct >= 60 else ("🟡 Borderline" if pct >= 38 else "🟢 Low")
            st.metric(label=title, value=f"{pct}%", delta=badge, delta_color="off")
            st.progress(min(1.0, max(0.0, prob)))
            st.caption(desc)

    # Temporal Signal Charts + Longitudinal Baseline Trend
    st.markdown("### Autonomic Signals & Personal Baseline Trend")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("**POS-rPPG Cardiac Pulse**")
        fig_pulse = go.Figure()
        t_axis = vision_res.timestamps
        fig_pulse.add_trace(go.Scatter(x=t_axis, y=metrics.filtered_rppg, mode="lines", name="POS-rPPG", line=dict(color="#0284c7", width=2)))
        if len(metrics.peak_indices) > 0:
            fig_pulse.add_trace(
                go.Scatter(x=t_axis[metrics.peak_indices], y=metrics.filtered_rppg[metrics.peak_indices], mode="markers", name="Beats", marker=dict(color="#dc2626", size=7))
            )
        fig_pulse.update_layout(height=270, margin=dict(l=10, r=10, t=15, b=25), xaxis_title="Time (s)")
        st.plotly_chart(fig_pulse, use_container_width=True)

    with v2:
        st.markdown("**Pupil Dilation & PLR Trace**")
        fig_pupil = go.Figure()
        fig_pupil.add_trace(
            go.Scatter(x=vision_res.timestamps, y=vision_res.pupil_ratio, mode="lines", name="Pupil/Iris", line=dict(color="#7c3aed", width=2), fill="tozeroy", fillcolor="rgba(124, 58, 237, 0.10)")
        )
        fig_pupil.update_layout(height=270, margin=dict(l=10, r=10, t=15, b=25), xaxis_title="Time (s)")
        st.plotly_chart(fig_pupil, use_container_width=True)

    with v3:
        st.markdown("**Longitudinal Baseline History**")
        fig_hist = go.Figure()
        x_idx = list(range(1, len(history) + 1))
        fig_hist.add_trace(go.Scatter(x=x_idx, y=[h["heart_rate_bpm"] for h in history], mode="lines+markers", name="HR (BPM)", line=dict(color="#ef4444", width=2)))
        fig_hist.add_trace(go.Scatter(x=x_idx, y=[h["stress_index"] for h in history], mode="lines+markers", name="Stress", line=dict(color="#f59e0b", width=2)))
        fig_hist.update_layout(height=270, margin=dict(l=10, r=10, t=15, b=25), xaxis_title="Scan #")
        st.plotly_chart(fig_hist, use_container_width=True)


def main():
    is_lite = bool(os.environ.get("RENDER") or os.environ.get("OCUVIITALS_LITE") == "1")

    st.markdown(
        """
        <div class="main-header">
            <h1>👁️ Ocuviitals — Zero-Hardware Ocular Health Triage</h1>
            <p>Strict Eye Gatekeeper · Pretrained Vision &amp; Super-Resolution · POS-rPPG &amp; PLR · Interactive 3D Ocular Digital Twin</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            <b>⚠️ Medical Disclaimer:</b> Ocuviitals is an early-warning wellness and triage screening prototype
            for research and hackathon demonstration purposes. It is <b>not</b> a standalone clinical diagnostic device
            and does not replace laboratory blood panels, ECGs, or physician evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Scan Input Mode")
        mode = st.radio(
            "Choose Input Source",
            [
                "Instant Clinical Sample Demo",
                "Upload Eye Video / Image",
                "Live Webcam 10-Second Eye Scan",
            ],
        )
        language = st.selectbox("Summary Language", list(SUPPORTED_LANGUAGES.keys()))
        st.divider()
        st.markdown("**Active Pretrained Models**")
        if "model_badges" in st.session_state:
            badge_lines = "<br>".join(f"• <b>{k}:</b> {v}" for k, v in st.session_state["model_badges"].items())
            st.markdown(f'<div class="model-badge">{badge_lines}</div>', unsafe_allow_html=True)
        else:
            mode_label = "MobileNetV3-Small (Render Lite)" if is_lite else "Swin-Tiny (timm)"
            st.markdown(
                f'<div class="model-badge">• <b>Landmarks:</b> MediaPipe FaceLandmarker v2<br>'
                f'• <b>Eye Cascade:</b> OpenCV Haar Cascade<br>'
                f'• <b>Backbone:</b> {mode_label}<br>'
                f'• <b>Status:</b> Ready (Lazy-loaded on scan)</div>',
                unsafe_allow_html=True,
            )

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if mode == "Instant Clinical Sample Demo":
        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            profile_name = st.selectbox("Select Synthetic Clinical Profile", list(DEMO_PROFILES.keys()))
            st.caption(DEMO_PROFILES[profile_name].description)
        with col_btn:
            st.write("")
            run_demo = st.button("🚀 Run Instant Clinical Sample Demo", type="primary", use_container_width=True)

        if run_demo:
            with st.spinner(f"Validating, enhancing & modeling '{profile_name}'..."):
                fps_demo = 24.0 if is_lite else 30.0
                dur_demo = 8.0 if is_lite else 10.0
                frames = generate_synthetic_eye_frames(profile_name=profile_name, duration_sec=dur_demo, fps=fps_demo)
                st.session_state["last_result"] = run_gated_pipeline_on_frames(frames, fps=fps_demo, scan_label=profile_name)
                from src.pretrained_loader import get_model_registry
                st.session_state["model_badges"] = get_model_registry().status_badges
                del frames
                gc.collect()

    elif mode == "Upload Eye Video / Image":
        uploaded = st.file_uploader(
            "Upload an eye/face video (.mp4, .avi, .mov) or close-up eye image (.png, .jpg, .jpeg)",
            type=["mp4", "avi", "mov", "png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg"}:
                import cv2
                from PIL import Image
                img = Image.open(uploaded).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if st.button("🔬 Validate & Analyze Eye Image", type="primary"):
                    with st.spinner("Running Eye Gatekeeper, Pretrained Enhancement & 3D Twin..."):
                        st.session_state["last_result"] = run_gated_pipeline_on_image(img_bgr, scan_label=uploaded.name)
            else:
                if st.button("🔬 Validate & Analyze Eye Video", type="primary"):
                    with st.spinner("Running Eye Gatekeeper, POS-rPPG extraction & 3D Twin..."):
                        import cv2
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frames = []
                        while len(frames) < 450:
                            ok, f = cap.read()
                            if not ok:
                                break
                            frames.append(f)
                        cap.release()
                        st.session_state["last_result"] = run_gated_pipeline_on_frames(frames, fps=float(fps), scan_label=uploaded.name)

    elif mode == "Live Webcam 10-Second Eye Scan":
        tab_browser, tab_local = st.tabs([
            "🌐 Browser Live Camera (Works on Render & Local)",
            "🎥 Local 10-Second Video Scan (rPPG + PLR)",
        ])

        with tab_browser:
            st.caption("Uses your browser's camera directly (works on mobile, laptop, and Render cloud). Align one eye close to the lens and capture.")
            cam_photo = st.camera_input("📷 Capture Live Eye Photo")
            if cam_photo is not None:
                import cv2
                from PIL import Image
                img = Image.open(cam_photo).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if st.button("🔬 Validate & Analyze Live Eye Capture", type="primary"):
                    with st.spinner("Running Eye Gatekeeper, Pretrained Enhancement & 3D Twin..."):
                        st.session_state["last_result"] = run_gated_pipeline_on_image(img_bgr, scan_label="Browser Live Eye Capture")
                        from src.pretrained_loader import get_model_registry
                        st.session_state["model_badges"] = get_model_registry().status_badges

        with tab_local:
            st.caption("Captures a 10-second video stream from a locally attached webcam for full POS-rPPG + HRV + Screen-Flash PLR.")
            flash_plr = st.checkbox("☀️ Enable Screen-Flash Pupillary Light Reflex (PLR) Stimulus at 4s", value=True)
            if st.button("📷 Start Local 10-Second Webcam Scan", type="primary"):
                frames, actual_fps, err = capture_webcam_10s(duration_sec=10, target_fps=20.0, flash_plr=flash_plr)
                if err:
                    st.warning(err)
                elif frames is not None:
                    with st.spinner("Running Eye Gatekeeper, POS-rPPG & 3D Twin..."):
                        st.session_state["last_result"] = run_gated_pipeline_on_frames(frames, fps=actual_fps, scan_label="Webcam Scan")
                        from src.pretrained_loader import get_model_registry
                        st.session_state["model_badges"] = get_model_registry().status_badges

    if st.session_state["last_result"] is not None:
        st.divider()
        render_results(st.session_state["last_result"], language=language)
    else:
        st.info("Select an input mode above and click the action button to run Ocuviitals.")


if __name__ == "__main__":
    main()
