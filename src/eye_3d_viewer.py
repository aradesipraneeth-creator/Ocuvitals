"""Interactive 3D Eye Digital Twin Visualization using Plotly 3D."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.models import SystemicRiskPrediction
from src.signal_processor import AutonomicMetrics
from src.vision_tracker import VisionExtractionResult


def _extract_iris_rgb(vision: VisionExtractionResult) -> tuple[int, int, int]:
    """Sample dominant iris RGB color from the central ring of the eye ROI."""
    img = vision.eye_roi_clahe
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_min, r_max = min(h, w) * 0.14, min(h, w) * 0.24
    ring = img[(dist >= r_min) & (dist <= r_max)]
    if ring.size == 0:
        return (72, 98, 132)
    b, g, r = np.median(ring, axis=0)
    return (int(np.clip(r, 30, 220)), int(np.clip(g, 30, 220)), int(np.clip(b, 30, 220)))


def build_3d_eye_digital_twin(
    vision: VisionExtractionResult,
    metrics: AutonomicMetrics,
    risk: SystemicRiskPrediction,
) -> go.Figure:
    """Build an interactive, rotatable 3D anatomical eyeball with diagnostic hover pins."""
    fig = go.Figure()

    # 1. Sclera Sphere (with subtle vascular/icterus tint from biomarkers)
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 36)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    # Color sclera based on icterus (yellowish) and vascular redness
    icterus = float(risk.lipid_icterus_prob)
    vasc = float(risk.vascular_strain_prob)
    r_scl = int(np.clip(244 - 10 * icterus + 8 * vasc, 200, 255))
    g_scl = int(np.clip(242 - 12 * vasc, 195, 255))
    b_scl = int(np.clip(242 - 58 * icterus - 20 * vasc, 150, 250))
    sclera_hex = f"rgb({r_scl},{g_scl},{b_scl})"

    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            surfacecolor=np.zeros_like(zs),
            colorscale=[[0, sclera_hex], [1, sclera_hex]],
            showscale=False,
            opacity=1.0,
            name="Sclera",
            hoverinfo="skip",
        )
    )

    # 2. Procedural 3D Scleral Micro-Vessels (density & redness scale with Vascular Strain)
    n_vessels = int(6 + round(vasc * 18))
    rng = np.random.default_rng(101)
    vessel_red = int(np.clip(175 + 75 * vasc, 160, 250))
    vessel_col = f"rgba({vessel_red}, 45, 55, 0.85)"
    for _ in range(n_vessels):
        phi0 = rng.uniform(0.55, 1.35)
        theta0 = rng.uniform(-np.pi, np.pi)
        t_curve = np.linspace(0, 1, 18)
        phi = phi0 + 0.35 * t_curve + 0.04 * np.sin(8 * t_curve)
        theta = theta0 + rng.uniform(-0.35, 0.35) * t_curve
        vx = 1.008 * np.cos(theta) * np.sin(phi)
        vy = 1.008 * np.sin(theta) * np.sin(phi)
        vz = 1.008 * np.cos(phi)
        fig.add_trace(
            go.Scatter3d(
                x=vx,
                y=vy,
                z=vz,
                mode="lines",
                line=dict(color=vessel_col, width=2.5 + 2.0 * vasc),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 3. Iris Disc & Dynamic Pupil Aperture (pupil expands with Stress Index)
    iris_r_val, iris_g_val, iris_b_val = _extract_iris_rgb(vision)
    iris_hex = f"rgb({iris_r_val},{iris_g_val},{iris_b_val})"

    # Stress-driven pupil radius (0.14 at low stress -> 0.28 at high stress)
    pupil_rad = 0.13 + 0.15 * float(np.clip(metrics.stress_index / 100.0, 0.0, 1.0))
    iris_outer = 0.44

    theta_ring = np.linspace(0, 2 * np.pi, 48)
    r_iris = np.linspace(pupil_rad, iris_outer, 10)
    rr, tt = np.meshgrid(r_iris, theta_ring)
    xi = rr * np.cos(tt)
    yi = rr * np.sin(tt)
    zi = np.sqrt(np.maximum(0.0, 1.01 - xi ** 2 - yi ** 2))

    fig.add_trace(
        go.Surface(
            x=xi,
            y=yi,
            z=zi,
            surfacecolor=rr,
            colorscale=[[0, iris_hex], [1, "rgb(35,45,65)"]],
            showscale=False,
            name="Iris",
            hoverinfo="skip",
        )
    )

    # Dynamic Pupil Aperture
    r_pup = np.linspace(0.0, pupil_rad, 8)
    rp, tp = np.meshgrid(r_pup, theta_ring)
    xp = rp * np.cos(tp)
    yp = rp * np.sin(tp)
    zp = np.sqrt(np.maximum(0.0, 1.012 - xp ** 2 - yp ** 2))
    fig.add_trace(
        go.Surface(
            x=xp,
            y=yp,
            z=zp,
            surfacecolor=np.zeros_like(zp),
            colorscale=[[0, "rgb(12,12,16)"], [1, "rgb(12,12,16)"]],
            showscale=False,
            name="Pupil",
            hoverinfo="skip",
        )
    )

    # 4. Translucent Corneal Dome (opacity=0.25) + Optional Lipid Arcus Ring
    phi_c = np.linspace(0, 0.48, 14)
    th_c, ph_c = np.meshgrid(theta_ring, phi_c)
    xc = 1.08 * np.cos(th_c) * np.sin(ph_c)
    yc = 1.08 * np.sin(th_c) * np.sin(ph_c)
    zc = 0.12 + 1.02 * np.cos(ph_c)
    fig.add_trace(
        go.Surface(
            x=xc,
            y=yc,
            z=zc,
            surfacecolor=np.zeros_like(zc),
            colorscale=[[0, "rgb(180,225,255)"], [1, "rgb(180,225,255)"]],
            showscale=False,
            opacity=0.25,
            name="Corneal Dome",
            hoverinfo="skip",
        )
    )

    if risk.lipid_icterus_prob >= 0.35:
        arc_r = iris_outer * 0.96
        xa = arc_r * np.cos(theta_ring)
        ya = arc_r * np.sin(theta_ring)
        za = np.sqrt(np.maximum(0.0, 1.02 - xa ** 2 - ya ** 2))
        fig.add_trace(
            go.Scatter3d(
                x=xa,
                y=ya,
                z=za,
                mode="lines",
                line=dict(color="rgba(250, 235, 175, 0.85)", width=5),
                name="Lipid Arcus Ring",
                hoverinfo="text",
                text=f"Peripheral Lipid/Icterus Ring (Risk: {risk.lipid_icterus_prob*100:.0f}%)",
            )
        )

    # 5. Interactive 3D Diagnostic Hover Pins
    pins = [
        (
            "Scleral Vessels",
            0.72,
            0.25,
            0.66,
            "#ef4444",
            f"<b>Scleral Micro-Vessels</b><br>Vascular Strain Risk: {risk.vascular_strain_prob*100:.0f}%<br>"
            f"Heart Rate: {metrics.heart_rate_bpm:.0f} BPM ({metrics.rhythm_flag})<br>"
            f"Vessel Density: {risk.physiological_features.get('vessel_density', 0)*100:.1f}%",
        ),
        (
            "Pupil Aperture",
            0.0,
            0.0,
            1.05,
            "#8b5cf6",
            f"<b>Dynamic Pupil Aperture</b><br>Autonomic Stress Index: {metrics.stress_index:.0f}/100 ({metrics.stress_level})<br>"
            f"RMSSD: {metrics.rmssd_ms:.1f} ms | Hippus Power: {metrics.hippus_power_ratio:.2f}",
        ),
        (
            "Lower Conjunctiva",
            0.0,
            -0.78,
            0.64,
            "#f59e0b",
            f"<b>Lower Palpebral Conjunctiva</b><br>Anemia / Pallor Risk: {risk.anemia_pallor_prob*100:.0f}%<br>"
            f"Erythema Index: {risk.physiological_features.get('erythema_index', 0):.1f} | "
            f"LAB a*: {risk.physiological_features.get('conj_a_star', 0):.1f}",
        ),
    ]

    fig.add_trace(
        go.Scatter3d(
            x=[p[1] for p in pins],
            y=[p[2] for p in pins],
            z=[p[3] for p in pins],
            mode="markers+text",
            text=[p[0] for p in pins],
            textposition="top center",
            hovertext=[p[5] for p in pins],
            hoverinfo="text",
            marker=dict(size=8, color=[p[4] for p in pins], line=dict(color="#ffffff", width=2)),
            name="Diagnostic Pins",
        )
    )

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-1.2, 1.2]),
            aspectmode="cube",
            camera=dict(eye=dict(x=0.35, y=-0.45, z=1.95)),
            bgcolor="rgba(15, 23, 42, 0.03)",
        ),
        showlegend=False,
    )
    return fig
