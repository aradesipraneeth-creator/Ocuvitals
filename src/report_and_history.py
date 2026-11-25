"""Longitudinal Baseline History, Multilingual Accessibility & PDF Triage Report."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.models import SystemicRiskPrediction
from src.signal_processor import AutonomicMetrics


class HistoryStore:
    """Persists scan history locally to compute personal baseline deltas."""

    def __init__(self, path: str | Path = "assets/scan_history.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def append_scan(
        self,
        label: str,
        metrics: AutonomicMetrics,
        risk: SystemicRiskPrediction,
    ) -> List[Dict]:
        history = self.load()
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "heart_rate_bpm": float(metrics.heart_rate_bpm),
            "stress_index": float(metrics.stress_index),
            "rmssd_ms": float(metrics.rmssd_ms),
            "sqi_score": float(metrics.sqi_score),
            "anemia_pct": round(risk.anemia_pallor_prob * 100, 1),
            "vascular_pct": round(risk.vascular_strain_prob * 100, 1),
            "lipid_pct": round(risk.lipid_icterus_prob * 100, 1),
            "priority": risk.triage_priority,
        }
        history.append(entry)
        history = history[-30:]  # Keep last 30 scans
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history

    @staticmethod
    def compute_baseline_deltas(history: List[Dict], current: Dict) -> Dict[str, float]:
        if len(history) < 2:
            return {"hr_delta": 0.0, "stress_delta": 0.0}
        prev = history[:-1]
        avg_hr = float(np.mean([h["heart_rate_bpm"] for h in prev]))
        avg_stress = float(np.mean([h["stress_index"] for h in prev]))
        return {
            "hr_delta": round(current["heart_rate_bpm"] - avg_hr, 1),
            "stress_delta": round(current["stress_index"] - avg_stress, 1),
        }


def build_multilingual_summary(
    metrics: AutonomicMetrics,
    risk: SystemicRiskPrediction,
    lang: str = "English",
) -> str:
    """Return a clear, primary-care-friendly triage summary in English, Hindi, or Spanish."""
    anemia = int(round(risk.anemia_pallor_prob * 100))
    vasc = int(round(risk.vascular_strain_prob * 100))
    lipid = int(round(risk.lipid_icterus_prob * 100))

    if lang == "हिन्दी (Hindi)":
        return (
            f"OcuVitals स्क्रीनिंग सारांश: प्राथमिकता स्तर {risk.triage_priority} है। "
            f"हृदय गति {metrics.heart_rate_bpm:.0f} BPM ({metrics.rhythm_flag}), "
            f"और तनाव सूचकांक {metrics.stress_index:.0f}/100 ({metrics.stress_level}) है। "
            f"एनीमिया (पीलापन) जोखिम {anemia}%, वैस्कुलर तनाव {vasc}%, और लिपिड/पीलिया संकेत {lipid}% हैं। "
            f"यह केवल प्रारंभिक स्क्रीनिंग है, डॉक्टर की सलाह अवश्य लें।"
        )
    if lang == "Español (Spanish)":
        return (
            f"Resumen de triaje OcuVitals: Prioridad {risk.triage_priority}. "
            f"Frecuencia cardíaca {metrics.heart_rate_bpm:.0f} BPM ({metrics.rhythm_flag}), "
            f"Índice de estrés {metrics.stress_index:.0f}/100 ({metrics.stress_level}). "
            f"Riesgo de anemia {anemia}%, tensión vascular {vasc}% e indicadores lipídicos {lipid}%. "
            f"Prototipo de detección temprana; consulte a un médico."
        )
    return (
        f"OcuVitals Triage Summary: Overall priority is {risk.triage_priority}. "
        f"Heart rate is {metrics.heart_rate_bpm:.0f} BPM ({metrics.rhythm_flag}), "
        f"and Autonomic Stress Index is {metrics.stress_index:.0f} out of 100 ({metrics.stress_level}). "
        f"Systemic risk flags: Anemia/Pallor {anemia}%, Vascular Strain {vasc}%, Lipid/Icterus {lipid}%. "
        f"This is an early-warning wellness screening prototype; consult a clinician for diagnosis."
    )


def _np_to_reader(rgb_img: np.ndarray) -> ImageReader:
    pil_img = Image.fromarray(rgb_img.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def generate_pdf_triage_report(
    raw_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    vessel_rgb: np.ndarray,
    metrics: AutonomicMetrics,
    risk: SystemicRiskPrediction,
) -> bytes:
    """Generate a 1-page clinical triage PDF report and return raw bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, height - 72, width, 72, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, height - 38, "OcuVitals / OptiScan AI - Clinical Triage Summary")
    c.setFont("Helvetica", 10)
    c.drawString(36, height - 56, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Signal Confidence: {risk.signal_confidence_pct:.0f}% (SQI {metrics.sqi_score:.0f}%)")

    # Triage Priority & Vitals
    y = height - 108
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(36, y, f"Overall Triage Priority: {risk.triage_priority}")
    c.setFont("Helvetica", 11)
    y -= 22
    c.drawString(
        36,
        y,
        f"Heart Rate: {metrics.heart_rate_bpm:.1f} BPM ({metrics.rhythm_flag})   |   "
        f"HRV RMSSD: {metrics.rmssd_ms:.1f} ms   |   SDNN: {metrics.sdnn_ms:.1f} ms",
    )
    y -= 18
    c.drawString(
        36,
        y,
        f"Autonomic Stress Index: {metrics.stress_index:.0f}/100 ({metrics.stress_level})   |   "
        f"PLR Constriction: {metrics.plr_constriction_pct:.1f}% (75% Rec: {metrics.plr_recovery_sec:.2f}s)",
    )

    # Systemic Risk Flags
    y -= 32
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, y, "Systemic Disease Risk Flags (Multimodal Fusion)")
    c.setFont("Helvetica", 10.5)
    y -= 20
    c.drawString(42, y, f"- Anemia / Conjunctival Pallor: {risk.anemia_pallor_prob*100:.0f}%  ({risk.rationale['anemia']})")
    y -= 16
    c.drawString(42, y, f"- Vascular / Hypertension Strain: {risk.vascular_strain_prob*100:.0f}%  ({risk.rationale['vascular']})")
    y -= 16
    c.drawString(42, y, f"- Lipid / Icterus Indicators: {risk.lipid_icterus_prob*100:.0f}%  ({risk.rationale['lipid']})")

    # Visual Eye Panels (Raw, Enhanced, Vessel Skeleton)
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, y, "Ocular Imaging & Vascular Explainability")
    img_y = y - 168
    img_w, img_h = 155, 155
    c.drawImage(_np_to_reader(raw_rgb), 36, img_y, width=img_w, height=img_h)
    c.drawImage(_np_to_reader(enhanced_rgb), 220, img_y, width=img_w, height=img_h)
    c.drawImage(_np_to_reader(vessel_rgb), 404, img_y, width=img_w, height=img_h)

    c.setFont("Helvetica", 9.5)
    c.drawString(36, img_y - 14, "1. Raw Anatomical Crop")
    c.drawString(220, img_y - 14, "2. AI-Enhanced Eye (512x512)")
    c.drawString(404, img_y - 14, "3. Scleral Vessel Skeleton")

    # Disclaimer Footer
    c.setFillColor(colors.HexColor("#78350f"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(36, 48, "MEDICAL DISCLAIMER:")
    c.setFont("Helvetica", 8.5)
    c.drawString(
        36,
        35,
        "OcuVitals / OptiScan AI is an early-warning wellness and triage screening prototype. "
        "It is NOT a standalone clinical diagnostic device.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()
