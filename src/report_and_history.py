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


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "en-IN",
    "हिन्दी (Hindi)": "hi-IN",
    "বাংলা (Bengali)": "bn-IN",
    "తెలుగు (Telugu)": "te-IN",
    "मराठी (Marathi)": "mr-IN",
    "தமிழ் (Tamil)": "ta-IN",
    "ગુજરાતી (Gujarati)": "gu-IN",
    "ಕನ್ನಡ (Kannada)": "kn-IN",
    "മലയാളം (Malayalam)": "ml-IN",
    "ਪੰਜਾਬੀ (Punjabi)": "pa-IN",
    "ଓଡ଼ିଆ (Odia)": "or-IN",
    "অসমীয়া (Assamese)": "as-IN",
    "اردو (Urdu)": "ur-IN",
    "Español (Spanish)": "es-ES",
}


def build_multilingual_summary(
    metrics: AutonomicMetrics,
    risk: SystemicRiskPrediction,
    lang: str = "English",
) -> str:
    """Return a clear, primary-care-friendly triage summary in English, all major Indian languages, or Spanish."""
    anemia = int(round(risk.anemia_pallor_prob * 100))
    vasc = int(round(risk.vascular_strain_prob * 100))
    lipid = int(round(risk.lipid_icterus_prob * 100))
    hr = f"{metrics.heart_rate_bpm:.0f}"
    stress = f"{metrics.stress_index:.0f}"
    prio = risk.triage_priority
    rhy = metrics.rhythm_flag
    lvl = metrics.stress_level

    templates = {
        "हिन्दी (Hindi)": (
            f"Ocuviitals स्क्रीनिंग सारांश: प्राथमिकता स्तर {prio} है। "
            f"हृदय गति {hr} BPM ({rhy}), तनाव सूचकांक {stress}/100 ({lvl}) है। "
            f"एनीमिया जोखिम {anemia}%, वैस्कुलर तनाव {vasc}%, लिपिड/पीलिया संकेत {lipid}% हैं। "
            f"यह केवल प्रारंभिक स्क्रीनिंग है, डॉक्टर की सलाह अवश्य लें।"
        ),
        "বাংলা (Bengali)": (
            f"Ocuviitals স্ক্রিনিং সারসংক্ষেপ: অগ্রাধিকার স্তর {prio}। "
            f"হৃদস্পন্দন {hr} BPM ({rhy}), স্ট্রেস সূচক {stress}/100 ({lvl})। "
            f"অ্যানিমিয়া ঝুঁকি {anemia}%, ভাস্কুলার চাপ {vasc}%, লিপিড/জন্ডিস সংকেত {lipid}%। "
            f"এটি কেবল প্রাথমিক স্ক্রিনিং; চিকিৎসকের পরামর্শ নিন।"
        ),
        "తెలుగు (Telugu)": (
            f"Ocuviitals స్క్రీనింగ్ సారాంశం: ప్రాధాన్యత స్థాయి {prio}. "
            f"హృదయ స్పందన {hr} BPM ({rhy}), ఒత్తిడి సూచిక {stress}/100 ({lvl}). "
            f"రక్తహీనత ప్రమాదం {anemia}%, రక్తనాళ ఒత్తిడి {vasc}%, లిపిడ్/కామెర్లు {lipid}%. "
            f"ఇది ప్రాథమిక స్క్రీనింగ్ మాత్రమే; వైద్యుడిని సంప్రదించండి."
        ),
        "मराठी (Marathi)": (
            f"Ocuviitals तपासणी सारांश: प्राधान्य पातळी {prio} आहे. "
            f"हृदय गती {hr} BPM ({rhy}), ताण निर्देशांक {stress}/100 ({lvl}) आहे. "
            f"अॅनिमिया धोका {anemia}%, रक्तवाहिन्यांचा ताण {vasc}%, लिपिड/कावीळ संकेत {lipid}% आहेत. "
            f"ही केवळ प्राथमिक तपासणी आहे; डॉक्टरांचा सल्ला घ्या."
        ),
        "தமிழ் (Tamil)": (
            f"Ocuviitals பரிசோதனை சுருக்கம்: முன்னுரிமை நிலை {prio}. "
            f"இதயத் துடிப்பு {hr} BPM ({rhy}), மன அழுத்த குறியீடு {stress}/100 ({lvl}). "
            f"இரத்த சோகை ஆபத்து {anemia}%, இரத்த நாள அழுத்தம் {vasc}%, கொழுப்பு/மஞ்சள் காமாலை அறிகுறி {lipid}%. "
            f"இது ஆரம்பநிலை பரிசோதனை மட்டுமே; மருத்துவரை அணுகவும்."
        ),
        "ગુજરાતી (Gujarati)": (
            f"Ocuviitals સ્ક્રીનીંગ સારાંશ: પ્રાથમિકતા સ્તર {prio} છે. "
            f"હૃદયના ધબકારા {hr} BPM ({rhy}), સ્ટ્રેસ ઇન્ડેક્સ {stress}/100 ({lvl}) છે. "
            f"એનિમિયા જોખમ {anemia}%, વાસ્ક્યુલર તાણ {vasc}%, લિપિડ/કમળો સંકેત {lipid}% છે. "
            f"આ માત્ર પ્રારંભિક તપાસ છે; ડૉક્ટરની સલાહ લો."
        ),
        "ಕನ್ನಡ (Kannada)": (
            f"Ocuviitals ತಪಾಸಣೆ ಸಾರಾಂಶ: ಆದ್ಯತೆಯ ಮಟ್ಟ {prio}. "
            f"ಹೃದಯ ಬಡಿತ {hr} BPM ({rhy}), ಒತ್ತಡ ಸೂಚ್ಯಂಕ {stress}/100 ({lvl}). "
            f"ರಕ್ತಹೀನತೆ ಅಪಾಯ {anemia}%, ರಕ್ತನಾಳದ ಒತ್ತಡ {vasc}%, ಲಿಪಿಡ್/ಕಾಮಾಲೆ ಸೂಚನೆ {lipid}%. "
            f"ಇದು ಪ್ರಾಥಮಿಕ ತಪಾಸಣೆ ಮಾತ್ರ; ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
        "മലയാളം (Malayalam)": (
            f"Ocuviitals സ്ക്രീനിംഗ് സംഗ്രഹം: മുൻഗണനാ നില {prio}. "
            f"ഹൃദയമിടിപ്പ് {hr} BPM ({rhy}), സ്ട്രെസ് സൂചിക {stress}/100 ({lvl}). "
            f"വിളർച്ച സാധ്യത {anemia}%, വാസ്കുലർ സമ്മർദ്ദം {vasc}%, ലിപിഡ്/മഞ്ഞപ്പിത്ത സൂചന {lipid}%. "
            f"ഇത് പ്രാഥമിക പരിശോധന മാത്രമാണ്; ഡോക്ടറെ സമീപിക്കുക."
        ),
        "ਪੰਜਾਬੀ (Punjabi)": (
            f"Ocuviitals ਜਾਂਚ ਸਾਰ: ਤਰਜੀਹ ਪੱਧਰ {prio} ਹੈ। "
            f"ਦਿਲ ਦੀ ਧੜਕਣ {hr} BPM ({rhy}), ਤਣਾਅ ਸੂਚਕ {stress}/100 ({lvl}) ਹੈ। "
            f"ਅਨੀਮੀਆ ਖਤਰਾ {anemia}%, ਨਾੜੀ ਦਬਾਅ {vasc}%, ਲਿਪਿਡ/ਪੀਲੀਆ ਸੰਕੇਤ {lipid}% ਹਨ। "
            f"ਇਹ ਸਿਰਫ਼ ਸ਼ੁਰੂਆਤੀ ਜਾਂਚ ਹੈ; ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਲਓ।"
        ),
        "ଓଡ଼ିଆ (Odia)": (
            f"Ocuviitals ସ୍କ୍ରିନିଂ ସାରାଂଶ: ପ୍ରାଥମିକତା ସ୍ତର {prio}। "
            f"ହୃଦସ୍ପନ୍ଦନ {hr} BPM ({rhy}), ଚାପ ସୂଚକାଙ୍କ {stress}/100 ({lvl})। "
            f"ରକ୍ତହୀନତା ବିପଦ {anemia}%, ରକ୍ତନଳୀ ଚାପ {vasc}%, ଲିପିଡ୍/ଜଣ୍ଡିସ୍ ସଙ୍କେତ {lipid}%। "
            f"ଏହା କେବଳ ପ୍ରାଥମିକ ପରୀକ୍ଷା; ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ନିଅନ୍ତୁ।"
        ),
        "অসমীয়া (Assamese)": (
            f"Ocuviitals স্ক্ৰীণিং সাৰাংশ: অগ্ৰাধিকাৰ স্তৰ {prio}। "
            f"হৃদস্পন্দন {hr} BPM ({rhy}), মানসিক চাপ সূচক {stress}/100 ({lvl})। "
            f"ৰক্তহীনতাৰ আশংকা {anemia}%, ৰক্তবাহী নলীৰ চাপ {vasc}%, লিপিড/জণ্ডিছ সংকেত {lipid}%। "
            f"এয়া কেৱল প্ৰাথমিক পৰীক্ষা; চিকিৎসকৰ পৰামৰ্শ লওক।"
        ),
        "اردو (Urdu)": (
            f"Ocuviitals اسکریننگ خلاصہ: ترجیحی سطح {prio} ہے۔ "
            f"دل کی دھڑکن {hr} BPM ({rhy})، تناؤ انڈیکس {stress}/100 ({lvl}) ہے۔ "
            f"انیمیا کا خطرہ {anemia}%، عروقی دباؤ {vasc}%، لپڈ/یرقان علامات {lipid}% ہیں۔ "
            f"یہ صرف ابتدائی اسکریننگ ہے؛ ڈاکٹر سے مشورہ کریں۔"
        ),
        "Español (Spanish)": (
            f"Resumen de triaje Ocuviitals: Prioridad {prio}. "
            f"Frecuencia cardíaca {hr} BPM ({rhy}), Índice de estrés {stress}/100 ({lvl}). "
            f"Riesgo de anemia {anemia}%, tensión vascular {vasc}% e indicadores lipídicos {lipid}%. "
            f"Prototipo de detección temprana; consulte a un médico."
        ),
    }

    if lang in templates:
        return templates[lang]
    return (
        f"Ocuviitals Triage Summary: Overall priority is {prio}. "
        f"Heart rate is {hr} BPM ({rhy}), and Autonomic Stress Index is {stress} out of 100 ({lvl}). "
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
    c.drawString(36, height - 38, "Ocuviitals - Clinical Triage Summary")
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
        "Ocuviitals is an early-warning wellness and triage screening prototype. "
        "It is NOT a standalone clinical diagnostic device.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()
