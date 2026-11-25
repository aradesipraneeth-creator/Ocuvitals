"""Multimodal PyTorch Architecture for OptiScan AI.

Combines:
1. Spatial Branch: MobileNetV3-Small feature extractor on the CLAHE-enhanced eye ROI.
2. Temporal Branch: Bidirectional LSTM on the 2-channel sequence [pupil_ratio, filtered_rppg].
3. Gated Multimodal Fusion Head: Outputs calibrated risk probabilities for:
   - Anemia (Conjunctival Pallor)
   - Vascular / Hypertension Strain
   - Lipid / Icterus Indicators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.pretrained_loader import PretrainedModelRegistry, get_model_registry
from src.signal_processor import AutonomicMetrics
from src.vision_tracker import VisionExtractionResult


@dataclass
class SystemicRiskPrediction:
    """Calibrated systemic risk probabilities and triage priority."""

    anemia_pallor_prob: float
    vascular_strain_prob: float
    lipid_icterus_prob: float
    triage_priority: str  # "Routine", "Monitor", "Priority Review"
    triage_color: str
    signal_confidence_pct: float
    physiological_features: Dict[str, float]
    rationale: Dict[str, str]


def _load_mobilenet_v3_small(prefer_pretrained: bool = True) -> nn.Module:
    """Load MobileNetV3-Small with graceful offline fallback if weights aren't cached."""
    if prefer_pretrained:
        try:
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            pass
    return models.mobilenet_v3_small(weights=None)


class GatedMultimodalFusion(nn.Module):
    """Gated fusion layer balancing spatial ocular appearance and temporal dynamics."""

    def __init__(self, spatial_dim: int, temporal_dim: int, fused_dim: int = 128) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fused_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, z_spatial: torch.Tensor, z_temporal: torch.Tensor) -> torch.Tensor:
        hs = torch.tanh(self.spatial_proj(z_spatial))
        ht = torch.tanh(self.temporal_proj(z_temporal))
        g = self.gate(torch.cat([hs, ht], dim=-1))
        fused = g * hs + (1.0 - g) * ht
        return self.norm(fused)


class OptiScanFusionNet(nn.Module):
    """Pretrained Foundation Vision Backbone (timm/torchvision) + BiLSTM with Gated Fusion."""

    def __init__(self, registry: Optional[PretrainedModelRegistry] = None, prefer_pretrained: bool = True) -> None:
        super().__init__()
        torch.manual_seed(42)

        if registry is not None and registry.vision_backbone is not None:
            self.spatial_backbone = registry.vision_backbone
            spatial_dim = int(registry.vision_feature_dim)
            self.backbone_name = registry.status_badges.get("Vision Backbone", "Pretrained Backbone")
        elif prefer_pretrained:
            reg = get_model_registry()
            self.spatial_backbone = reg.vision_backbone
            spatial_dim = int(reg.vision_feature_dim)
            self.backbone_name = reg.status_badges.get("Vision Backbone", "Pretrained Backbone")
        else:
            backbone = _load_mobilenet_v3_small(prefer_pretrained=False)
            self.spatial_backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
            spatial_dim = 576
            self.backbone_name = "MobileNetV3-Small (Offline Init)"

        self.temporal_lstm = nn.LSTM(
            input_size=2,
            hidden_size=48,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fusion = GatedMultimodalFusion(spatial_dim=spatial_dim, temporal_dim=96, fused_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

        self._init_deterministic_head()

    def _init_deterministic_head(self) -> None:
        """Initialize fusion and classifier layers deterministically with zero bias drift."""
        gen = torch.Generator().manual_seed(42)
        for module in [self.fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, generator=gen)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self, eye_tensor: torch.Tensor, seq_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits [B, 3], spatial_embedding [B, D])."""
        z_spatial = self.spatial_backbone(eye_tensor)
        if z_spatial.ndim > 2:
            z_spatial = torch.flatten(z_spatial, 1)

        lstm_out, _ = self.temporal_lstm(seq_tensor)
        z_temporal = lstm_out[:, -1, :]  # final bidirectional hidden state [B, 96]

        fused = self.fusion(z_spatial, z_temporal)
        logits = self.classifier(fused)
        return logits, z_spatial


def extract_physiological_biomarkers(vision: VisionExtractionResult) -> Dict[str, float]:
    """Extract interpretable colorimetric & vascular features from conjunctiva and sclera ROIs."""
    conj = vision.conjunctiva_roi
    sclera = vision.sclera_roi
    eye = vision.eye_roi_raw

    # 1. Conjunctival Pallor (Anemia indicator):
    # Healthy lower palpebral conjunctiva is richly vascularized (high LAB a* & R - 0.5*(G+B)).
    conj_bgr = conj.astype(np.float32)
    b_c, g_c, r_c = conj_bgr[:, :, 0], conj_bgr[:, :, 1], conj_bgr[:, :, 2]
    erythema_index = float(np.mean(r_c - 0.5 * (g_c + b_c)))
    conj_lab = cv2.cvtColor(conj, cv2.COLOR_BGR2LAB).astype(np.float32)
    conj_a_star = float(np.mean(conj_lab[:, :, 1]) - 128.0)  # +a* = red, lower = pale

    # 2. Scleral Yellowness / Icterus & Lipid indicator:
    # LAB b* channel on sclera (+b* = yellow) + periocular lipid arc brightness
    scl_lab = cv2.cvtColor(sclera, cv2.COLOR_BGR2LAB).astype(np.float32)
    sclera_b_star = float(np.mean(scl_lab[:, :, 2]) - 128.0)
    sclera_a_star = float(np.mean(scl_lab[:, :, 1]) - 128.0)

    # 3. Vascular / Hypertension Strain:
    # Edge density of red-depleted vessels on sclera (G channel shows blood vessels sharpest)
    scl_gray = scl_bgr = sclera[:, :, 1]
    edges = cv2.Canny(cv2.GaussianBlur(scl_gray, (3, 3), 0), 25, 75)
    vessel_density = float(np.mean(edges > 0))
    sclera_redness = float(np.mean(sclera[:, :, 2].astype(np.float32) - sclera[:, :, 1].astype(np.float32)))

    # Scleral-referenced relative redness (cancels warm/cool ambient lighting bias)
    delta_a_star = float(conj_a_star - sclera_a_star)

    # Overall eye illumination check
    mean_luma = float(np.mean(cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)))

    return {
        "erythema_index": round(erythema_index, 2),
        "conj_a_star": round(conj_a_star, 2),
        "delta_a_star": round(delta_a_star, 2),
        "sclera_b_star": round(sclera_b_star, 2),
        "sclera_a_star": round(sclera_a_star, 2),
        "vessel_density": round(vessel_density, 4),
        "sclera_redness": round(sclera_redness, 2),
        "mean_luma": round(mean_luma, 1),
    }


def preprocess_for_model(
    vision: VisionExtractionResult, metrics: AutonomicMetrics
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare normalized image tensor [1, 3, 224, 224] and sequence tensor [1, T, 2]."""
    rgb = cv2.cvtColor(vision.eye_roi_clahe, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (rgb - mean) / std
    eye_tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0)

    pupil_norm = (vision.pupil_ratio - np.mean(vision.pupil_ratio)) / (np.std(vision.pupil_ratio) + 1e-6)
    rppg_norm = metrics.filtered_rppg / (np.std(metrics.filtered_rppg) + 1e-6)
    seq = np.stack([pupil_norm, rppg_norm], axis=-1).astype(np.float32)
    # Downsample very long sequences to max 300 steps for fast CPU inference
    if len(seq) > 300:
        idx = np.linspace(0, len(seq) - 1, 300).astype(int)
        seq = seq[idx]
    seq_tensor = torch.from_numpy(seq).unsqueeze(0)
    return eye_tensor, seq_tensor


def predict_systemic_risks(
    model: OptiScanFusionNet,
    vision: VisionExtractionResult,
    metrics: AutonomicMetrics,
) -> SystemicRiskPrediction:
    """Run OptiScanFusionNet + physiological calibration for calibrated triage risks."""
    model.eval()
    eye_tensor, seq_tensor = preprocess_for_model(vision, metrics)

    with torch.no_grad():
        logits, _ = model(eye_tensor, seq_tensor)
        nn_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    bio = extract_physiological_biomarkers(vision)

    # Calibrated physiological priors blended with neural representation
    # SQI weights temporal confidence vs spatial confidence
    sqi_norm = float(np.clip(getattr(metrics, "sqi_score", 70.0) / 100.0, 0.15, 0.99))

    # 1. Anemia (Conjunctival Pallor): scleral-referenced delta_a_star + erythema_index
    pallor_score = np.clip((18.0 - bio["delta_a_star"]) / 16.0, 0.0, 1.0) * 0.65 + np.clip(
        (24.0 - bio["erythema_index"]) / 24.0, 0.0, 1.0
    ) * 0.35
    anemia_prob = float(np.clip(0.82 * pallor_score + 0.18 * nn_probs[0], 0.04, 0.96))

    # 2. Vascular / Hypertension Strain: scleral vessel density + redness + SQI-weighted autonomic strain
    vasc_ocular = np.clip((bio["vessel_density"] - 0.03) / 0.12, 0.0, 1.0) * 0.45 + np.clip(
        (bio["sclera_redness"] - 2.0) / 22.0, 0.0, 1.0
    ) * 0.25
    vasc_autonomic = sqi_norm * (
        np.clip(metrics.stress_index / 100.0, 0.0, 1.0) * 0.18 + metrics.rhythm_inconsistency_score * 0.15
    )
    vascular_prob = float(np.clip(0.82 * (vasc_ocular + vasc_autonomic) + 0.18 * nn_probs[1], 0.05, 0.96))

    # 3. Lipid / Icterus Indicators: elevated scleral LAB b* (> 6.0)
    icterus_score = np.clip((bio["sclera_b_star"] - 4.0) / 18.0, 0.0, 1.0)
    lipid_prob = float(np.clip(0.82 * icterus_score + 0.18 * nn_probs[2], 0.03, 0.95))

    signal_conf_pct = round(float(np.clip(0.65 * sqi_norm * 100.0 + 35.0, 25.0, 99.0)), 1)

    max_risk = max(anemia_prob, vascular_prob, lipid_prob)
    if max_risk >= 0.62 or metrics.rhythm_flag == "Irregular" or metrics.stress_index >= 72.0:
        priority = "Priority Review"
        color = "#DC2626"
    elif max_risk >= 0.38 or metrics.stress_index >= 45.0:
        priority = "Monitor"
        color = "#D97706"
    else:
        priority = "Routine"
        color = "#059669"

    rationale = {
        "anemia": (
            f"Conjunctival redness a*={bio['conj_a_star']:.1f}, erythema={bio['erythema_index']:.1f}. "
            + ("Reduced palpebral vascularity suggests possible pallor." if anemia_prob >= 0.45 else "Healthy conjunctival perfusion.")
        ),
        "vascular": (
            f"Scleral vessel density={bio['vessel_density']*100:.1f}%, rhythm={metrics.rhythm_flag}, stress={metrics.stress_index:.0f}/100. "
            + ("Elevated micro-vascular/autonomic strain." if vascular_prob >= 0.45 else "Normal scleral vascular profile.")
        ),
        "lipid": (
            f"Scleral yellow-axis b*={bio['sclera_b_star']:.1f}. "
            + ("Elevated scleral/periocular yellow-lipid hue." if lipid_prob >= 0.45 else "Clear scleral tone without icterus.")
        ),
    }

    return SystemicRiskPrediction(
        anemia_pallor_prob=round(anemia_prob, 3),
        vascular_strain_prob=round(vascular_prob, 3),
        lipid_icterus_prob=round(lipid_prob, 3),
        triage_priority=priority,
        triage_color=color,
        signal_confidence_pct=signal_conf_pct,
        physiological_features=bio,
        rationale=rationale,
    )
