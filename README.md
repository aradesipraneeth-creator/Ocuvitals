# Ocuviitals — Zero-Hardware Ocular Health Triage

Ocuviitals is a zero-hardware ocular health triage prototype that analyzes eye images and short eye videos to estimate Autonomic Stress, Heart Rate & Rhythm Stability (POS-rPPG + HRV), Pupillary Light Reflex (PLR), and Systemic Risk Flags (Anemia/Conjunctival Pallor, Vascular Strain, Lipid/Icterus Indicators) with an interactive 3D Eye Digital Twin.

## Deploy Globally on Render

1. Push this repository to GitHub (already configured with `render.yaml`).
2. In [Render Dashboard](https://dashboard.render.com/), click **New + -> Blueprint** (or **Web Service**) and connect `https://github.com/aradesipraneeth-creator/Ocuvitals.git`.
3. Render automatically reads `render.yaml`:
   - **Runtime:** Python `3.11.9`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python self_test.py
streamlit run app.py
```

> **Medical Disclaimer:** Ocuviitals is an early-warning wellness and triage screening prototype and is not a standalone clinical diagnostic device.
