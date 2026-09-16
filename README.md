# EdgeLock: Edge AI Fail-Safe Interlock System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Target_Hardware-SiMA_Dev_Kit_3.0-blue.svg)](https://sima.ai/)
[![Hackathon](https://img.shields.io/badge/Built_At-AI_Infra_Summit_2026-orange.svg)](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon)

> **Ultra-low-power on-device vision system preventing raw material misfeeding in chemical plants via real-time barcode/OCR BOM validation and GPIO fail-safe interlocks.**

---

## 📌 Problem Statement

Small-to-medium enterprises (SMEs) at the outer tier of the chemical manufacturing supply chain face a critical dilemma:

* **High Barrier to Explosion-Proof Automation**: Fully automated feeding robots certified for explosive atmospheres (Zone 1/Zone 2) require capital investments exceeding hundreds of thousands of dollars.
* **Reliance on Aging Workforce**: Operations remain heavily manual—from warehouse picking to reactor hopper charging. Senior operators must visually inspect dozens of near-identical raw material bags and drums daily.
* **Catastrophic Impact of Human Error**: A single wrong ingredient or Lot mischarge ruins an entire batch, creates chemical hazards, and leads to severe disposal and decontamination penalties.
* **Harsh Industrial Conditions**: Dust, chemical splashes, and metallic vessel shielding create Wi-Fi dead zones, making cloud-dependent AI systems unreliable for instantaneous safety decisions.

---

## 💡 Solution: EdgeLock

**EdgeLock** acts as a retrofit, empathetic safety net for shop-floor operators. Deployed directly at the reactor vessel feeding hatch, EdgeLock uses on-device edge AI to validate ingredients in milliseconds before they enter the reaction process.

```
[ Raw Material Label ]
  (Barcode / Lot text / Bag)
              │
              ▼
     [ Wide-Angle Camera ]
              │
              ▼
 ┌─────────────────────────────┐
 │   SiMA Dev Kit 3.0 (MLSoC)  │  <-- Sealed Inside Standard
 │  - Fast Barcode / QR Decode │      Explosion-Proof Enclosure
 │  - Low-Latency On-Device OCR│      (Ultra-low power, Fanless)
 └────────────┬────────────────┘
              │
              ▼
  [ Active BOM Validation ]
              │
    ┌─────────┴─────────┐
 [ MATCH ]          [ MISMATCH ]
    │                   │
    ▼                   ▼
[ GPIO: Unlock Gate ]  [ GPIO: Lock Hatch  ]
[ Log to Plant DB   ]  [ Sound Alarm Light ]
[ Operator: PROCEED ]  [ Operator: STOP    ]
```

1. **Multi-Modal Vision Verification**: Extracts Code 128 / QR codes and deep-learning OCR Lot numbers simultaneously, handling crumpled, stained, or non-standard supplier labels.
2. **Deterministic GPIO Interlock**: Compares decoded Lot numbers against active production Bills of Materials (BOM). On mismatch, a hardware relay physically locks the hopper hatch in <50ms.
3. **End-to-End Plant Traceability**: Successful charges generate an immutable audit log linking warehouse dispatch, operator ID, and reactor charging timestamps.

---

## ⚡ Why SiMA Dev Kit 3.0?

* **Fanless in Sealed Enclosures**: Traditional AI accelerators generate excessive heat, requiring costly thermal solutions inside sealed explosion-proof housings (NEMA 7 / ATEX). SiMA MLSoC's market-leading **TOPS/Watt** efficiency allows continuous fanless operation inside off-the-shelf housings.
* **Zero-Latency Offline Autonomy**: Operates 100% locally with no cloud dependency, ensuring sub-50ms safety interlocks even during complete network dropouts.
* **Native Industrial I/O**: Direct GPIO/Serial signaling bridges modern computer vision pipelines with legacy PLC valves and warning relays.

---

## 🗂 Project Structure

```bash
edgelock/
├── core/
│   ├── detector.py          # Vision pipeline (Barcode + On-device OCR)
│   ├── interlock.py         # GPIO relay control & safety latch
│   └── validator.py         # BOM matching logic engine
├── data/
│   ├── bom_sample.json      # Sample production batch recipes
│   └── mock_labels/         # Sample test label images (Valid & Invalid)
├── web/
│   ├── server.py            # Local dashboard API (FastAPI)
│   └── static/              # Real-time operator dashboard UI
├── tests/
│   └── test_pipeline.py     # End-to-end mock test suite
├── requirements.txt
└── README.md
```

## 🚀 Quick Start (Local & Mock Mode)
Run EdgeLock in simulation mode on any workstation without physical hardware connected:

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/](https://github.com/)<your-username>/edgelock.git
cd edgelock
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline in Mock Mode
```bash
# Simulates label recognition and mock GPIO relay output using test assets
python core/detector.py --input data/mock_labels/sample_valid.jpg --mock-gpio
```

### 3. Launch Operator Dashboard
```bash
uvicorn web.server:app --reload --port 8000
```
Open http://localhost:8000 to view the real-time charging status, camera stream, and batch audit log.

## 📊 Data Schema Example
### Active Batch BOM (data/bom_sample.json)
```json
{
  "batch_id": "BATCH-2026-0916-01",
  "product_code": "PROD-101",
  "product_name": "Product 101",
  "batch_total_kg": 100.0,
  "active_step": 1,
  "recipe": [
    {
      "step": 1,
      "material_code": "RM-A",
      "material_name": "Base Solvent A",
      "allowed_lot": "LOT-20260914A",
      "target_weight_kg": 55.0
    },
    {
      "step": 2,
      "material_code": "RM-B",
      "material_name": "Resin B",
      "allowed_lot": "LOT-20260914B",
      "target_weight_kg": 30.0
    },
    {
      "step": 3,
      "material_code": "RM-C",
      "material_name": "Colorant C",
      "allowed_lot": "LOT-20260915C",
      "target_weight_kg": 13.0
    },
    {
      "step": 4,
      "material_code": "RM-D",
      "material_name": "Additive D",
      "allowed_lot": "LOT-20260916D",
      "target_weight_kg": 2.0
    }
  ]
}
```
### Verification Event Output
```json
{
  "timestamp": "2026-09-16T08:15:32Z",
  "batch_id": "BATCH-2026-0916-01",
  "scanned_item": "RM-02",
  "scanned_lot": "LOT-202608C",
  "result": "PASS",
  "interlock_action": "GATE_UNLOCKED"
}
```

## 👥 Team EdgeLock
Developed on-site for the AI Infra Summit 2026 Hackathon (Santa Clara, CA) supported by SiMA.ai & Lablab.ai.

* Mideum Kim - Concept, Industrial Pipeline, Edge Vision & Embedded Integration

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
