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
