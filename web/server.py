#!/usr/bin/env python3
# web/server.py
"""
EdgeLock Real-Time Operator Dashboard API Server
FastAPI backend connecting edge vision detection, deterministic BOM validation,
fail-safe interlock control, and plant audit trail.
"""

import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.detector import ContainerLabelDetector
from core.interlock import InterlockController
from core.validator import BOMValidator

BOM_PATH = os.path.join(PROJECT_ROOT, "data", "bom_sample.json")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "event_log.json")
MOCK_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "mock_labels")
REALISTIC_DIR = os.path.join(MOCK_LABELS_DIR, "realistic_labels")
STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")
TEMP_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "temp_uploads")

os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="EdgeLock Operator Safety Dashboard API",
    description="Edge AI Fail-Safe Interlock & Traceability Engine for Chemical Plants",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(MOCK_LABELS_DIR):
    app.mount("/mock_labels", StaticFiles(directory=MOCK_LABELS_DIR), name="mock_labels")

# Engine singletons
detector = ContainerLabelDetector()
validator = BOMValidator(bom_path=BOM_PATH)
interlock = InterlockController(mock_gpio=True, log_path=LOG_PATH, bom_path=BOM_PATH)


class PresetRequest(BaseModel):
    preset_filename: str
    operator_id: Optional[str] = "mfg01"
    location: Optional[str] = "Reactor R-101 Hatch"


@app.get("/")
def serve_dashboard():
    """Serves real-time operator HMI dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        return JSONResponse(
            status_code=200,
            content={
                "message": "EdgeLock API is running. Create web/static/index.html to view dashboard UI.",
                "endpoints": ["/api/status", "/api/bom", "/api/logs", "/api/presets", "/api/verify/preset"],
            },
        )
    return FileResponse(index_file)


@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Returns current physical hatch interlock status and active batch metadata."""
    validator.reload_bom()
    return {
        "interlock_state": interlock.current_state,
        "batch_id": validator.bom_data.get("batch_id", "UNKNOWN"),
        "product_name": validator.bom_data.get("product_name", "UNKNOWN"),
        "active_step": validator.bom_data.get("active_step", 1),
        "mock_gpio": interlock.mock_gpio,
    }


@app.get("/api/bom")
def get_active_bom() -> Dict[str, Any]:
    """Retrieves active batch recipe and ingredient whitelist."""
    validator.reload_bom()
    return validator.bom_data


@app.get("/api/logs")
def get_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves immutable batch lifecycle audit logs."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
            return logs[-limit:] if isinstance(logs, list) else []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read audit logs: {exc}")


@app.get("/api/presets")
def get_test_presets() -> List[Dict[str, str]]:
    """Lists available realistic test container images for one-click demo."""
    if not os.path.exists(REALISTIC_DIR):
        return []
    presets = []
    for fname in sorted(os.listdir(REALISTIC_DIR)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            presets.append({
                "filename": fname,
                "url": f"/mock_labels/realistic_labels/{fname}",
            })
    return presets


def _process_image_pipeline(image_path: str, operator_id: str, location: str) -> Dict[str, Any]:
    """Executes Detection -> Whitelist Validation -> GPIO Latch -> Audit Append."""
    scanned_code, scanned_lot, _ = detector.scan_image(image_path)

    if not scanned_code:
        val_result = {
            "timestamp": "",
            "batch_id": validator.bom_data.get("batch_id", "UNKNOWN"),
            "scanned_code": "UNREADABLE",
            "scanned_lot": scanned_lot or "",
            "status": "LABEL_UNREADABLE",
            "interlock_action": "LOCK",
            "reason": "CRITICAL INTERLOCK: Barcode unreadable. Fail-safe locked.",
        }
    else:
        val_result = validator.validate(scanned_code=scanned_code, scanned_lot=scanned_lot or "")

    event_record = interlock.execute(
        val_result,
        operator_id=operator_id,
        location=location,
    )

    # Mirrors core/detector.py's run_pipeline(): only advance active_step
    # after a genuine UNLOCK, never on a LOCK/deny decision. Without this,
    # every material charged through the dashboard after the first one
    # would incorrectly LOCK on SEQUENCE_ERROR even when scanned correctly.
    if val_result["interlock_action"] == "UNLOCK":
        validator.advance_active_step()

    return {
        "scanned_code": scanned_code,
        "scanned_lot": scanned_lot,
        "validation": val_result,
        "interlock_action": val_result["interlock_action"],
        "event_record": event_record,
    }


@app.post("/api/verify/preset")
def verify_preset(req: PresetRequest) -> Dict[str, Any]:
    """Executes validation pipeline using a preset realistic container image."""
    target_path = os.path.join(REALISTIC_DIR, req.preset_filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Preset image '{req.preset_filename}' not found.")

    result = _process_image_pipeline(target_path, req.operator_id or "mfg01", req.location or "Reactor R-101 Hatch")
    result["image_url"] = f"/mock_labels/realistic_labels/{req.preset_filename}"
    return result


@app.post("/api/verify/upload")
async def verify_upload(
    file: UploadFile = File(...),
    operator_id: str = "mfg01",
    location: str = "Reactor R-101 Hatch",
) -> Dict[str, Any]:
    """Executes validation pipeline on an image uploaded via drag-and-drop."""
    saved_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = _process_image_pipeline(saved_path, operator_id, location)
    result["image_url"] = f"/static/temp_uploads/{file.filename}"
    return result


@app.post("/api/step/advance")
def advance_batch_step() -> Dict[str, Any]:
    """Advances batch active_step to simulate multi-stage production sequence."""
    validator.reload_bom()
    current_step = validator.bom_data.get("active_step", 1)
    recipe = validator.bom_data.get("recipe", [])
    max_step = max([item.get("step", 1) for item in recipe], default=current_step)

    next_step = current_step + 1 if current_step < max_step else 1
    validator.bom_data["active_step"] = next_step

    with open(BOM_PATH, "w", encoding="utf-8") as f:
        json.dump(validator.bom_data, f, indent=2, ensure_ascii=False)

    return {
        "batch_id": validator.bom_data.get("batch_id"),
        "previous_step": current_step,
        "active_step": next_step,
        "message": f"Active batch sequence advanced to Step {next_step}.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)