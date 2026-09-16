#!/usr/bin/env python3
# tests/test_pipeline.py
"""
EdgeLock End-to-End Pipeline Tests
Runs every "realistic_labels" drum image through the real
Detect -> Validate -> Interlock chain and asserts the expected UNLOCK/LOCK
outcome for each. This is the regression guard for the detector hardening
work (label ROI localization + genuine RM-X code OCR replacing the
filename-fallback crutch) -- these exact 7 images are what exposed that bug.

Run with a single command:
    python3 -m pytest tests/test_pipeline.py -v

Every test writes to an isolated tmp_path log file, never to the real
data/event_log.json, so running the suite can never pollute the project's
audit trail / demo narrative.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # detector/validator/interlock default paths are repo-relative

from core.detector import ContainerLabelDetector, run_pipeline  # noqa: E402
from core.validator import BOMValidator  # noqa: E402

LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "mock_labels", "realistic_labels")
BOM_PATH = os.path.join(PROJECT_ROOT, "data", "bom_sample.json")

# Expected outcome per drum, given the current data/bom_sample.json
# (active_step=1, recipe defines only RM-A..RM-D). RM-E/F/G are the
# "floor leftover" / unauthorized materials from the event_log.json narrative.
#
#   material -> (expected interlock_status, substring that must appear in
#                the audit note, proving the *correct* rule fired --
#                not just a coincidentally-matching lock/unlock)
EXPECTED = {
    "RM-A": ("UNLOCK", "VALIDATION CLEARED"),
    "RM-B": ("LOCK", "SEQUENCE VIOLATION"),
    "RM-C": ("LOCK", "SEQUENCE VIOLATION"),
    "RM-D": ("LOCK", "SEQUENCE VIOLATION"),
    "RM-E": ("LOCK", "not registered in active batch BOM"),
    "RM-F": ("LOCK", "not registered in active batch BOM"),
    "RM-G": ("LOCK", "not registered in active batch BOM"),
}


def _image_path(material_code: str) -> str:
    matches = [
        f for f in os.listdir(LABELS_DIR) if f.upper().startswith(f"LABEL_{material_code}_")
    ]
    assert matches, f"No realistic_labels image found for {material_code} in {LABELS_DIR}"
    return os.path.join(LABELS_DIR, matches[0])


@pytest.fixture(scope="module")
def detector() -> ContainerLabelDetector:
    return ContainerLabelDetector()


@pytest.mark.parametrize("material_code", sorted(EXPECTED.keys()))
def test_detector_extracts_correct_code_and_lot(detector, material_code):
    """
    Regression test for the bug this hardening pass fixed: barcode decode
    fails on 100% of these photo-composited labels (the barcode graphic isn't
    actually scannable), so the detector must fall back to genuine OCR of the
    localized label region for BOTH the material code and lot number --
    not silently lean on the filename (which wouldn't exist on a live camera
    frame).
    """
    image_path = _image_path(material_code)
    code, lot, _frame = detector.scan_image(image_path)

    assert code == material_code, f"Expected code {material_code}, got {code} for {image_path}"
    assert lot is not None, f"Lot number was not extracted at all for {image_path}"
    assert lot.startswith("LOT-"), f"Lot '{lot}' doesn't match expected LOT-XXXXXXXXX format"
    assert lot.endswith(material_code[-1]), (
        f"Lot '{lot}' should end in the material's letter suffix ({material_code[-1]}) "
        f"per this batch's naming convention -- if it doesn't, OCR likely mis-cropped "
        f"or mis-read the trailing character again."
    )


@pytest.mark.parametrize("material_code", sorted(EXPECTED.keys()))
def test_full_pipeline_unlock_or_lock_decision(material_code, tmp_path):
    """
    Full Detect -> Validate -> Interlock integration test. Runs the real
    pipeline (mock GPIO, isolated audit log) and checks both the binary
    UNLOCK/LOCK decision AND that the *correct* validation rule fired --
    two different materials can both LOCK for entirely different reasons
    (out-of-sequence vs. not-on-the-BOM-at-all), so asserting only the
    lock/unlock bit would hide a wrong-reason regression.
    """
    image_path = _image_path(material_code)
    expected_status, expected_note_substring = EXPECTED[material_code]
    isolated_log_path = str(tmp_path / "test_event_log.json")

    event_record = run_pipeline(
        image_path=image_path,
        bom_path=BOM_PATH,
        log_path=isolated_log_path,
        mock_gpio=True,
    )

    assert event_record["interlock_status"] == expected_status, (
        f"{material_code}: expected interlock_status={expected_status}, "
        f"got {event_record['interlock_status']} (note: {event_record['note']})"
    )
    assert expected_note_substring in event_record["note"], (
        f"{material_code}: expected note to mention '{expected_note_substring}', "
        f"got: {event_record['note']}"
    )
    assert event_record["material_code"] == material_code

    # The isolated log must exist and contain exactly this one run's event --
    # proves we never touched the real data/event_log.json.
    assert os.path.exists(isolated_log_path)
    with open(isolated_log_path, "r", encoding="utf-8") as f:
        logged_events = json.load(f)
    assert len(logged_events) == 1
    assert logged_events[0]["event_id"] == event_record["event_id"]


def test_pipeline_never_touches_real_audit_log(tmp_path):
    """Sanity check: the real data/event_log.json is untouched by running any of the above."""
    real_log_path = os.path.join(PROJECT_ROOT, "data", "event_log.json")
    with open(real_log_path, "r", encoding="utf-8") as f:
        before = f.read()

    run_pipeline(
        image_path=_image_path("RM-A"),
        bom_path=BOM_PATH,
        log_path=str(tmp_path / "throwaway_log.json"),
        mock_gpio=True,
    )

    with open(real_log_path, "r", encoding="utf-8") as f:
        after = f.read()
    assert before == after, "Running the pipeline must never mutate the real audit log"


def test_bom_active_step_matches_expected_sequence_gate():
    """
    Documents a pre-existing, known limitation (not something this test suite
    should silently hide): active_step in bom_sample.json is pinned at 1 and
    nothing advances it, which is why RM-B/C/D all LOCK on SEQUENCE_ERROR
    above even though they're legitimate BOM materials. If someone fixes that
    bug, this assertion -- and the EXPECTED table above -- will need updating.
    """
    validator = BOMValidator(bom_path=BOM_PATH)
    assert validator.bom_data.get("active_step") == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
