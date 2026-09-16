#!/usr/bin/env python3
# tests/test_pipeline.py
"""
EdgeLock End-to-End Pipeline Tests
Runs every "realistic_labels" drum image through the real
Detect -> Validate -> Interlock chain and asserts the expected UNLOCK/LOCK
outcome for each. This is the regression guard for two hardening passes:

1. Detector: barcode decode fails on 100% of these photo-composited labels
   (the barcode graphic isn't actually scannable), so extraction must come
   from genuine OCR of the localized label region -- not a filename crutch.
2. Validator: active_step now persists forward after a successful (UNLOCK)
   charge, via BOMValidator.advance_active_step(). Before this fix,
   active_step was pinned at 1 forever, so every material after the first
   legitimately-charged one would fail SEQUENCE_ERROR even when scanned
   correctly, in order.

Run with a single command:
    python3 -m pytest tests/test_pipeline.py -v

Every test operates on an isolated tmp_path copy of BOTH data/bom_sample.json
and the audit log -- never the real files. This matters more now than before
the active_step fix: run_pipeline() persists BOM state to disk on every
UNLOCK, so a test suite pointed at the real file would permanently corrupt
the team's shared BOM on the very first run.
"""

import json
import os
import shutil
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # detector/validator/interlock default paths are repo-relative

from core.detector import ContainerLabelDetector, run_pipeline  # noqa: E402
from core.validator import BOMValidator  # noqa: E402

LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "mock_labels", "realistic_labels")
REAL_BOM_PATH = os.path.join(PROJECT_ROOT, "data", "bom_sample.json")

# Expected outcome for a single, isolated scan against the BOM's pristine
# starting state (active_step=1, recipe defines only RM-A..RM-D). RM-E/F/G
# are the "floor leftover" / unauthorized materials from the event_log.json
# narrative -- never on the BOM at all, regardless of active_step.
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

# The BOM's real recipe order, used by the full-sequence progression test.
RECIPE_ORDER = ["RM-A", "RM-B", "RM-C", "RM-D"]


def _image_path(material_code: str) -> str:
    matches = [
        f for f in os.listdir(LABELS_DIR) if f.upper().startswith(f"LABEL_{material_code}_")
    ]
    assert matches, f"No realistic_labels image found for {material_code} in {LABELS_DIR}"
    return os.path.join(LABELS_DIR, matches[0])


@pytest.fixture
def isolated_bom_path(tmp_path):
    """A fresh tmp copy of the real BOM for every test -- never mutate the shared file."""
    dest = tmp_path / "bom_sample.json"
    shutil.copy(REAL_BOM_PATH, dest)
    return str(dest)


@pytest.fixture(scope="module")
def detector() -> ContainerLabelDetector:
    return ContainerLabelDetector()


@pytest.mark.parametrize("material_code", sorted(EXPECTED.keys()))
def test_detector_extracts_correct_code_and_lot(detector, material_code):
    """
    Regression test for the detector bug: barcode decode fails on every one
    of these labels, so the detector must fall back to genuine OCR of the
    localized label region for BOTH the material code and lot number --
    not silently lean on the filename (which wouldn't exist on a live
    camera frame).
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
def test_full_pipeline_unlock_or_lock_decision(material_code, tmp_path, isolated_bom_path):
    """
    Full Detect -> Validate -> Interlock integration test, against the BOM's
    pristine starting state (active_step=1). Checks both the binary
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
        bom_path=isolated_bom_path,
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

    # The isolated log must exist and contain exactly this one run's event.
    assert os.path.exists(isolated_log_path)
    with open(isolated_log_path, "r", encoding="utf-8") as f:
        logged_events = json.load(f)
    assert len(logged_events) == 1
    assert logged_events[0]["event_id"] == event_record["event_id"]


def test_active_step_advances_after_successful_charge(tmp_path, isolated_bom_path):
    """
    Regression test for the active_step fix. Before it: active_step never
    moved, so scanning RM-B (a real, correctly-sequenced BOM material)
    right after a successful RM-A charge would incorrectly LOCK on
    SEQUENCE_ERROR. After it: the batch progresses and RM-B correctly
    UNLOCKs in turn.
    """
    log_path = str(tmp_path / "event_log.json")

    validator_before = BOMValidator(bom_path=isolated_bom_path)
    assert validator_before.bom_data.get("active_step") == 1

    record_a = run_pipeline(
        image_path=_image_path("RM-A"), bom_path=isolated_bom_path, log_path=log_path, mock_gpio=True
    )
    assert record_a["interlock_status"] == "UNLOCK"

    validator_after_a = BOMValidator(bom_path=isolated_bom_path)
    assert validator_after_a.bom_data.get("active_step") == 2, (
        "active_step should have advanced to 2 on disk after RM-A's successful charge"
    )

    # The real regression: RM-B must now UNLOCK, not SEQUENCE_ERROR, because
    # active_step genuinely advanced.
    record_b = run_pipeline(
        image_path=_image_path("RM-B"), bom_path=isolated_bom_path, log_path=log_path, mock_gpio=True
    )
    assert record_b["interlock_status"] == "UNLOCK", (
        f"RM-B should UNLOCK once active_step has advanced to 2, got: {record_b['note']}"
    )


def test_full_recipe_sequence_completes_in_order(tmp_path, isolated_bom_path):
    """
    Charges RM-A -> RM-B -> RM-C -> RM-D in the BOM's real order and asserts
    every single one UNLOCKs. This is the "full happy path" the active_step
    fix exists to make possible -- previously only the first material in any
    batch could ever succeed.
    """
    log_path = str(tmp_path / "event_log.json")

    for material_code in RECIPE_ORDER:
        record = run_pipeline(
            image_path=_image_path(material_code), bom_path=isolated_bom_path, log_path=log_path, mock_gpio=True
        )
        assert record["interlock_status"] == "UNLOCK", (
            f"{material_code} should UNLOCK when scanned in correct recipe order, "
            f"got: {record['note']}"
        )

    # All 4 steps charged -- no next step should remain.
    final_validator = BOMValidator(bom_path=isolated_bom_path)
    assert final_validator.advance_active_step() is None, "Batch should be complete after all 4 steps"


def test_pipeline_never_mutates_real_files(tmp_path, isolated_bom_path):
    """Sanity check: the real event_log.json and bom_sample.json are untouched by the suite."""
    real_log_path = os.path.join(PROJECT_ROOT, "data", "event_log.json")
    with open(real_log_path, "r", encoding="utf-8") as f:
        log_before = f.read()
    with open(REAL_BOM_PATH, "r", encoding="utf-8") as f:
        bom_before = f.read()

    run_pipeline(
        image_path=_image_path("RM-A"),
        bom_path=isolated_bom_path,
        log_path=str(tmp_path / "throwaway_log.json"),
        mock_gpio=True,
    )

    with open(real_log_path, "r", encoding="utf-8") as f:
        log_after = f.read()
    with open(REAL_BOM_PATH, "r", encoding="utf-8") as f:
        bom_after = f.read()

    assert log_before == log_after, "Running the pipeline must never mutate the real audit log"
    assert bom_before == bom_after, "Running the pipeline must never mutate the real bom_sample.json"


def test_real_bom_starts_at_step_one():
    """Documents the BOM's committed starting state (not itself a regression guard)."""
    validator = BOMValidator(bom_path=REAL_BOM_PATH)
    assert validator.bom_data.get("active_step") == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
