#!/usr/bin/env python3
# core/validator.py
"""
EdgeLock Core Validation Engine
Zero-Trust whitelist BOM verification for chemical reactor feeding interlocks.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class BOMValidator:
    """Deterministic whitelist verification engine cross-referencing scanned labels against active batch BOMs."""

    def __init__(self, bom_path: str = "data/bom_sample.json") -> None:
        self.bom_path = bom_path
        self.bom_data: Dict[str, Any] = self._load_bom()

    def _load_bom(self) -> Dict[str, Any]:
        """Loads and parses the active BOM JSON recipe."""
        if not os.path.exists(self.bom_path):
            raise FileNotFoundError(
                f"[EdgeLock:Validator] BOM file not found at '{self.bom_path}'. "
                "Ensure data/bom_sample.json exists."
            )
        try:
            with open(self.bom_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("BOM root structure must be a valid JSON object.")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[EdgeLock:Validator] Invalid JSON syntax in '{self.bom_path}': {exc}"
            ) from exc

    def reload_bom(self) -> None:
        """Reloads the active BOM file from disk to reflect real-time production recipe updates."""
        self.bom_data = self._load_bom()

    def advance_active_step(self) -> Optional[int]:
        """
        Advances active_step to the next step in the recipe sequence and persists
        the change to the BOM file on disk.

        Without this, active_step never moves past its initial value, so every
        material after the first legitimately-charged one would incorrectly
        fail SEQUENCE_ERROR forever -- the interlock would look broken even
        when every scan was correct. Callers should invoke this only after a
        successful (UNLOCK) charge, never on a LOCK/deny decision.

        Returns the new active_step, or None if the current step was already
        the last one in the recipe (batch complete, nothing further to charge).
        """
        recipe: List[Dict[str, Any]] = self.bom_data.get("recipe", [])
        current_step = self.bom_data.get("active_step", 1)
        remaining_steps = sorted(
            {item.get("step") for item in recipe if isinstance(item.get("step"), int) and item.get("step") > current_step}
        )
        if not remaining_steps:
            return None

        self.bom_data["active_step"] = remaining_steps[0]
        self._persist_bom()
        return remaining_steps[0]

    def _persist_bom(self) -> None:
        """Writes the current in-memory BOM state back to disk."""
        with open(self.bom_path, "w", encoding="utf-8") as f:
            json.dump(self.bom_data, f, indent=2)
            f.write("\n")

    def validate(self, scanned_code: str, scanned_lot: str = "") -> Dict[str, Any]:
        """
        Validates scanned raw material credentials against the active BOM recipe.

        Validation Steps:
        1. Whitelist Membership: Material code must exist in the active recipe.
        2. Sequence Control: Material step must match the current active_step.
        3. Lot Integrity: Lot number must match allowed_lot if defined and provided.
        """
        code_clean = scanned_code.strip().upper()
        lot_clean = scanned_lot.strip().upper()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        batch_id = self.bom_data.get("batch_id", "UNKNOWN_BATCH")

        # 1. Whitelist Verification
        recipe: List[Dict[str, Any]] = self.bom_data.get("recipe", [])
        matched_item: Optional[Dict[str, Any]] = next(
            (
                item
                for item in recipe
                if str(item.get("material_code", "")).strip().upper() == code_clean
            ),
            None,
        )

        if not matched_item:
            return {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "scanned_code": code_clean,
                "scanned_lot": lot_clean,
                "status": "UNAUTHORIZED_MATERIAL",
                "interlock_action": "LOCK",
                "reason": (
                    f"CRITICAL INTERLOCK: Material [{code_clean}] is not registered "
                    f"in active batch BOM ({batch_id}). Default deny enforced."
                ),
            }

        # 2. Sequence / Active Step Verification
        active_step = self.bom_data.get("active_step", 1)
        item_step = matched_item.get("step")
        if item_step != active_step:
            return {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "scanned_code": code_clean,
                "scanned_lot": lot_clean,
                "status": "SEQUENCE_ERROR",
                "interlock_action": "LOCK",
                "reason": (
                    f"SEQUENCE VIOLATION: Current process expects Step {active_step}, "
                    f"but scanned material [{code_clean}] belongs to Step {item_step}."
                ),
            }

        # 3. Lot Number Verification
        allowed_lot = str(matched_item.get("allowed_lot", "")).strip().upper()
        if allowed_lot and not lot_clean:
            # Fail-safe deny: a required lot that OCR/scan couldn't read must NOT
            # be treated as a silent pass. Without this, a blurry/degraded label
            # (barcode readable, lot text not) would unlock on material code alone.
            return {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "scanned_code": code_clean,
                "scanned_lot": lot_clean,
                "status": "LOT_UNREADABLE",
                "interlock_action": "LOCK",
                "reason": (
                    f"LOT UNREADABLE: Batch requires Lot [{allowed_lot}], but no lot number "
                    f"was decoded from the scanned label. Fail-safe deny enforced."
                ),
            }
        if allowed_lot and allowed_lot != lot_clean:
            return {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "scanned_code": code_clean,
                "scanned_lot": lot_clean,
                "status": "LOT_MISMATCH",
                "interlock_action": "LOCK",
                "reason": (
                    f"LOT MISMATCH: Batch requires Lot [{allowed_lot}], "
                    f"but scanned Lot [{lot_clean}]."
                ),
            }

        # 4. Successful Verification
        resolved_lot = lot_clean if lot_clean else allowed_lot
        material_name = matched_item.get("material_name", code_clean)
        return {
            "timestamp": timestamp,
            "batch_id": batch_id,
            "scanned_code": code_clean,
            "scanned_lot": resolved_lot,
            "status": "PASS",
            "interlock_action": "UNLOCK",
            "reason": (
                f"VALIDATION CLEARED: {material_name} [{code_clean}] approved for "
                f"Step {item_step} charging. Solenoid valve hatch unlocked."
            ),
        }


if __name__ == "__main__":
    # ANSI terminal color escape codes
    CLR_GREEN = "\033[92m"
    CLR_RED = "\033[91m"
    CLR_CYAN = "\033[96m"
    CLR_BOLD = "\033[1m"
    CLR_RESET = "\033[0m"

    # Temporary fallback mock generation if executed before bom_sample.json is present
    MOCK_PATH = "data/bom_sample.json"
    created_mock = False
    if not os.path.exists(MOCK_PATH):
        os.makedirs("data", exist_ok=True)
        fallback_bom = {
            "batch_id": "BATCH-2026-0916-01",
            "product_code": "PROD-101",
            "active_step": 1,
            "recipe": [
                {
                    "step": 1,
                    "material_code": "RM-A",
                    "material_name": "Base Solvent A",
                    "allowed_lot": "LOT-20260901A",
                    "target_weight_kg": 55.0,
                },
                {
                    "step": 2,
                    "material_code": "RM-B",
                    "material_name": "Resin B",
                    "allowed_lot": "LOT-20260902B",
                    "target_weight_kg": 30.0,
                },
                {
                    "step": 3,
                    "material_code": "RM-C",
                    "material_name": "Colorant C",
                    "allowed_lot": "LOT-20260903C",
                    "target_weight_kg": 13.0,
                },
                {
                    "step": 4,
                    "material_code": "RM-D",
                    "material_name": "Additive D",
                    "allowed_lot": "LOT-20260904D",
                    "target_weight_kg": 2.0,
                },
            ],
        }
        with open(MOCK_PATH, "w", encoding="utf-8") as fallback_file:
            json.dump(fallback_bom, fallback_file, indent=2)
        created_mock = True

    validator = BOMValidator(MOCK_PATH)

    test_scenarios = [
        ("RM-A", "LOT-20260901A", "Scenario 1: Authorized Raw Material (Step 1 Match)"),
        ("RM-G", "LOT-20260820G", "Scenario 2: Unauthorized Raw Material (Default Deny)"),
        ("RM-B", "LOT-20260902B", "Scenario 3: Sequence Violation (Step 2 Premature Feed)"),
        ("RM-A", "LOT-9999999X", "Scenario 4: Lot Number Mismatch (Defective Lot)"),
        ("RM-A", "", "Scenario 5: Lot Unreadable (Degraded Label, Fail-Safe Deny)"),
    ]

    print(f"\n{CLR_BOLD}======================================================================{CLR_RESET}")
    print(f"{CLR_BOLD}  EdgeLock Deterministic Interlock Engine Test Suite                  {CLR_RESET}")
    print(f"  Target Batch : {CLR_CYAN}{validator.bom_data.get('batch_id')}{CLR_RESET} | Active Step: {CLR_CYAN}{validator.bom_data.get('active_step')}{CLR_RESET}")
    print(f"{CLR_BOLD}======================================================================{CLR_RESET}\n")

    for code, lot, description in test_scenarios:
        result = validator.validate(code, lot)
        is_unlock = result["interlock_action"] == "UNLOCK"
        badge_color = CLR_GREEN if is_unlock else CLR_RED

        print(f"{CLR_BOLD}[TEST]{CLR_RESET} {description}")
        print(f"  Input Scanned  : Material={CLR_BOLD}{code}{CLR_RESET} | Lot={CLR_BOLD}{lot}{CLR_RESET}")
        print(f"  Interlock Act  : {badge_color}{CLR_BOLD}[{result['interlock_action']}]{CLR_RESET} ({result['status']})")
        print(f"  Audit Trace    : {result['reason']}")
        print(f"  UTC Timestamp  : {result['timestamp']}")
        print("-" * 70)

    if created_mock:
        os.remove(MOCK_PATH)