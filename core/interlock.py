#!/usr/bin/env python3
# core/interlock.py
"""
EdgeLock Hardware Fail-Safe Interlock Controller
Bridges BOM validation decisions to physical solenoid relays, alarm beacons,
and immutable audit log records.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ANSI Color Codes for Industrial Status Logging
CLR_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"


class InterlockController:
    """Controls physical solenoid valves, alarm relays, and audit logs based on validation results."""

    def __init__(
        self,
        mock_gpio: bool = True,
        log_path: str = "data/event_log.json",
        bom_path: str = "data/bom_sample.json",
        relay_pin: int = 17,
        alarm_pin: int = 27,
    ) -> None:
        self.mock_gpio = mock_gpio
        self.log_path = log_path
        self.bom_path = bom_path
        self.relay_pin = relay_pin
        self.alarm_pin = alarm_pin
        self.current_state = "LOCK"  # Fail-Safe default

        self._init_gpio()

    def _init_gpio(self) -> None:
        """Initializes target hardware GPIO or activates mock simulation."""
        if not self.mock_gpio:
            try:
                # Production SiMA Dev Kit / Embedded Linux GPIO (e.g. gpiod or RPi.GPIO)
                import RPi.GPIO as GPIO  # type: ignore

                self.gpio = GPIO
                self.gpio.setmode(self.gpio.BCM)
                self.gpio.setup(self.relay_pin, self.gpio.OUT, initial=self.gpio.LOW)
                self.gpio.setup(self.alarm_pin, self.gpio.OUT, initial=self.gpio.LOW)
            except (ImportError, RuntimeError) as exc:
                print(
                    f"{CLR_YELLOW}[WARN] Hardware GPIO unavailable ({exc}). "
                    f"Falling back to Mock GPIO mode.{CLR_RESET}"
                )
                self.mock_gpio = True
                self.gpio = None
        else:
            self.gpio = None

    def execute(
        self,
        validation_result: Dict[str, Any],
        operator_id: str = "mfg01",
        location: str = "Reactor R-101 Hatch",
        quantity_kg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes fail-safe interlock actions and registers an audit record.

        Args:
            validation_result: Dictionary returned by BOMValidator.validate().
            operator_id: Operator ID handling the charging step.
            location: Plant physical feeding location.
            quantity_kg: Optional measured feed quantity.
        """
        action = validation_result.get("interlock_action", "LOCK")
        status = validation_result.get("status", "UNKNOWN")
        code = validation_result.get("scanned_code", "")
        lot = validation_result.get("scanned_lot", "")
        batch_id = validation_result.get("batch_id", "")
        reason = validation_result.get("reason", "")

        # 1. Hardware Actuation
        if action == "UNLOCK":
            self._set_hardware(relay=True, alarm=False)
            self.current_state = "UNLOCK"
            event_type = "CHARGE_PASS"
        else:
            self._set_hardware(relay=False, alarm=True)
            self.current_state = "LOCK"
            event_type = "CHARGE_BLOCKED"

        # 2. Extract recipe metadata for trace log
        material_name = self._resolve_material_name(code)
        resolved_qty = quantity_kg if quantity_kg is not None else self._resolve_target_weight(code)

        # 3. Write Immutable Audit Log
        event_record = self._append_event_log(
            event_type=event_type,
            operator_id=operator_id,
            location=location,
            material_code=code,
            material_name=material_name,
            lot_no=lot,
            quantity_kg=resolved_qty,
            batch_id=batch_id,
            interlock_status=action,
            note=reason,
        )

        # 4. Console Logging
        self._print_audit_banner(action, status, code, lot, event_record["event_id"], reason)
        return event_record

    def _set_hardware(self, relay: bool, alarm: bool) -> None:
        """Drives hardware pins or outputs state change logs."""
        if self.mock_gpio:
            relay_str = f"{CLR_GREEN}ENERGIZED (Hatch Open){CLR_RESET}" if relay else f"{CLR_RED}DE-ENERGIZED (Hatch Locked){CLR_RESET}"
            alarm_str = f"{CLR_RED}ON (Audible Warning){CLR_RESET}" if alarm else f"{CLR_GREEN}OFF{CLR_RESET}"
            print(f"  [MOCK GPIO] Solenoid Relay (Pin {self.relay_pin}) : {relay_str}")
            print(f"  [MOCK GPIO] Alarm Beacon   (Pin {self.alarm_pin}) : {alarm_str}")
        else:
            self.gpio.output(self.relay_pin, self.gpio.HIGH if relay else self.gpio.LOW)
            self.gpio.output(self.alarm_pin, self.gpio.HIGH if alarm else self.gpio.LOW)

    def _resolve_material_name(self, code: str) -> str:
        """Extracts material name from BOM recipe if present."""
        if os.path.exists(self.bom_path):
            try:
                with open(self.bom_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("recipe", []):
                    if item.get("material_code") == code:
                        return item.get("material_name", code)
            except Exception:
                pass
        return code

    def _resolve_target_weight(self, code: str) -> float:
        """Extracts target weight from BOM recipe if available."""
        if os.path.exists(self.bom_path):
            try:
                with open(self.bom_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("recipe", []):
                    if item.get("material_code") == code:
                        return float(item.get("target_weight_kg", 0.0))
            except Exception:
                pass
        return 0.0

    def _append_event_log(
        self,
        event_type: str,
        operator_id: str,
        location: str,
        material_code: str,
        material_name: str,
        lot_no: str,
        quantity_kg: float,
        batch_id: str,
        interlock_status: str,
        note: str,
    ) -> Dict[str, Any]:
        """Appends a new verified event record to data/event_log.json."""
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        events = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        events = json.loads(content)
            except json.JSONDecodeError:
                events = []

        # Generate next sequential event ID: EVT-2026-XXXX
        next_seq = len(events) + 1
        event_id = f"EVT-2026-{next_seq:04d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        new_record = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "operator_id": operator_id,
            "department": "MFG",
            "location": location,
            "material_code": material_code,
            "material_name": material_name,
            "lot_no": lot_no,
            "quantity_kg": quantity_kg,
            "batch_id": batch_id,
            "interlock_status": interlock_status,
            "note": note,
        }

        events.append(new_record)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        return new_record

    def _print_audit_banner(
        self, action: str, status: str, code: str, lot: str, event_id: str, reason: str
    ) -> None:
        color = CLR_GREEN if action == "UNLOCK" else CLR_RED
        print(f"\n{CLR_BOLD}--- [INTERLOCK ACTION TRIGGERED] ---{CLR_RESET}")
        print(f"  Decision Signal: {color}{CLR_BOLD}[{action}]{CLR_RESET} ({status})")
        print(f"  Target Scanned : Code={code} | Lot={lot}")
        print(f"  Audit Logged   : {CLR_CYAN}{event_id}{CLR_RESET} -> {self.log_path}")
        print(f"  Trace Detail   : {reason}\n")

    def cleanup(self) -> None:
        """Fail-Safe release on process termination."""
        if not self.mock_gpio and self.gpio:
            self.gpio.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeLock Hardware Interlock Controller")
    parser.add_argument("--mock-gpio", action="store_true", default=True, help="Simulate GPIO via console output")
    parser.add_argument("--log-path", type=str, default="data/event_log.json", help="Path to event audit log")
    args = parser.parse_args()

    controller = InterlockController(mock_gpio=args.mock_gpio, log_path=args.log_path)

    print(f"\n{CLR_BOLD}======================================================================{CLR_RESET}")
    print(f"{CLR_BOLD}  EdgeLock Interlock Controller Lifecycle Test Suite                  {CLR_RESET}")
    print(f"  Mode: {'MOCK SIMULATION' if controller.mock_gpio else 'HARDWARE INTERFACE'}")
    print(f"{CLR_BOLD}======================================================================{CLR_RESET}\n")

    batch_id = "BATCH-2026-0916-01"

    # [1단계] SCM 담당자(scm03)의 올바른 원료(RM-D) 재불출 (DISPATCH)
    print(f"{CLR_BOLD}[STEP 1] scm03: Corrective Dispatch for Step 4 (RM-D 16.0kg Can){CLR_RESET}")
    rec_disp = controller._append_event_log(
        event_type="DISPATCH",
        operator_id="scm03",
        location="Warehouse Chemical Locker",
        material_code="RM-D",
        material_name="Additive D",
        lot_no="LOT-20260904D",
        quantity_kg=16.0,
        batch_id=batch_id,
        interlock_status="BYPASS",
        note="CORRECTIVE DISPATCH: scm03 dispatched authorized RM-D (16.0kg can) after previous hatch interlock.",
    )
    print(f"  Audit Logged   : {CLR_CYAN}{rec_disp['event_id']}{CLR_RESET} [DISPATCH] -> {controller.log_path}\n")

    # [2단계] 현장 작업자(mfg01) 스테이징 인수인계 검수 (HANDOVER_PASS)
    print(f"{CLR_BOLD}[STEP 2] mfg01: Handover Inspection at Reactor R-101 Staging{CLR_RESET}")
    rec_handover = controller._append_event_log(
        event_type="HANDOVER_PASS",
        operator_id="mfg01",
        location="Reactor R-101 Staging",
        material_code="RM-D",
        material_name="Additive D",
        lot_no="LOT-20260904D",
        quantity_kg=16.0,
        batch_id=batch_id,
        interlock_status="PASS",
        note="HANDOVER ACCEPTED: mfg01 verified 16.0kg RM-D can label against Step 4 requirements.",
    )
    print(f"  Audit Logged   : {CLR_CYAN}{rec_handover['event_id']}{CLR_RESET} [HANDOVER_PASS] -> {controller.log_path}\n")

    # [3단계] 현장 투입구(EdgeLock) 비전 검증 통과 및 릴레이 해제 (CHARGE_PASS / UNLOCK)
    print(f"{CLR_BOLD}[STEP 3] mfg01: Feed Hatch Clearance & Actuation (Pass -> Unlock Relay){CLR_RESET}")
    step4_validation = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_id": batch_id,
        "scanned_code": "RM-D",
        "scanned_lot": "LOT-20260904D",
        "status": "PASS",
        "interlock_action": "UNLOCK",
        "reason": "VALIDATION CLEARED: Additive D [RM-D] approved for Step 4 charging. Solenoid valve hatch unlocked.",
    }
    controller.execute(step4_validation, operator_id="mfg01", location="Reactor R-101 Hatch", quantity_kg=2.0)

    # [4단계] 투입 완료 후 용기 잔량 계근 (POST_CHARGE_WEIGH)
    print(f"{CLR_BOLD}[STEP 4] mfg01: Post-Charge Residual Weigh-in (16.0kg - 2.0kg = 14.0kg){CLR_RESET}")
    rec_weigh = controller._append_event_log(
        event_type="POST_CHARGE_WEIGH",
        operator_id="mfg01",
        location="Reactor R-101 Floor Scale",
        material_code="RM-D",
        material_name="Additive D",
        lot_no="LOT-20260904D",
        quantity_kg=14.0,
        batch_id=batch_id,
        interlock_status="PASS",
        note="RESIDUAL WEIGH-IN: Initial 16.0kg - Fed 2.0kg = 14.0kg residual can weighed and sealed for return.",
    )
    print(f"  Audit Logged   : {CLR_CYAN}{rec_weigh['event_id']}{CLR_RESET} [POST_CHARGE_WEIGH] -> {controller.log_path}\n")

    # [5단계] 창고 반납 및 scm03 잔량 입고 검수 (RETURN_PASS)
    print(f"{CLR_BOLD}[STEP 5] scm03: Residual Return Acceptance Inspection{CLR_RESET}")
    rec_return = controller._append_event_log(
        event_type="RETURN_PASS",
        operator_id="scm03",
        location="Warehouse Chemical Locker",
        material_code="RM-D",
        material_name="Additive D",
        lot_no="LOT-20260904D",
        quantity_kg=14.0,
        batch_id=batch_id,
        interlock_status="PASS",
        note="RETURN ACCEPTED: scm03 inspected and checked in 14.0kg residual RM-D back to warehouse storage.",
    )
    print(f"  Audit Logged   : {CLR_CYAN}{rec_return['event_id']}{CLR_RESET} [RETURN_PASS] -> {controller.log_path}\n")