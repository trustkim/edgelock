#!/usr/bin/env python3
# scripts/run_demo.py
"""
EdgeLock Submission Demo Runner

Simulates a live camera feed by stepping through a curated sequence of label
images -- one 'frame' at a time -- running each through the real
Detect -> Validate -> Interlock pipeline (core.detector.run_pipeline).
Built for screen-recording the hackathon submission video: paced output, one
clear decision per frame, a running scoreboard, and a final summary.

Usage:
    python3 scripts/run_demo.py                 # default pacing (1.5s/frame)
    python3 scripts/run_demo.py --delay 0        # no pauses, fast run for testing
    python3 scripts/run_demo.py --reset-log      # start audit log fresh for this recording
    python3 scripts/run_demo.py --keep-bom-state # don't reset active_step to 1 first

Note: run_pipeline() persists active_step to data/bom_sample.json on every
UNLOCK (that's the whole point of the sequence-progression fix this demo
exists to show off). By default this script resets active_step back to 1
before each run so the story is reproducible take after take -- pass
--keep-bom-state if you deliberately want to continue from wherever a
previous run left the BOM.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from core.detector import run_pipeline  # noqa: E402
from core.validator import BOMValidator  # noqa: E402

CLR_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[2m"
CLR_RESET = "\033[0m"

# Curated "camera feed": every failure mode first (while active_step is still
# 1, so they're unambiguous), then the full happy path -- RM-A through RM-D,
# genuinely progressing step by step thanks to the active_step fix, with one
# out-of-sequence attempt (RM-D jumping ahead) caught mid-batch. Leads with
# the real submission photos (realistic_labels/); the two failure modes those
# photos don't happen to exercise (lot mismatch, fully unreadable label) use
# the synthetic set. Order is deliberate -- tells a story for the video.
DEMO_SEQUENCE = [
    ("data/mock_labels/sample_invalid_lot_mismatch.jpg",
     "Operator scans the right material, but a mismatched lot"),
    ("data/mock_labels/sample_degraded_valid.jpg",
     "Camera catches a crumpled/blurry label -- lot text unreadable"),
    ("data/mock_labels/realistic_labels/label_RM-E_LOT-20260914E.PNG",
     "Operator grabs a floor-leftover drum not on the BOM at all"),
    ("data/mock_labels/realistic_labels/label_RM-A_LOT-20260901A.PNG",
     "Operator scans Step 1 material correctly -- batch begins"),
    ("data/mock_labels/realistic_labels/label_RM-D_LOT-20260904D.PNG",
     "Operator grabs ahead to Step 4 material -- out of sequence"),
    ("data/mock_labels/realistic_labels/label_RM-B_LOT-20260902B.PNG",
     "Operator scans Step 2 material correctly -- batch continues"),
    ("data/mock_labels/realistic_labels/label_RM-C_LOT-20260903C.PNG",
     "Operator scans Step 3 material correctly -- batch continues"),
    ("data/mock_labels/realistic_labels/label_RM-D_LOT-20260904D.PNG",
     "Operator scans Step 4 material correctly -- batch complete"),
]


def _clear_line() -> None:
    print("\r" + " " * 70 + "\r", end="")


def _simulate_camera_scan(delay: float) -> None:
    if delay <= 0:
        return
    frames = ["   ", ".  ", ".. ", "..."]
    steps = max(1, int(delay / 0.25))
    for i in range(steps):
        print(f"\r  {CLR_DIM}[CAMERA] Scanning label{frames[i % 4]}{CLR_RESET}", end="", flush=True)
        time.sleep(delay / steps)
    _clear_line()


def main() -> int:
    parser = argparse.ArgumentParser(description="EdgeLock Submission Demo Runner")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds of simulated 'camera scan' per frame (0 = instant)")
    parser.add_argument("--reset-log", action="store_true", help="Start data/event_log.json fresh for this recording (backs up the old one first)")
    parser.add_argument("--keep-bom-state", action="store_true", help="Don't reset active_step to 1 before running (continue from wherever it is)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    bom_path = "data/bom_sample.json"
    log_path = "data/event_log.json"

    if args.reset_log and os.path.exists(log_path):
        backup_path = log_path.replace(".json", ".backup.json")
        os.replace(log_path, backup_path)
        print(f"{CLR_YELLOW}[INFO] Existing audit log backed up to {backup_path}{CLR_RESET}")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[]")

    if not args.keep_bom_state:
        with open(bom_path, "r", encoding="utf-8") as f:
            bom_data = json.load(f)
        if bom_data.get("active_step") != 1:
            print(f"{CLR_YELLOW}[INFO] Resetting active_step to 1 for a fresh demo take "
                  f"(was {bom_data.get('active_step')}). Pass --keep-bom-state to skip this.{CLR_RESET}")
        bom_data["active_step"] = 1
        with open(bom_path, "w", encoding="utf-8") as f:
            json.dump(bom_data, f, indent=2)
            f.write("\n")

    validator = BOMValidator(bom_path=bom_path)

    print(f"\n{CLR_BOLD}{'=' * 72}{CLR_RESET}")
    print(f"{CLR_BOLD}  EDGELOCK LIVE DEMO -- Simulated Camera Feed at Reactor R-101 Hatch{CLR_RESET}")
    print(f"{CLR_BOLD}{'=' * 72}{CLR_RESET}")
    print(f"  Active Batch : {CLR_CYAN}{validator.bom_data.get('batch_id')}{CLR_RESET}")
    print(f"  Active Step  : {CLR_CYAN}{validator.bom_data.get('active_step')}{CLR_RESET}\n")

    scoreboard = []

    for idx, (image_path, scenario) in enumerate(DEMO_SEQUENCE, start=1):
        print(f"{CLR_BOLD}[FRAME {idx}/{len(DEMO_SEQUENCE)}]{CLR_RESET} {scenario}")
        _simulate_camera_scan(args.delay)

        event_record = run_pipeline(
            image_path=image_path,
            bom_path=bom_path,
            log_path=log_path,
            mock_gpio=True,
        )
        print()
        scoreboard.append((scenario, event_record["interlock_status"], event_record["note"]))

        if args.delay > 0:
            time.sleep(min(args.delay, 1.0))

    print(f"{CLR_BOLD}{'=' * 72}{CLR_RESET}")
    print(f"{CLR_BOLD}  DEMO SUMMARY{CLR_RESET}")
    print(f"{CLR_BOLD}{'=' * 72}{CLR_RESET}")
    for scenario, action, note in scoreboard:
        color = CLR_GREEN if action == "UNLOCK" else CLR_RED
        print(f"  {color}[{action:6}]{CLR_RESET} {scenario}")
        print(f"           {CLR_DIM}{note}{CLR_RESET}")
    unlocks = sum(1 for _, a, _ in scoreboard if a == "UNLOCK")
    locks = len(scoreboard) - unlocks
    print(f"\n  {unlocks} UNLOCK / {locks} LOCK across {len(scoreboard)} frames. Full audit trail -> {log_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
