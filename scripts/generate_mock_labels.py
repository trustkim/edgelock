#!/usr/bin/env python3
"""Generate synthetic chemical labels from the EdgeLock event log."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_LOG = PROJECT_ROOT / "data" / "event_log.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "mock_labels"
DEFAULT_GHS_ASSETS_DIR = PROJECT_ROOT / "data" / "assets" / "ghs"

LabelTarget = tuple[str, str, str, float, str]


class GHSRuleEngine:
    """Map package quantities to the label's packaging and GHS information."""

    @staticmethod
    def map_quantity(quantity_kg: float) -> dict[str, str]:
        """Return the packaging/GHS rule matching ``quantity_kg``.

        The source specification does not define a package for the interval from
        50 kg (inclusive) to 150 kg (exclusive), so that interval is rejected
        instead of silently assigning an incorrect label.
        """
        if isinstance(quantity_kg, bool):
            raise TypeError("quantity_kg must be a number, not bool")

        try:
            quantity = float(quantity_kg)
        except (TypeError, ValueError) as exc:
            raise TypeError("quantity_kg must be a number") from exc

        if quantity < 0:
            raise ValueError("quantity_kg cannot be negative")
        if quantity >= 200.0:
            return {
                "pack_type": "220kg Drum",
                "ghs_code": "ghs07_harmful",
                "ghs_symbol": "EXCLAMATION",
                "signal_word": "WARNING",
                "h_code": "H315: Causes skin irritation",
            }
        if quantity >= 150.0:
            return {
                "pack_type": "180kg Drum",
                "ghs_code": "ghs02_flammable",
                "ghs_symbol": "FLAME",
                "signal_word": "DANGER",
                "h_code": "H225: Highly flammable liquid and vapour",
            }
        if quantity >= 50.0:
            raise ValueError(
                "No GHS/package rule is defined for quantities from 50kg to 150kg"
            )
        if quantity >= 20.0:
            return {
                "pack_type": "25kg Pail",
                "ghs_code": "ghs08_hazardous_to_health",
                "ghs_symbol": "HEALTH_HAZARD",
                "signal_word": "WARNING",
                "h_code": "H335: May cause respiratory irritation",
            }
        return {
            "pack_type": "16kg Can",
            "ghs_code": "ghs05_corrosive",
            "ghs_symbol": "CORROSIVE",
            "signal_word": "DANGER",
            "h_code": "H314: Causes severe skin burns and eye damage",
        }


class EventLogParser:
    """Parse event logs and extract one label target per material lot."""

    def __init__(self, event_log_path: str | Path = DEFAULT_EVENT_LOG) -> None:
        self.event_log_path = Path(event_log_path)

    def load_events(self) -> list[dict[str, Any]]:
        """Load and minimally validate the event log's top-level structure."""
        try:
            with self.event_log_path.open("r", encoding="utf-8") as stream:
                events = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {self.event_log_path}: {exc}") from exc

        if not isinstance(events, list):
            raise ValueError("Event log must contain a JSON array")
        if not all(isinstance(event, dict) for event in events):
            raise ValueError("Every event log entry must be a JSON object")
        return events

    @staticmethod
    def build_po_quantities(events: list[dict[str, Any]]) -> dict[str, float]:
        """Build the representative package quantity for each PO material."""
        po_quantities: dict[str, float] = {}
        for event in events:
            if event.get("event_type") != "PURCHASE_ORDER":
                continue
            material_code = str(event.get("material_code") or "").strip()
            quantity = event.get("quantity_kg")
            if not material_code or quantity is None:
                continue
            try:
                # Keep the first PO as the representative value. Repeated POs in
                # this log describe replacement orders of the same package size.
                po_quantities.setdefault(material_code, float(quantity))
            except (TypeError, ValueError):
                continue
        return po_quantities

    @staticmethod
    def _is_label_event(event_type: str) -> bool:
        return (
            event_type.startswith("RECEIVE_")
            or event_type == "DISPATCH"
            or event_type.startswith("CHARGE_")
        )

    def extract_targets(
        self, events: list[dict[str, Any]] | None = None
    ) -> list[LabelTarget]:
        """Extract ordered, unique label targets from relevant lot events."""
        event_rows = self.load_events() if events is None else events
        po_quantities = self.build_po_quantities(event_rows)
        targets: list[LabelTarget] = []
        seen: set[tuple[str, str]] = set()

        for event in event_rows:
            event_type = str(event.get("event_type") or "")
            if not self._is_label_event(event_type):
                continue

            material_code = str(event.get("material_code") or "").strip()
            raw_material_name = str(event.get("material_name") or "").strip()
            material_name = re.sub(r"\s*\(.*?\)", "", raw_material_name).strip()
            lot_no = str(event.get("lot_no") or "").strip()
            key = (material_code, lot_no)
            if not material_code or not lot_no or key in seen:
                continue

            raw_quantity = po_quantities.get(material_code, event.get("quantity_kg"))
            try:
                quantity_kg = float(raw_quantity)
            except (TypeError, ValueError):
                continue

            event_status = str(
                event.get("event_status")
                or event.get("interlock_status")
                or event_type
            )
            targets.append(
                (material_code, material_name, lot_no, quantity_kg, event_status)
            )
            seen.add(key)

        return targets


class LabelRenderer:
    """Render high-contrast 600x400 synthetic labels."""

    WIDTH = 600
    HEIGHT = 400
    FONT_CANDIDATES = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    )
    BOLD_FONT_CANDIDATES = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    )

    def __init__(self, ghs_dir: str | Path = DEFAULT_GHS_ASSETS_DIR) -> None:
        self.ghs_dir = Path(ghs_dir)

    def _get_ghs_image(self, ghs_code: str) -> Image.Image | None:
        local_path = self.ghs_dir / f"{ghs_code.lower()}.png"
        if local_path.exists():
            try:
                img = Image.open(local_path).convert("RGBA")
                img.load()
                return img
            except Exception:
                return None
        return None

    @classmethod
    def _font(cls, size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        candidates = cls.BOLD_FONT_CANDIDATES if bold else cls.FONT_CANDIDATES
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _render_barcode(value: str) -> Image.Image:
        try:
            from barcode import Code128
            from barcode.writer import ImageWriter
        except ImportError as exc:
            raise RuntimeError(
                "python-barcode is required; install it with "
                "`python -m pip install python-barcode`"
            ) from exc

        buffer = BytesIO()
        Code128(value, writer=ImageWriter()).write(
            buffer,
            options={
                "module_width": 0.30,
                "module_height": 15.0,
                "quiet_zone": 2.0,
                "write_text": False,
                "dpi": 200,
            },
        )
        buffer.seek(0)
        barcode_image = Image.open(buffer).convert("RGB")
        barcode_image.load()
        return barcode_image

    @staticmethod
    def _fit_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
        scale = min(max_size[0] / image.width, max_size[1] / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.NEAREST)

    def render(self, target: LabelTarget) -> Image.Image:
        material_code, material_name, lot_no, quantity_kg, event_status = target
        rule = GHSRuleEngine.map_quantity(quantity_kg)

        canvas = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((5, 5, 594, 394), outline="black", width=4)

        title_font = self._font(28, bold=True)
        code_font = self._font(21, bold=True)
        body_font = self._font(17)
        small_font = self._font(14)
        signal_font = self._font(22, bold=True)
        lot_font = self._font(25, bold=True)

        title = material_name or material_code
        while len(title) > 4 and draw.textlength(title, font=title_font) > 430:
            title = title[:-1]
        if title != (material_name or material_code):
            title = title.rstrip() + "..."
        draw.text((18, 14), title, fill="black", font=title_font)
        draw.text((18, 51), f"RM CODE: {material_code}", fill="black", font=code_font)
        draw.line((15, 80, 585, 80), fill="black", width=2)

        barcode_image = self._fit_image(self._render_barcode(material_code), (280, 120))
        barcode_x = 20 + (280 - barcode_image.width) // 2
        barcode_y = 103 + (120 - barcode_image.height) // 2
        canvas.paste(barcode_image, (barcode_x, barcode_y))
        draw.text((20, 230), material_code, fill="black", font=code_font)

        # Real GHS pictogram image (PNG already contains the red diamond border)
        # 픽토그램 중심 좌표 및 크기 확대 (95x95 -> 130x130)
        cx, cy = 445, 180
        ghs_img = self._get_ghs_image(rule.get("ghs_code", ""))
        if ghs_img is not None:
            ghs_sized = self._fit_image(ghs_img, (130, 130))
            paste_x = cx - ghs_sized.width // 2
            paste_y = cy - ghs_sized.height // 2
            if ghs_sized.mode == "RGBA":
                canvas.paste(ghs_sized, (paste_x, paste_y), mask=ghs_sized.split()[3])
            else:
                canvas.paste(ghs_sized, (paste_x, paste_y))

        # 신호어는 픽토그램 하단에 컴팩트하게 배치 (검은색 볼드)
        draw.text((405, 95), rule["signal_word"], fill="black", font=body_font)
        h_lines = textwrap.wrap(rule["h_code"], width=30)
        for index, line in enumerate(h_lines[:2]):
            draw.text((315, 250 + index * 18), line, fill="black", font=small_font)

        draw.line((15, 298, 585, 298), fill="black", width=2)
        draw.text((20, 310), f"LOT: {lot_no}", fill="black", font=lot_font)
        draw.text((380, 354), f"PACK: {rule['pack_type']}", fill="black", font=body_font)
        return canvas

    def save(self, target: LabelTarget, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.render(target).save(destination, format="PNG")
        return destination


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def _print_results(rows: list[tuple[LabelTarget, Path]]) -> None:
    headers = ("MATERIAL", "LOT", "QTY_KG", "STATUS", "FILE")
    table_rows = [
        (
            target[0],
            target[2],
            f"{target[3]:g}",
            target[4],
            str(path),
        )
        for target, path in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print(f"\nGenerated {len(rows)} label(s).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-log", type=Path, default=DEFAULT_EVENT_LOG, help="event log JSON path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="PNG output directory"
    )
    args = parser.parse_args(argv)

    targets = EventLogParser(args.event_log).extract_targets()
    renderer = LabelRenderer()
    results: list[tuple[LabelTarget, Path]] = []
    for target in targets:
        material_code, _, lot_no, _, _ = target
        filename = (
            f"label_{_safe_filename_part(material_code)}_"
            f"{_safe_filename_part(lot_no)}.png"
        )
        output_path = renderer.save(target, args.output_dir / filename)
        results.append((target, output_path))

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
