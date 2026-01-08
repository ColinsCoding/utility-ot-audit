from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CableRecord:
    """Represents a single cable from the drawing export (one CSV row)."""

    drawing_id: str
    sheet: str
    device_tag: str
    symbol: str
    cable_id: str
    from_device: str
    to_device: str
    location: str
    cable_length_m: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CableRecord":
        """Create a CableRecord from a dict (e.g., CSV row).

        Expected keys (case-insensitive, whitespace-insensitive):
            drawing_id, sheet, device_tag, symbol, cable_id, from_device, to_device, location, cable_length_m
        """
        norm = {str(k).strip().lower(): v for k, v in data.items()}

        def req(key: str) -> str:
            if key not in norm or norm[key] is None:
                raise KeyError(f"Missing required field '{key}'")
            val = str(norm[key]).strip()
            if not val:
                raise ValueError(f"Empty required field '{key}'")
            return val

        def as_float(key: str) -> float:
            if key not in norm or norm[key] is None:
                raise KeyError(f"Missing required field '{key}'")
            try:
                return float(str(norm[key]).strip())
            except Exception as e:
                raise ValueError(f"Invalid float for '{key}': {norm[key]!r}") from e

        return cls(
            drawing_id=req("drawing_id"),
            sheet=req("sheet"),
            device_tag=req("device_tag"),
            symbol=str(norm.get("symbol", "")).strip(),
            cable_id=req("cable_id"),
            from_device=req("from_device"),
            to_device=req("to_device"),
            location=str(norm.get("location", "")).strip(),
            cable_length_m=as_float("cable_length_m"),
        )


@dataclass(frozen=True)
class ValidationIssue:
    """A single audit finding."""

    severity: str  # "error" | "warning" | "info"
    issue_type: str
    message: str
    cable_id: str
    device_tag: Optional[str] = None
    drawing_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "type": self.issue_type,
            "message": self.message,
            "cable_id": self.cable_id,
            "device_tag": self.device_tag,
            "drawing_id": self.drawing_id,
        }
