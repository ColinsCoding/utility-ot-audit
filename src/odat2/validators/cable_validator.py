import re
from typing import List

from odat2.models import CableRecord, ValidationIssue


class CableValidator:
    """Validates cable-related data."""

    def __init__(self, max_length_m: float = 1000.0, suspicious_length_m: float = 9999.0):
        self.max_length_m = max_length_m
        self.suspicious_length_m = suspicious_length_m
        # Common placeholder patterns engineers use in exports
        self.placeholder_ids = re.compile(r"^(TBD|UNKNOWN|XX+|\?+)$", re.IGNORECASE)

    def validate(self, record: CableRecord) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Cable ID sanity
        if self.placeholder_ids.match(record.cable_id.strip()):
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="placeholder_cable_id",
                    message=f"Cable ID '{record.cable_id}' looks like a placeholder.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )

        # Length checks
        if record.cable_length_m < 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="invalid_cable_length",
                    message=f"Cable length cannot be negative: {record.cable_length_m} m",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )
            return issues

        if record.cable_length_m >= self.suspicious_length_m:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="suspicious_cable_length",
                    message=f"Cable length {record.cable_length_m} m appears to be a placeholder value.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )
        elif record.cable_length_m > self.max_length_m:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="long_cable",
                    message=f"Cable length {record.cable_length_m} m exceeds typical maximum {self.max_length_m} m. "
                    "Verify routing, units, and drawing export settings.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )

        # Self-loop wiring (often a data error)
        if record.from_device.strip() == record.to_device.strip():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="self_loop_connection",
                    message=f"Cable connects a device to itself ({record.from_device}). Verify from/to mapping.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )

        return issues
