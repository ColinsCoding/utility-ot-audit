import re
from typing import List

from odat2.models import CableRecord, ValidationIssue


class DeviceValidator:
    """Validates device tag naming conventions and basic referential sanity."""

    def __init__(self):
        # Example: SS-COMM-001
        self.tag_pattern = re.compile(r"^[A-Z]{2}-[A-Z]{3,6}-\d{3}$")
        self.bad_tokens = re.compile(r"^(TBD|UNKNOWN|NONE|N/A)$", re.IGNORECASE)

    def _check_tag(self, tag: str, field: str, record: CableRecord) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        if self.bad_tokens.match(tag.strip()):
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type=f"placeholder_{field}",
                    message=f"{field.replace('_', ' ').title()} '{tag}' looks like a placeholder.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )
            return issues

        if not self.tag_pattern.match(tag.strip()):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type=f"nonstandard_{field}",
                    message=f"{field.replace('_', ' ').title()} '{tag}' does not match expected format (e.g., SS-COMM-001). "
                    "This may be valid, but it's worth checking.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )
        return issues

    def validate(self, record: CableRecord) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        issues.extend(self._check_tag(record.device_tag, "device_tag", record))
        issues.extend(self._check_tag(record.from_device, "from_device", record))
        issues.extend(self._check_tag(record.to_device, "to_device", record))

        # Missing/blank location can break construction packs
        if not record.location.strip():
            issues.append(
                ValidationIssue(
                    severity="info",
                    issue_type="missing_location",
                    message="Location field is empty; consider populating for construction/field crews.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )

        return issues
