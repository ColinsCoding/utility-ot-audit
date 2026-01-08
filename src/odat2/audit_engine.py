from typing import List

from odat2.io.csv_reader import CSVReader
from odat2.models import ValidationIssue
from odat2.validators import CableValidator, DeviceValidator, CablePhysicsValidator, NetworkTopologyValidator


class AuditEngine:
    """Orchestrates the validation process."""

    def __init__(self):
        self.cable_validator = CableValidator()
        self.device_validator = DeviceValidator()
        self.physics_validator = CablePhysicsValidator()
        self.topology_validator = NetworkTopologyValidator()

    def audit(self, filepath: str) -> List[ValidationIssue]:
        """Run complete audit on a CSV file."""
        records = CSVReader(filepath).read()

        all_issues: List[ValidationIssue] = []

        # Record-level validators
        for record in records:
            all_issues.extend(self.cable_validator.validate(record))
            all_issues.extend(self.device_validator.validate(record))
            all_issues.extend(self.physics_validator.validate(record))

        # Dataset-level validators
        all_issues.extend(self.topology_validator.validate(records))

        return all_issues
