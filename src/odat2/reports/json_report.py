import json
from typing import List
from pathlib import Path
from odat2.models import ValidationIssue

class JSONReporter:
    """Generates JSON audit reports"""
    
    def generate(self, issues: List[ValidationIssue], output_path: str):
        """Write issues to JSON file"""
        report = {
            "total_issues": len(issues),
            "errors": len([i for i in issues if i.severity == "error"]),
            "warnings": len([i for i in issues if i.severity == "warning"]),
            "issues": [issue.to_dict() for issue in issues]
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
            
        return report
