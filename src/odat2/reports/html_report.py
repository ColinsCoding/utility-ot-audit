from typing import List
from pathlib import Path
from odat2.models import ValidationIssue

class HTMLReporter:
    """Generates HTML audit reports"""
    
    def generate(self, issues: List[ValidationIssue], output_path: str):
        """Write issues to HTML file"""
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ODAT Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #ecf0f1; padding: 15px 20px; border-radius: 5px; flex: 1; }}
        .stat-number {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ color: #7f8c8d; font-size: 14px; }}
        .error {{ background: #ffe5e5; border-left: 4px solid #e74c3c; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #f39c12; }}
        .issue {{ margin: 15px 0; padding: 15px; border-radius: 5px; }}
        .issue-header {{ font-weight: bold; margin-bottom: 5px; }}
        .issue-details {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ODAT Audit Report</h1>
        <div class="summary">
            <div class="stat">
                <div class="stat-number">{len(issues)}</div>
                <div class="stat-label">Total Issues</div>
            </div>
            <div class="stat">
                <div class="stat-number">{len(errors)}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat">
                <div class="stat-number">{len(warnings)}</div>
                <div class="stat-label">Warnings</div>
            </div>
        </div>
        
        <h2>Errors</h2>
"""
        
        for issue in errors:
            html += f"""
        <div class="issue error">
            <div class="issue-header">{issue.issue_type.replace('_', ' ').title()}</div>
            <div>{issue.message}</div>
            <div class="issue-details">Cable: {issue.cable_id} | Device: {issue.device_tag} | Drawing: {issue.drawing_id}</div>
        </div>
"""
        
        html += "<h2>Warnings</h2>"
        
        for issue in warnings:
            html += f"""
        <div class="issue warning">
            <div class="issue-header">{issue.issue_type.replace('_', ' ').title()}</div>
            <div>{issue.message}</div>
            <div class="issue-details">Cable: {issue.cable_id} | Device: {issue.device_tag} | Drawing: {issue.drawing_id}</div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, "w") as f:
            f.write(html)
