import json
import subprocess
import sys
from pathlib import Path

def test_cli_smoke(tmp_path: Path):
    # Use the sample.csv in repo root
    sample = Path("sample.csv")
    assert sample.exists(), "sample.csv must exist at repo root for this smoke test"

    out_json = tmp_path / "report.json"

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path("src"))
    env["PYTHONIOENCODING"] = "utf-8"
    env["RICH_FORCE_TERMINAL"] = "0"


    cmd = [
        sys.executable,
        "src/odat2/cli.py",
        str(sample),
        "--out-json",
        str(out_json),
        "--today",
        "2026-01-08",
    ]

    r = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


    assert r.returncode == 0, r.stderr + "\n" + r.stdout
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "total_issues" in data
    assert "issues" in data
