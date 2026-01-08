import csv
from pathlib import Path
from typing import List

from odat2.models import CableRecord


class CSVReader:
    """Handles reading and parsing CSV files."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def read(self) -> List[CableRecord]:
        """Read CSV and return list of CableRecord objects."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {self.filepath}")

        records: List[CableRecord] = []
        with self.filepath.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):  # header is line 1
                try:
                    records.append(CableRecord.from_dict(row))
                except Exception as e:
                    # Keep going; bad row shouldn't kill the audit
                    print(f"Warning: skipping invalid row at line {idx}: {e}")

        return records
