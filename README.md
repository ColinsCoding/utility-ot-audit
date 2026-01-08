# ODAT2 - Operations Data Audit Tool

A command-line tool for validating telecommunications cable design data exported from engineering drawings.

## Features

- **Cable Length Validation**: Flag suspiciously long cables (>1000m)
- **Device Tag Validation**: Catch malformed device identifiers
- **Connection Integrity**: Verify device references exist
- **Drawing Cross-Reference**: Ensure devices appear on correct sheets
- **Multiple Output Formats**: JSON and HTML reports

## Installation
```bash
pip install -e .
```

## Usage
```bash
odat2 audit sample.csv --out-json report.json
odat2 audit sample.csv --out-html report.html
```

## Relevant for Utility Telecommunications

Designed for validating substation telecommunications infrastructure data:
- AutoCAD drawing exports
- Cable schedule verification
- Material management accuracy
- Quality control before construction

## Development
```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black src/
```
