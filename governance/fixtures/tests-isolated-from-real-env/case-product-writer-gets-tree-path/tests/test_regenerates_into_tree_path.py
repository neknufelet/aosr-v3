from pathlib import Path
from aosr.physics import report_io

REPO = Path(__file__).resolve().parents[1]

def test_regenerates():
    report_io.regenerate_schema_files(directory=REPO / "blueprint" / "schemas")
