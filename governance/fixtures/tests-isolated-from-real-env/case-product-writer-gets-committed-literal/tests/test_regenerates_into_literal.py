from pathlib import Path
from aosr.physics import report_io

def test_regenerates():
    report_io.regenerate_schema_files(Path("blueprint/schemas"))
