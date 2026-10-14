from aosr.physics import report_io

def test_regenerates():
    report_io.regenerate_schema_files(report_io._SCHEMA_DIR)
