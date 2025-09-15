import os

# Default paths and constants with environment variable overrides
RUNS_LOCATION = os.getenv("RUNS_LOCATION", "./")
REPORT_FOLDER = os.getenv("REPORT_FOLDER", "./")
SAMPLE_LIST = os.getenv("SAMPLE_LIST", "709").split(",")
ALPHA1 = float(os.getenv("ALPHA1", "1.54056"))
ALPHA2 = float(os.getenv("ALPHA2", "1.54439"))
