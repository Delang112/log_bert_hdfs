from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "logbert"
OUTPUT_DIR = ROOT / "output"

WINDOWS_LOG_PATH = DATA_DIR / "windows.log"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"
LINE_PREDICTIONS_PATH = OUTPUT_DIR / "line_predictions.json"

OUTPUT_DIR.mkdir(exist_ok=True)

def ensure_paths():
    OUTPUT_DIR.mkdir(exist_ok=True)
