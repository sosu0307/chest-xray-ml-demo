from pathlib import Path

# src/streamlit/
STREAMLIT_DIR = Path(__file__).resolve().parent
# Projekt-Root (eine Ebene über src)
PROJECT_ROOT = STREAMLIT_DIR.parent.parent

# Wichtige App-Ordner
ASSETS_DIR = STREAMLIT_DIR / "assets"
DATA_DIR = STREAMLIT_DIR / "data"
PAGES_DIR = STREAMLIT_DIR / "pages"
ARTIFACTS_DIR = STREAMLIT_DIR / "artifacts"

# Model-Artefakte
MODELS_RESULTS_DIR = PROJECT_ROOT / "models" / "results"
MODEL_PATH = MODELS_RESULTS_DIR / "resnet50_best.pt"
PROBS_CSV_PATH = MODELS_RESULTS_DIR / "resnet50_probs_all.csv"

# Borderline-Gallery
BORDERLINE_DIR = ASSETS_DIR / "borderline_by_class_040"
BORDERLINE_COVID_DIR = BORDERLINE_DIR / "covid"
BORDERLINE_NONCOVID_DIR = BORDERLINE_DIR / "noncovid"

# Optionaler Fallback: echte Test-Images
TEST_IMAGES_DIR = PROJECT_ROOT / "data_bin" / "test" / "images"

# Optional: Sample Labels (falls bei dir vorhanden)
SAMPLE_LABELS_CSV = DATA_DIR / "sample_labels.csv"
