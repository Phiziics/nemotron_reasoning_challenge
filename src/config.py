from pathlib import Path


class ProjectConfig:
    """
    Central project paths for the NVIDIA Nemotron reasoning challenge.
    Keeping paths in one place improves reproducibility and prevents messy notebooks.
    """

    ROOT_DIR = Path(__file__).resolve().parents[1]

    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

    OUTPUTS_DIR = ROOT_DIR / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    ADAPTERS_DIR = OUTPUTS_DIR / "adapters"
    SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
    REPORTS_DIR = OUTPUTS_DIR / "reports"

    TRAIN_FILE = RAW_DATA_DIR / "train.csv"
    TEST_FILE = RAW_DATA_DIR / "test.csv"

    RANDOM_STATE = 42
    VALIDATION_SIZE = 0.15