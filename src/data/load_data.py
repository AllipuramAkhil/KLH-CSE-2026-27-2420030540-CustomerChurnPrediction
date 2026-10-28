"""Dataset discovery, (optional) download and loading."""
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

ALTERNATIVE_FILENAMES = ["Telco-Customer-Churn.csv", "telco_customer_churn.csv", "telco_churn.csv"]

MANUAL_INSTRUCTIONS = (
    "Dataset not found and automatic download failed.\n"
    "Please download the IBM Telco Customer Churn CSV manually\n"
    "  (Kaggle: 'blastchar/telco-customer-churn' or the IBM GitHub repository)\n"
    f"and place it here:\n  {config.RAW_DATA_PATH}\n"
)


def _looks_like_telco_csv(path: Path) -> bool:
    """Cheap sanity check so a wrong or truncated file is rejected early."""
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:  # noqa: BLE001 - any parse problem means "not valid"
        return False
    needed = {config.ID_COLUMN, config.TARGET_COLUMN, "tenure", "TotalCharges"}
    return needed.issubset(df.columns)


def find_dataset() -> Path | None:
    """Return the path of the raw CSV if it exists (any accepted file name)."""
    if config.RAW_DATA_PATH.exists() and _looks_like_telco_csv(config.RAW_DATA_PATH):
        return config.RAW_DATA_PATH
    for name in ALTERNATIVE_FILENAMES:
        candidate = config.RAW_DATA_DIR / name
        if candidate.exists() and _looks_like_telco_csv(candidate):
            return candidate
    for candidate in sorted(config.RAW_DATA_DIR.glob("*.csv")):
        if _looks_like_telco_csv(candidate):
            return candidate
    return None


def download_dataset() -> Path | None:
    """Try each public mirror; return the saved path or None on failure."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url in config.DATASET_URLS:
        try:
            logger.info("Downloading dataset from %s", url)
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                content = response.read()
            config.RAW_DATA_PATH.write_bytes(content)
            if _looks_like_telco_csv(config.RAW_DATA_PATH):
                logger.info("Dataset saved to %s", config.RAW_DATA_PATH)
                return config.RAW_DATA_PATH
            config.RAW_DATA_PATH.unlink(missing_ok=True)
            logger.warning("Downloaded file did not look like the Telco dataset.")
        except Exception as exc:  # noqa: BLE001 - network errors vary widely
            logger.warning("Download failed (%s): %s", url, exc)
    return None


def ensure_dataset() -> Path:
    """Locate the dataset, downloading it if necessary; raise a clear error otherwise."""
    path = find_dataset() or download_dataset()
    if path is None:
        raise FileNotFoundError(MANUAL_INSTRUCTIONS)
    return path


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as stored on disk (no cleaning)."""
    path = Path(path) if path else ensure_dataset()
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows x %d columns from %s", df.shape[0], df.shape[1], path.name)
    return df
