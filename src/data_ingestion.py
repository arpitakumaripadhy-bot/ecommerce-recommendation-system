from pathlib import Path
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "data"
    / "raw"
    / "ecommerce_recommendation_interactions.csv"
)

REQUIRED_COLUMNS = [
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "rating",
    "timestamp"
]


def load_data(file_path=RAW_DATA_PATH):
    logger.info("Starting data ingestion.")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip().str.lower()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("Dataset loaded successfully.")
    logger.info("Dataset shape: %s", df.shape)
    logger.info(
        "Unique customers: %s",
        df["customer_id"].nunique()
    )
    logger.info(
        "Unique products: %s",
        df["product_id"].nunique()
    )

    return df


if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print("Dataset shape:", data.shape)
