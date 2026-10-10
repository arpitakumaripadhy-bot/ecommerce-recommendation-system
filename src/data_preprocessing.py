from pathlib import Path
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "data"
    / "processed"
    / "interactions_processed.csv"
)

REQUIRED_COLUMNS = [
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "rating",
    "timestamp"
]


def preprocess_data(df, save_data=True):
    logger.info("Starting data preprocessing.")

    df = df.copy()

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

    initial_rows = len(df)

    # Remove records missing required values.
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Clean text fields.
    text_columns = [
        "customer_id",
        "product_id",
        "product_name",
        "category"
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    # Remove empty text values.
    for column in text_columns:
        df = df[df[column] != ""].copy()

    # Convert ratings to numeric.
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna(subset=["rating"]).copy()

    # Keep ratings between 1 and 5, inclusive.
    df = df[
        df["rating"].between(1, 5)
    ].copy()

    # Ratings should be whole numbers.
    df = df[
        df["rating"] % 1 == 0
    ].copy()

    df["rating"] = df["rating"].astype(int)

    # Parse the dataset's timestamp format.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"]).copy()

    # Remove exact duplicate records.
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    # Sort records chronologically within each customer.
    df = df.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)

    if save_data:
        PROCESSED_DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            PROCESSED_DATA_PATH,
            index=False
        )

        logger.info(
            "Processed data saved to %s",
            PROCESSED_DATA_PATH
        )

    logger.info("Data preprocessing completed.")
    logger.info("Initial rows: %s", initial_rows)
    logger.info("Final rows: %s", len(df))
    logger.info("Duplicate rows removed: %s", duplicate_count)
    logger.info("Final shape: %s", df.shape)

    return df


if __name__ == "__main__":
    from src.data_ingestion import load_data

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    print(processed_df.head())
    print("Processed shape:", processed_df.shape)
