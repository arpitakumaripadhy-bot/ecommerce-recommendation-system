
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def train_models(df, output_dir=MODELS_DIR):
    logger.info("Starting model training.")

    df = df.copy()

    required_columns = [
        "customer_id",
        "product_id",
        "product_name",
        "category",
        "rating"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Aggregate repeated customer-product interactions.
    interactions = (
        df.groupby(
            ["customer_id", "product_id"],
            as_index=False
        )
        .agg(
            rating=("rating", "mean"),
            product_name=("product_name", "first"),
            category=("category", "first")
        )
    )

    # Build product metadata.
    product_metadata = (
        interactions.groupby("product_id", as_index=False)
        .agg(
            product_name=("product_name", "first"),
            category=("category", "first"),
            average_rating=("rating", "mean"),
            rating_count=("rating", "size")
        )
    )

    product_metadata["average_rating"] = (
        product_metadata["average_rating"].round(3)
    )

    # Popularity baseline.
    popularity_model = (
        product_metadata.sort_values(
            ["rating_count", "average_rating"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )

    # User-item rating matrix.
    rating_matrix = interactions.pivot(
        index="customer_id",
        columns="product_id",
        values="rating"
    ).fillna(0)

    # Item-based collaborative filtering.
    item_similarity = cosine_similarity(
        rating_matrix.T
    )

    item_similarity = np.nan_to_num(
        item_similarity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity_path = output_dir / "popularity_model.joblib"
    collaborative_path = (
        output_dir / "collaborative_filtering_model.joblib"
    )

    joblib.dump(
        {
            "method": "popularity",
            "product_metadata": product_metadata,
            "popularity_ranking": popularity_model
        },
        popularity_path
    )

    joblib.dump(
        {
            "method": "item_based_collaborative_filtering",
            "customer_ids": rating_matrix.index.tolist(),
            "product_ids": rating_matrix.columns.tolist(),
            "rating_matrix": rating_matrix,
            "item_similarity": item_similarity,
            "product_metadata": product_metadata
        },
        collaborative_path
    )

    logger.info("Popularity model saved to %s", popularity_path)
    logger.info(
        "Collaborative filtering model saved to %s",
        collaborative_path
    )
    logger.info("Model training completed.")

    return popularity_model, {
        "rating_matrix": rating_matrix,
        "item_similarity": item_similarity,
        "product_metadata": product_metadata
    }


if __name__ == "__main__":
    from src.data_ingestion import load_data
    from src.data_preprocessing import preprocess_data

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    train_models(processed_df)

    print("Model training completed successfully.")