from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

POPULARITY_MODEL_PATH = MODELS_DIR / "popularity_model.joblib"
COLLABORATIVE_MODEL_PATH = (
    MODELS_DIR / "collaborative_filtering_model.joblib"
)


def load_recommendation_models():
    if not POPULARITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Popularity model not found. Run train_pipeline.py first."
        )

    if not COLLABORATIVE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Collaborative filtering model not found. "
            "Run train_pipeline.py first."
        )

    popularity_model = joblib.load(POPULARITY_MODEL_PATH)
    collaborative_model = joblib.load(COLLABORATIVE_MODEL_PATH)

    logger.info("Recommendation models loaded.")

    return popularity_model, collaborative_model


def recommend_popular_products(
    customer_id,
    popularity_model,
    collaborative_model,
    top_n=5
):
    rating_matrix = collaborative_model["rating_matrix"]
    product_metadata = popularity_model["product_metadata"]

    seen_products = set()

    if customer_id in rating_matrix.index:
        seen_products = set(
            rating_matrix.columns[
                rating_matrix.loc[customer_id].values > 0
            ]
        )

    recommendations = product_metadata[
        ~product_metadata["product_id"].isin(seen_products)
    ].copy()

    recommendations = recommendations.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )

    recommendations = recommendations.head(top_n).copy()

    recommendations["recommendation_score"] = (
        recommendations["average_rating"]
    )

    recommendations["recommendation_method"] = (
        "Popularity-based"
    )

    return recommendations[
        [
            "product_id",
            "product_name",
            "category",
            "average_rating",
            "rating_count",
            "recommendation_score",
            "recommendation_method"
        ]
    ].reset_index(drop=True)


def recommend_products(
    customer_id,
    top_n=5,
    popularity_model=None,
    collaborative_model=None
):
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if (
        popularity_model is None
        or collaborative_model is None
    ):
        popularity_model, collaborative_model = (
            load_recommendation_models()
        )

    rating_matrix = collaborative_model["rating_matrix"]
    product_ids = collaborative_model["product_ids"]
    item_similarity = collaborative_model["item_similarity"]
    product_metadata = collaborative_model["product_metadata"]

    # Unknown customers receive popular products.
    if customer_id not in rating_matrix.index:
        logger.info(
            "Customer %s not found. Using popularity recommendations.",
            customer_id
        )

        return recommend_popular_products(
            customer_id,
            popularity_model,
            collaborative_model,
            top_n
        )

    customer_ratings = rating_matrix.loc[customer_id]

    seen_products = set(
        customer_ratings[customer_ratings > 0].index
    )

    candidate_products = [
        product_id
        for product_id in product_ids
        if product_id not in seen_products
    ]

    recommendations = []

    for product_id in candidate_products:
        product_index = product_ids.index(product_id)

        similarities = item_similarity[product_index]
        ratings = customer_ratings.values

        # Use only products the customer has rated.
        valid_mask = ratings > 0

        similarities = similarities[valid_mask]
        rated_values = ratings[valid_mask]

        # Use positive similarities only.
        positive_mask = similarities > 0

        similarities = similarities[positive_mask]
        rated_values = rated_values[positive_mask]

        if (
            len(similarities) > 0
            and similarities.sum() > 0
        ):
            score = np.average(
                rated_values,
                weights=similarities
            )
            method = "Item-based collaborative filtering"
        else:
            # Fallback score if no useful similarity exists.
            product_row = product_metadata.loc[
                product_metadata["product_id"] == product_id
            ]

            if product_row.empty:
                continue

            score = float(
                product_row["average_rating"].iloc[0]
            )
            method = "Popularity fallback"

        recommendations.append({
            "product_id": product_id,
            "recommendation_score": float(score),
            "recommendation_method": method
        })

    if not recommendations:
        return recommend_popular_products(
            customer_id,
            popularity_model,
            collaborative_model,
            top_n
        )

    recommendation_df = pd.DataFrame(recommendations)

    recommendation_df = recommendation_df.merge(
        product_metadata,
        on="product_id",
        how="left"
    )

    recommendation_df = recommendation_df.sort_values(
        ["recommendation_score", "rating_count"],
        ascending=[False, False]
    ).head(top_n)

    recommendation_df = recommendation_df[
        [
            "product_id",
            "product_name",
            "category",
            "average_rating",
            "rating_count",
            "recommendation_score",
            "recommendation_method"
        ]
    ].reset_index(drop=True)

    logger.info(
        "Generated %s recommendations for customer %s.",
        len(recommendation_df),
        customer_id
    )

    return recommendation_df


if __name__ == "__main__":
    popularity_model, collaborative_model = (
        load_recommendation_models()
    )

    if collaborative_model["customer_ids"]:
        customer_id = collaborative_model["customer_ids"][0]

        recommendations = recommend_products(
            customer_id=customer_id,
            top_n=5,
            popularity_model=popularity_model,
            collaborative_model=collaborative_model
        )

        print("\nCustomer ID:", customer_id)
        print("\nRecommended Products:")
        print(recommendations.to_string(index=False))
    else:
        print("No customers are available in the trained model.")