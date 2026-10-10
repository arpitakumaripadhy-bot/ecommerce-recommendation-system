
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUALS_DIR = PROJECT_ROOT / "artifacts" / "visuals"


def split_interactions(df, test_size=0.2, random_state=42):
    train_parts = []
    test_parts = []

    for _, group in df.groupby("customer_id"):
        if len(group) < 2:
            train_parts.append(group)
            continue

        train_part, test_part = train_test_split(
            group,
            test_size=test_size,
            random_state=random_state
        )

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    if test_parts:
        test_df = pd.concat(
            test_parts,
            ignore_index=True
        )
    else:
        test_df = pd.DataFrame(columns=df.columns)

    return train_df, test_df


def prepare_models(train_df):
    interactions = (
        train_df.groupby(
            ["customer_id", "product_id"],
            as_index=False
        )
        .agg(
            rating=("rating", "mean"),
            product_name=("product_name", "first"),
            category=("category", "first")
        )
    )

    product_stats = (
        interactions.groupby("product_id", as_index=False)
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "size"),
            product_name=("product_name", "first"),
            category=("category", "first")
        )
    )

    rating_matrix = interactions.pivot(
        index="customer_id",
        columns="product_id",
        values="rating"
    ).fillna(0)

    item_similarity = cosine_similarity(
        rating_matrix.T
    )

    item_similarity = np.nan_to_num(
        item_similarity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    global_mean = (
        float(interactions["rating"].mean())
        if not interactions.empty
        else 3.0
    )

    return {
        "interactions": interactions,
        "product_stats": product_stats,
        "rating_matrix": rating_matrix,
        "item_similarity": item_similarity,
        "global_mean": global_mean
    }


def predict_rating(customer_id, product_id, model_data):
    rating_matrix = model_data["rating_matrix"]
    item_similarity = model_data["item_similarity"]

    if (
        customer_id not in rating_matrix.index
        or product_id not in rating_matrix.columns
    ):
        return model_data["global_mean"]

    customer_ratings = rating_matrix.loc[customer_id].values
    product_index = rating_matrix.columns.get_loc(product_id)

    similarities = item_similarity[product_index]

    valid_mask = customer_ratings > 0
    valid_mask[product_index] = False

    similarities = similarities[valid_mask]
    ratings = customer_ratings[valid_mask]

    positive_mask = similarities > 0
    similarities = similarities[positive_mask]
    ratings = ratings[positive_mask]

    if len(similarities) == 0 or similarities.sum() == 0:
        return model_data["global_mean"]

    prediction = np.average(
        ratings,
        weights=similarities
    )

    return float(np.clip(prediction, 1, 5))


def recommend_popular(customer_id, model_data, top_n=5):
    product_stats = model_data["product_stats"]
    rating_matrix = model_data["rating_matrix"]

    seen_products = set()

    if customer_id in rating_matrix.index:
        seen_products = set(
            rating_matrix.columns[
                rating_matrix.loc[customer_id].values > 0
            ]
        )

    recommendations = product_stats[
        ~product_stats["product_id"].isin(seen_products)
    ].copy()

    recommendations = recommendations.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )

    return recommendations.head(top_n)


def recommend_collaborative(customer_id, model_data, top_n=5):
    product_stats = model_data["product_stats"]
    rating_matrix = model_data["rating_matrix"]

    if customer_id not in rating_matrix.index:
        return recommend_popular(
            customer_id,
            model_data,
            top_n
        )

    customer_ratings = rating_matrix.loc[customer_id]

    seen_products = set(
        customer_ratings[customer_ratings > 0].index
    )

    candidates = product_stats[
        ~product_stats["product_id"].isin(seen_products)
    ].copy()

    candidates["recommendation_score"] = candidates[
        "product_id"
    ].apply(
        lambda product_id: predict_rating(
            customer_id,
            product_id,
            model_data
        )
    )

    candidates = candidates.sort_values(
        ["recommendation_score", "rating_count"],
        ascending=[False, False]
    )

    return candidates.head(top_n)


def calculate_ranking_metrics(
    test_df,
    model_data,
    top_n=5,
    relevance_threshold=4
):
    precisions = []
    recalls = []

    product_ids = set(
        model_data["product_stats"]["product_id"]
    )

    for customer_id, group in test_df.groupby("customer_id"):
        relevant_products = set(
            group.loc[
                group["rating"] >= relevance_threshold,
                "product_id"
            ]
        )

        relevant_products &= product_ids

        if not relevant_products:
            continue

        recommendations = recommend_collaborative(
            customer_id,
            model_data,
            top_n=top_n
        )

        recommended_products = set(
            recommendations["product_id"]
        )

        hits = len(
            relevant_products & recommended_products
        )

        precisions.append(hits / top_n)
        recalls.append(hits / len(relevant_products))

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))


def evaluate_models(df):
    logger.info("Starting model evaluation.")

    # One rating per customer-product pair avoids duplicated
    # pairs appearing on both sides of the split.
    df = (
        df.groupby(
            ["customer_id", "product_id"],
            as_index=False
        )
        .agg(
            rating=("rating", "mean"),
            product_name=("product_name", "first"),
            category=("category", "first"),
            timestamp=("timestamp", "max")
        )
    )

    train_df, test_df = split_interactions(df)

    print("Unique customer-product pairs:", len(df))
    print("Training interactions:", len(train_df))
    print("Testing interactions:", len(test_df))

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty. More interactions per customer are needed."
        )

    model_data = prepare_models(train_df)

    # Rating prediction metrics for the test interactions.
    y_true = []
    y_pred = []

    for row in test_df.itertuples(index=False):
        if row.product_id not in model_data["rating_matrix"].columns:
            continue

        y_true.append(row.rating)
        y_pred.append(
            predict_rating(
                row.customer_id,
                row.product_id,
                model_data
            )
        )

    if not y_true:
        raise ValueError(
            "No test products were available in the training data."
        )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Ranking metrics.
    precision, recall = calculate_ranking_metrics(
        test_df,
        model_data,
        top_n=5
    )

    # Evaluate popularity recommendations separately.
    popularity_precisions = []
    popularity_recalls = []

    train_products = set(
        model_data["product_stats"]["product_id"]
    )

    for customer_id, group in test_df.groupby("customer_id"):
        relevant_products = set(
            group.loc[
                group["rating"] >= 4,
                "product_id"
            ]
        ) & train_products

        if not relevant_products:
            continue

        recommendations = recommend_popular(
            customer_id,
            model_data,
            top_n=5
        )

        recommended_products = set(
            recommendations["product_id"]
        )

        hits = len(
            relevant_products & recommended_products
        )

        popularity_precisions.append(hits / 5)
        popularity_recalls.append(
            hits / len(relevant_products)
        )

    popularity_precision = (
        float(np.mean(popularity_precisions))
        if popularity_precisions else 0.0
    )

    popularity_recall = (
        float(np.mean(popularity_recalls))
        if popularity_recalls else 0.0
    )

    results = pd.DataFrame([
        {
            "Model": "Popularity baseline",
            "MAE": np.nan,
            "RMSE": np.nan,
            "Precision@5": popularity_precision,
            "Recall@5": popularity_recall
        },
        {
            "Model": "Item-based collaborative filtering",
            "MAE": mae,
            "RMSE": rmse,
            "Precision@5": precision,
            "Recall@5": recall
        }
    ])

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        VISUALS_DIR / "model_evaluation_results.csv",
        index=False
    )

    print("\nEVALUATION RESULTS")
    print(results.round(4).to_string(index=False))

    logger.info("Model evaluation completed.")

    return results


if __name__ == "__main__":
    from src.data_ingestion import load_data
    from src.data_preprocessing import preprocess_data

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    evaluate_models(processed_df)