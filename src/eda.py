from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISUALS_DIR = PROJECT_ROOT / "artifacts" / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(file_name):
    file_path = VISUALS_DIR / file_name

    plt.tight_layout()
    plt.savefig(
        file_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    logger.info("Saved visualization: %s", file_path)


def perform_eda(df):
    logger.info("Starting exploratory data analysis.")

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    summary = {
        "total_interactions": len(df),
        "unique_customers": df["customer_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "unique_categories": df["category"].nunique(),
        "average_rating": round(df["rating"].mean(), 2)
    }

    print("\nEDA SUMMARY")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    # 1. Rating distribution.
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="rating",
        order=sorted(df["rating"].dropna().unique())
    )
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Interactions")
    save_plot("rating_distribution.png")

    # 2. Top 10 products by interaction count.
    top_products = (
        df.groupby("product_name")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    top_products.plot(kind="barh")
    plt.title("Top 10 Products by Interactions")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Product")
    save_plot("top_10_products.png")

    print("\nTop 10 products:")
    print(top_products.sort_values(ascending=False))

    # 3. Category interaction counts.
    category_counts = (
        df["category"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    category_counts.plot(kind="bar")
    plt.title("Interactions by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Interactions")
    plt.xticks(rotation=35, ha="right")
    save_plot("category_interactions.png")

    print("\nCategory interactions:")
    print(category_counts)

    # 4. Average rating by category.
    category_ratings = (
        df.groupby("category")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    category_ratings.plot(kind="bar")
    plt.title("Average Rating by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=35, ha="right")
    save_plot("average_rating_by_category.png")

    print("\nAverage rating by category:")
    print(category_ratings.round(2))

    # 5. Customer activity distribution.
    customer_activity = df.groupby("customer_id").size()

    plt.figure(figsize=(9, 5))
    sns.histplot(
        customer_activity,
        bins=20,
        kde=True
    )
    plt.title("Customer Interaction Distribution")
    plt.xlabel("Interactions per Customer")
    plt.ylabel("Number of Customers")
    save_plot("customer_activity_distribution.png")

    print("\nCustomer activity summary:")
    print(customer_activity.describe())

    # 6. Monthly interactions.
    monthly_interactions = (
        df.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("MS")
        .size()
    )

    if not monthly_interactions.empty:
        plt.figure(figsize=(11, 5))
        monthly_interactions.plot(marker="o")
        plt.title("Monthly Interactions")
        plt.xlabel("Month")
        plt.ylabel("Number of Interactions")
        save_plot("monthly_interactions.png")

        monthly_interactions.rename(
            "interaction_count"
        ).to_csv(
            VISUALS_DIR / "monthly_interactions.csv",
            header=True
        )

    # Save summary report.
    pd.DataFrame(
        [summary]
    ).to_csv(
        VISUALS_DIR / "eda_summary.csv",
        index=False
    )

    logger.info("EDA completed successfully.")

    return summary


if __name__ == "__main__":
    from src.data_ingestion import load_data
    from src.data_preprocessing import preprocess_data

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    perform_eda(processed_df)