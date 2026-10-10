from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models
from src.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline():
    logger.info("Starting recommendation system training pipeline.")

    # Step 1: Load raw data.
    raw_df = load_data()

    # Step 2: Clean and save processed data.
    processed_df = preprocess_data(raw_df)

    # Step 3: Train and save recommendation models.
    train_models(processed_df)

    # Step 4: Evaluate recommendation performance.
    evaluation_results = evaluate_models(processed_df)

    logger.info("Training pipeline completed successfully.")

    return evaluation_results


if __name__ == "__main__":
    results = run_training_pipeline()

    print("\nPipeline completed.")
    print(results.to_string(index=False))