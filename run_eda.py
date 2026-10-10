from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.eda import perform_eda
from src.logger import get_logger

logger = get_logger(__name__)


def run_eda_pipeline():
    logger.info("Starting EDA pipeline.")

    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    perform_eda(processed_df)

    logger.info("EDA pipeline finished successfully.")


if __name__ == "__main__":
    run_eda_pipeline()
