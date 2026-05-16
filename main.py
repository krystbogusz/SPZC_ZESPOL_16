from src import download_datasets, process_raw_dataset
from config import RAW_DATA_DIR


def main():

    download_datasets()

    process_raw_dataset(
        csv_path=RAW_DATA_DIR / "final_dataset_with_all_features_v3.1.csv",
        url_column="url",
        label_column="type",
    )
    process_raw_dataset(
        csv_path=RAW_DATA_DIR / "malicious_phish.csv",
        url_column="url",
        label_column="type",
    )
    process_raw_dataset(
        csv_path=RAW_DATA_DIR / "balanced_urls.csv",
        url_column="url",
        label_column="label",
    )


if __name__ == "__main__":
    main()
