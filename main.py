from src import (
    download_datasets, process_raw_dataset,
    run_feature_extraction_test,
    run_lr,run_nb,run_rf,run_svm,run_xgboost
)
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

    run_feature_extraction_test()
    print("\n------------------------------------\n")
    run_lr()
    print("\n------------------------------------\n")
    run_nb()
    print("\n------------------------------------\n")
    run_rf()
    print("\n------------------------------------\n")
    run_svm()
    print("\n------------------------------------\n")
    run_xgboost()


if __name__ == "__main__":
    main()
