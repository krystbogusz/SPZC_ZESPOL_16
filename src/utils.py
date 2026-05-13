import pandas as pd
from pathlib import Path
from kaggle_downloader import KaggleDatasetDownloader
from feature_extractor import FeatureExtractor
from config import PROCESSED_DATA_DIR


def download_datasets():
    downloader = KaggleDatasetDownloader()

    downloader.download_dataset(
        dataset_identifier="moutasmtamimi/malicious-url-detection-dataset-enhanced-2026"
    )

    downloader.download_dataset(
        dataset_identifier="sid321axn/malicious-urls-dataset"
    )

    downloader.download_dataset(
        dataset_identifier="samahsadiq/benign-and-malicious-urls"
    )


def process_raw_dataset(csv_path, url_column, label_column):
    extractor = FeatureExtractor()
    input_file = Path(csv_path)
    output_path = PROCESSED_DATA_DIR / input_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if url_column not in df.columns or label_column not in df.columns:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        if url_column not in df.columns or label_column not in df.columns:
            raise ValueError(
                f"Brak oczekiwanych nazw kolumn: "
                f"'{url_column}' oraz '{label_column}'."
            )

    label_counts = df.groupby(url_column)[label_column].transform('nunique')
    df = df[label_counts == 1].copy()

    features_df = df[url_column].apply(
        extractor.extract_features
    ).apply(pd.Series)

    final_df = pd.concat([df[[url_column, label_column]], features_df], axis=1)
    final_df = final_df.rename(
        columns={url_column: 'url', label_column: 'type'}
    )

    columns_to_save = ['url'] + list(features_df.columns) + ['type']
    final_df = final_df[columns_to_save]

    final_df.to_csv(output_path, index=False)

    return final_df