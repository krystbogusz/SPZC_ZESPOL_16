from src import (
    KaggleDatasetDownloader,
    FeatureExtractor
)
import time
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path


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

def process_and_measure(
    csv_path,
    url_column,
    label_column,
    extractor,
    output_dir="data/processed"
):
    input_file = Path(csv_path)
    output_path = Path(output_dir) / input_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    parsed_urls = df[url_column].apply(urlparse)
    execution_times = {}

    methods_parsed = {
        "url_to_path_length": extractor.get_url_to_path_length,
        "has_ip": extractor.has_ip,
        "hostname_length": extractor.get_hostname_length,
        "has_www": extractor.has_www,
        "has_tld": extractor.has_tld,
        "has_decimal_in_hostname": extractor.has_decimal_in_hostname,
        "path_length": extractor.get_path_length,
        "num_subdirectories": extractor.get_num_subdirectories,
        "longest_subdirectory_length": extractor.get_longest_sub_length,
        "has_date_in_path": extractor.has_date_in_path,
        "has_hex_in_path": extractor.has_hex_in_path,
        "num_dots_in_hostname": extractor.get_num_dots_in_hostname,
        "has_hyphen_in_hostname": extractor.has_hyphen_in_hostname,
        "has_port": extractor.has_port
    }

    methods_raw = {
        "has_at_symbol": extractor.has_at_symbol
    }

    feature_columns = []

    for feature, func in methods_parsed.items():
        start_time = time.perf_counter()
        df[feature] = parsed_urls.apply(func)
        end_time = time.perf_counter()
        execution_times[feature] = end_time - start_time
        feature_columns.append(feature)

    for feature, func in methods_raw.items():
        start_time = time.perf_counter()
        df[feature] = df[url_column].apply(func)
        end_time = time.perf_counter()
        execution_times[feature] = end_time - start_time
        feature_columns.append(feature)

    times_df = pd.DataFrame(
        list(execution_times.items()),
        columns=['feature', 'execution_time_seconds']
    )

    columns_to_save = [url_column] + feature_columns + [label_column]
    final_df = df[columns_to_save]

    final_df.to_csv(output_path, index=False)

    return final_df, times_df



def main():

    download_datasets()

    feature_extractor = FeatureExtractor()

    process_and_measure(
        csv_path='data/raw/final_dataset_with_all_features_v3.1.csv',
        url_column='url',
        label_column='type',
        extractor=feature_extractor
    )


if __name__ == '__main__':
    main()