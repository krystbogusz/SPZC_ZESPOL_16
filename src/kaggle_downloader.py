import os
import shutil
import kagglehub


class KaggleDatasetDownloader:
    def __init__(self, default_download_path="data/raw"):
        self.default_download_path = default_download_path

    def download_dataset(self, dataset_identifier, target_path=None):
        download_dir = target_path if target_path else self.default_download_path
        os.makedirs(download_dir, exist_ok=True)
        cached_path = kagglehub.dataset_download(dataset_identifier)
        for item in os.listdir(cached_path):
            source = os.path.join(cached_path, item)
            destination = os.path.join(download_dir, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        return download_dir