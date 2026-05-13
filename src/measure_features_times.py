import glob
import os
import urllib.parse

import matplotlib.pyplot as plt
import pandas as pd

from config import RAW_DATA_DIR, RUNS_DIR
from feature_extractor import FeatureExtractor
from timer import Timer


def mian():
    csv_files = glob.glob(str(RAW_DATA_DIR / "*.csv"))
    dataframes = {}

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            if "url" not in df.columns:
                if not df.empty and "url" in df.iloc[0].values:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)

            if "url" in df.columns:
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                dataframes[dataset_name] = df
        except Exception:
            pass

    if not dataframes:
        return

    extractor = FeatureExtractor()
    results_dict = {}

    methods = {
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
        "has_at_symbol": extractor.has_at_symbol,
        "num_dots_in_hostname": extractor.get_num_dots_in_hostname,
        "has_hyphen_in_hostname": extractor.has_hyphen_in_hostname,
        "has_port": extractor.has_port,
    }

    dataset_names = list(dataframes.keys())
    mapping = {name: f"Dataset {i+1}" for i, name in enumerate(dataset_names)}
    legend = "\n".join([f"Dataset {i+1}: {name}" for i, name in enumerate(dataset_names)])
    title = "Execution Time for Feature Extraction [seconds]"

    for dataset_name, df in dataframes.items():
        times = {key: 0.0 for key in methods.keys()}
        urls = df["url"].dropna().astype(str).tolist()

        for url in urls:
            parsed = urllib.parse.urlparse(url)

            for name, func in methods.items():
                arg = url if name == "has_at_symbol" else parsed
                timer = Timer(logger=None)
                timer.start()
                func(arg)
                times[name] += timer.stop()

        results_dict[mapping[dataset_name]] = times

    if results_dict:
        results_df = pd.DataFrame(results_dict)
        results_df["Average"] = results_df.mean(axis=1)
        results_df = results_df.sort_values(by="Average")

        print(f"\n{title}\n")
        print(results_df.to_string())
        print(f"\nLegend:\n{legend}")

        output_dir = RUNS_DIR / "features_extraction_times"
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_path = output_dir / "times.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n")
            f.write(results_df.to_string())
            f.write(f"\n\nLegend:\n{legend}")

        png_path = output_dir / "times.png"
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis("tight")
        ax.axis("off")

        ax.set_title(title, fontsize=20, pad=5, fontweight='bold')

        table = ax.table(
            cellText=results_df.round(6).values,
            colLabels=results_df.columns,
            rowLabels=results_df.index,
            loc="center",
            cellLoc="center",
        )

        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(fontweight="bold")

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.5)

        plt.subplots_adjust(bottom=0.2)
        plt.figtext(0.1, 0.05, f"Legend:\n{legend}", fontsize=14, ha="left")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    mian()