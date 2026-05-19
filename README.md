# URL Phishing Detection - Machine Learning Classification

A comprehensive machine learning project for detecting phishing URLs using multiple classification algorithms and URL feature analysis.

## Project Overview

This project implements and compares multiple machine learning models to classify URLs as legitimate or phishing. The system extracts 15 distinctive URL features and trains 5 different classification models to evaluate their performance on balanced and imbalanced datasets.

### Key Features:
- **URL Feature Extraction**: 15 engineered features from URLs including hostname properties, path structure, and URL format characteristics
- **Multiple Models**: Logistic Regression, Naive Bayes, Random Forest, SVM (LinearSVC), and XGBoost
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and feature importance analysis
- **Kaggle Integration**: Automatic dataset download from Kaggle
- **Automated Pipeline**: Complete end-to-end workflow from data preprocessing to model evaluation

## Project Structure

```
SPZC_ZESPOL_16/
├── main.py                                              # Main entry point script
├── config.py                                            # Configuration and path definitions
├── requirements.txt                                     # Python package dependencies
├── data/
│   ├── raw/                                             # Raw downloaded datasets
│   │   ├── final_dataset_with_all_features_v3.1.csv
│   │   ├── malicious_phish.csv
│   │   └── balanced_urls.csv
│   └── processed/                                       # Processed datasets ready for training
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py                                 # Dataset download and preprocessing
│   ├── feature_extractor.py                             # URL feature extraction logic
│   ├── training_utils.py                                # Training utilities (data loading, splitting)
│   ├── evaluate.py                                      # Model evaluation functions
│   ├── plots.py                                         # Visualization functions
│   ├── timer.py                                         # Performance timing utilities
│   ├── measure_features_times.py                        # Feature extraction performance testing
│   ├── kaggle_downloader.py                             # Kaggle dataset integration
│   └── models/                                          # Machine learning models
│       ├── logistic_regression.py
│       ├── naive_bayes.py
│       ├── random_forest.py
│       ├── svm.py
│       └── xgboost_model.py
└── runs/
    ├── features_extraction_times/                       # Performance timing for feature extraction
    │   ├── times.png
    │   └── times.txt    
    └── {model_name}_run_{timestamp}/                    # Results for each model run
        ├── metrics.json
        ├── classification_report.txt
        ├── confusion_matrix.png
        ├── {model_name}_feature_importance.png
        └── {model_name}_feature_importance_using_permutation_importance.png
```

## Extracted URL Features

The project extracts the following 15 features from each URL:

- **url_to_path_length**: Length of URL up to path component
- **has_ip**: Binary flag indicating if hostname is an IP address
- **hostname_length**: Length of the hostname
- **has_www**: Binary flag for 'www' prefix in hostname
- **has_tld**: Binary flag for valid top-level domain
- **has_decimal_in_hostname**: Binary flag for numeric characters in hostname
- **path_length**: Length of URL path
- **num_subdirectories**: Number of subdirectory levels in path
- **longest_subdirectory_length**: Length of the longest subdirectory
- **has_date_in_path**: Binary flag for date patterns in path
- **has_hex_in_path**: Binary flag for hexadecimal values in path
- **has_at_symbol**: Binary flag for '@' symbol in URL
- **num_dots_in_hostname**: Count of dots in hostname
- **has_hyphen_in_hostname**: Binary flag for hyphens in hostname
- **has_port**: Binary flag for explicit port specification

## Machine Learning Models

The project trains and evaluates the following classification models:

- **Logistic Regression**: Fast baseline linear model
- **Naive Bayes**: Probabilistic classifier
- **Random Forest**: Ensemble tree-based model
- **Support Vector Machine (SVM)**: LinearSVC kernel with class weights
- **XGBoost**: Gradient boosting ensemble

Each model is trained with default parameters and evaluated using:
- Classification metrics (Precision, Recall, F1-Score, Accuracy)
- Confusion matrices
- Feature importance analysis

## Installation

### Prerequisites
- Python 3.12
- pip package manager

### Setup Instructions

1. **Clone and navigate to the project directory**:
   ```bash
   git clone git@github.com:krystbogusz/SPZC_ZESPOL_16.git
   cd SPZC_ZESPOL_16
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

Execute the main script to run the entire workflow:

```bash
python main.py
```

This will:
1. Download datasets from Kaggle
2. Process raw datasets and prepare them for training
3. Extract features from all URLs
4. Train all 5 machine learning models
5. Generate evaluation metrics and visualizations
6. Save all results to the `runs/` directory

### Output

After execution, results are organized in timestamped directories under `runs/`:
- `features_extraction_times/`: Performance timings for feature extraction
- `{model_name}_run_{YYYY-MM-DD_HH-MM-SS}/`
  - **metrics.json**: Key performance metrics (accuracy, precision, recall, F1, etc.)
  - **classification_report.txt**: Detailed classification metrics per class
  - **confusion_matrix.png**: Visual confusion matrix heatmap
  - **{model_name}_feature_importance.png**: Model-specific feature importance
  - **{model_name}_feature_importance_using_permutation_importance.png**: Permutation-based importance

## Configuration

Edit `config.py` to customize:

```python
PROJ_ROOT               # Project root directory
DATA_DIR                # Data storage location
RAW_DATA_DIR            # Raw datasets location
PROCESSED_DATA_DIR      # Processed datasets location
FINAL_DATASET           # Path to final processed dataset
RUNS_DIR                # Results output directory
N_REPEATS               # Number of repetitions for generating permutation based feature importance
```

## Dependencies

Key dependencies include:
- **scikit-learn**: Machine learning models and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **xgboost**: Gradient boosting implementation
- **matplotlib**: Data visualization
- **kagglehub**: Kaggle dataset integration
- **requests**: HTTP library for data downloads

See `requirements.txt` for the complete list of dependencies.

## Dataset Information

The project uses the following public benchmark datasets:

- [Malicious URL Detection Dataset (Enhanced 2026)](https://www.kaggle.com/datasets/moutasmtamimi/malicious-url-detection-dataset-enhanced-2026)
- [Benign and Malicious URLs](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls)
- [Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

All datasets contain URLs with labels: benign, defacement, malware, phishing. The project processes and combines these datasets to create a comprehensive training set.

## Model Evaluation Approach

- **Train-Test Split**: 80-20 split with stratified sampling
- **Class Weights**: Balanced class weights for handling imbalanced data
- **Feature Importance**: Both permutation importance and model-specific importance calculated

## Performance Results

Results from the latest model runs are stored in the `runs/` directory. Each model generates:
- Individual classification reports
- Confusion matrices for visual assessment
- Feature importance rankings
- Overall metrics comparison across models

## Authors

- Krystian Bogusz
- Dominika Boguszewska
