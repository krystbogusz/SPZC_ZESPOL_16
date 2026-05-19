from dataset_utils import download_datasets, process_raw_dataset
from models.logistic_regression import run_lr
from models.naive_bayes import run_nb
from models.random_forest import run_rf
from models.svm import run_svm
from models.xgboost_model import run_xgboost
from measure_features_times import run_feature_extraction_test