# Network Intrusion Classifier Comparison 🔐
Year: 2024

This project compares multiple machine learning classifiers to predict and detect network intrusions, combining data mining techniques with advanced model evaluation.

## Project Overview
This project aims to:
    - Preprocess and clean network traffic data
    - Engineer features for better model performance
    - Train and evaluate multiple machine learning classifiers
    - Compare classifier performance based on accuracy, F1 score, and precision
    - Export predictions and model evaluation outputs for analysis

## Key Technical Highlights
    - Multiple classifiers implemented: K-Nearest Neighbors, Naive Bayes, Decision Trees, Random Forest, Gradient Boosting
    - Hyperparameter tuning with GridSearchCV and cross-validation
    - Handling of imbalanced data using SMOTE oversampling
    - Feature engineering including binarisation, one-hot encoding, and edit distance calculations
    - Automated export of confusion matrices and top predictions

## Features
### Data Preprocessing
    - Clean and prepare training and test datasets
    - Feature binarisation and boolean transformation
    - One-hot encoding of categorical variables
    - Feature engineering (edit distance, means, digit counting)

### Classification Models
    - Train classifiers using optimised parameters
    - Hyperparameter tuning option for each model
    - Cross-validation for performance validation
    - Model evaluation via confusion matrices, accuracy, F1, and precision metrics

### Outputs
    - Processed datasets exported in ARFF and CSV formats
    - Top two prediction sets for each test dataset
    - Saved confusion matrices for visual evaluation

## Dependencies 
arff, -U pandas, numpy, matplotlib, seaborn, sklearn.model_selection, sklearn.over_sampling, sklearn.under_sampling, sklearn.neighbors, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble, sklearn.metrics, sklearn.inspection,  imbalanced-learn, openpyxl, python-Levenshtein, warnings, os

## How to Run
1. Ensure all required files are in your working directory:
    - `run.py`
    - Supporting file: `Runpy_Functions.py`
    - Datasets: `Assignment-2024-training-data-set.xlsx`, `Test-Data-Set-1-2024.xlsx`, `Test-Data-Set-2-2024.xlsx`
2. Open and run `run.py`

## Additional Features
    - Option to toggle hyperparameter tuning on or off for faster execution
    - Parallel processing enabled for GridSearchCV and cross-validation (adjustable n_jobs setting)
    - Attack categories encoded for clean CSV prediction exports
        attack_cat_encoding = {
        "Analysis": 1,
        "Backdoor": 2,
        "Exploits": 3,
        "Fuzzers": 4,
        "Generic": 5,
        "Normal": 6,
        "Reconnaissance": 7,
        "Shellcode": 8,
        "Worms": 9
        }

## Important Notes for the User: 
Parallel Processing Warning!
This project uses n_jobs=-1 in certain functions (e.g., GridSearchCV (def hypertune_w_CV), cross_val_score (def cross_val_score x 2) and some classifiers (def classifier_parameter - kNN and Random Forest)) to enable parallel processing by utilising all available CPU cores. This may increase memory usage: Large datasets or models could cause memory overload, leading to system slowdowns or crashes on machines with limited RAM. If you encounter performance issues, consider reducing the value of n_jobs (e.g., n_jobs=2) to limit the number of cores used.

## Credits
Network data based on simulated intrusion detection datasets provided for academic use.


## Connect with Me
📫 [LinkedIn](https://www.linkedin.com/in/safflatters/)


## License and Usage
![Personal Use Only](https://img.shields.io/badge/Personal%20Use-Only-blueviolet?style=for-the-badge)

This project is intended for personal, educational, and portfolio purposes only.
You are welcome to view and learn from this work, but you may not copy, modify, or submit it as your own for academic, commercial, or credit purposes.