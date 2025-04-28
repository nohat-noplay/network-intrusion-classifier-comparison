# Network Intrusion Classifier Comparison
# Runpy_Functions.py
# Author: Saf Flatters 
# Year: 2024


# This is the functions used by run.py
# These functions are divided by:
#           - Preprocessing Functions
#           - Transformation Functions
#           - Feature Selection Functions
#           - Classifier Functions
#           - Model Evaluation Functions



#General Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arff
import openpyxl

# For Feature Engineering:
import Levenshtein as lev   #Edit Distance

# For Transformation:
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# For Training and Classification:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier,  RandomForestClassifier
import os

# For Evaluation: 
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
from sklearn.inspection import permutation_importance

# For future warnings: (regarding incompatable dtype when changing True False to 1 0)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#################################################################################################
#### Functions for Preprocessing

def one_hot_encode(dataframe, column):
    ohe = pd.get_dummies(dataframe[column], prefix=column, drop_first=False)

    return ohe


def calc_edit_distance(dataframe, column_i, column_j):
    edit_distance = dataframe.apply(lambda row: lev.distance(str(row[column_i]), str(row[column_j])), axis=1)

    return edit_distance


def make_bins(dataframe, column, values_to_be_1):
    # make sure it's an integer
    dataframe[column] = dataframe[column].astype(int)
    # if in [values_to_be_1] make 1, else make 0
    binned = np.where(dataframe[column].isin(values_to_be_1), True, False)
    
    return binned


def find_mean(dataframe, prefix, exclude_columns):
    filtered_df = dataframe.filter(like=prefix)
    # Exclude the specified columns
    filtered_df = filtered_df.drop(columns=exclude_columns, errors='ignore')
    # Calculate the mean of the filtered columns for each row
    collective_mean = filtered_df.mean(axis=1)
    
    return collective_mean
 

def make_boolean(dataframe, column, overthreshold):
    # make sure it's an integer
    dataframe[column] = dataframe[column].astype(int)
    # if column contains number above overthreshold, make 1, else make 0
    bool_column = np.where(dataframe[column] > overthreshold, True, False)

    return bool_column


#possible rate - categories: number of digits before decimal
def count_digits(dataframe, column):
    # Convert the column to integers
    dataframe[column] = dataframe[column].astype(int)
    # Calculate the number of digits for each number in the column
    number_of_digits = dataframe[column].apply(lambda x: len(str(abs(x))))
    
    return number_of_digits 


def export_arff(dataframe, filename, categorical_column):
    # Define ARFF attribute types for each column
    attribute_types = []
    for column in dataframe.columns:
        if column in categorical_column:
            unique_values = dataframe[column].unique()
            attribute_types.append((column, unique_values.tolist()))  # For categorical columns
        else:
            attribute_types.append((column, 'NUMERIC'))  # For numeric columns

    # Prepare the ARFF data
    relation_name = filename # Name of the dataset
    data = dataframe.values.tolist()  # The actual data

    # Save to ARFF file
    arff_filename = filename if filename.endswith('.arff') else filename + '.arff'  # Ensure .arff extension
    with open(arff_filename, 'w') as f:
        arff.dump(f.name, data, relation=relation_name, names=[col[0] for col in attribute_types])

    print("ARFF file saved as:", arff_filename)

  
#################################################################################################
#### Functions for Transformations

def balance_w_SMOTE(dataframe, target_col, target_size=12500, random_state=42):
    # Convert all boolean columns to integers
    binary_columns = dataframe.select_dtypes(include=['bool']).columns
    dataframe.loc[:, binary_columns] = dataframe.loc[:, binary_columns].astype(int)
    X = dataframe.drop(columns=[target_col])  # Features
    y = dataframe[target_col]  # Target variable
    smote = SMOTE(random_state=random_state,
                  sampling_strategy={k: target_size for k in y.value_counts().index if y.value_counts()[k] < target_size})
    undersampler = RandomUnderSampler(sampling_strategy={k: target_size for k in y.value_counts().index if y.value_counts()[k] > target_size},
                                      random_state=random_state)
    # Apply SMOTE to the minority classes
    X_oversampled, y_oversampled = smote.fit_resample(X, y)
    # Apply undersampling to the combined oversampled data
    X_balanced, y_balanced = undersampler.fit_resample(X_oversampled, y_oversampled)
    # Combine the features and target variable back into a DataFrame
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)

    return df_balanced
    

def find_numerical_columns(df):
# Try to convert each column to numeric, forcing non-convertible values to NaN
    numerical_columns = []
    for col in df.columns:
        # Attempt to convert the column to numeric
        converted_col = pd.to_numeric(df[col], errors='coerce')
        # If there are any non-NaN values after conversion, consider it as a numerical column
        if converted_col.notna().all():
            numerical_columns.append(col)

    return numerical_columns

# Function to standardise the dataset
def standardise_data(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    non_binary_columns = [col for col in numerical_columns if df[col].nunique() > 2] # filter out binary
    scaler = StandardScaler()
    df[non_binary_columns] = scaler.fit_transform(df[non_binary_columns])
    
    return scaler

# Standardise Test Sets
def standardise_test_data(test_df, scaler):
    non_binary_columns = test_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    non_binary_columns = [col for col in non_binary_columns if test_df[col].nunique() > 2]
    test_df.loc[:, non_binary_columns] = scaler.transform(test_df[non_binary_columns])


#################################################################################################
#### Functions for Feature Selection

def plot_permutation_importance(classifier_with_setted_params, X_test, y_test, test_type):
    result = permutation_importance(classifier_with_setted_params, X_test, y_test, n_repeats=30, random_state=42)
    importance = result.importances_mean
    # Display feature importances
    for i in range(len(importance)):
        print(f"Feature {i}: {importance[i]:.4f}")

    plt.barh(range(len(importance)), importance)
    plt.yticks(range(len(importance)), X_test.columns)  # Assuming X_test is a DataFrame
    plt.xlabel("Mean decrease in accuracy")
    plt.title(f"Permutation Feature Importances in {test_type} Classification")
    plt.draw()


#################################################################################################
#### Functions for Classification

def classifier_parameters(classifier):
    param_grid = {}
    try: 
        if classifier == 'K-Nearest Neighbour':
            param_grid = {
            'n_neighbors': np.arange(3, 30, 2),  # 3 - 17 (odd only)
            'weights': ['distance', 'uniform'],
            'metric': ['manhattan', 'euclidean'], 
            'p': [1, 2],
            'n_jobs': [-1]
                }

        elif classifier == 'Bernoulli Naive Bayes':
            param_grid = {
            # 'var_smoothing': [1e-9, 1e-5]  
                }

        elif classifier == 'Gaussian Naive Bayes':
            param_grid = {
            # 'var_smoothing': [1e-9, 1e-5]  
                } 

        elif classifier == 'Decision Tree':
            param_grid = {
            'criterion': ['gini', 'entropy'],  
            'max_depth': [ 10, 20, 30],
            'splitter': ['best', 'random']  
                }

        elif classifier == 'Random Forest':
            param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_leaf': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_jobs':[-1]
                }
        
        elif classifier == 'Gradient Boost': #this is HistGradientBoost
            param_grid = {
            'max_iter': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [5, 8], 
            'class_weight': [None, 'balanced']
                }
        
        else: raise ValueError(f"Invalid classifier type: '{classifier}'. Supported types: 'K-Nearest Neighbour', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boost'.")

    except ValueError as e:
            print(e)

            return {}
        
    return param_grid


def hypertune_w_CV(classifier_type, scoring, X_train, y_train):
    # Initialise the classifier based on the type
    if classifier_type == 'K-Nearest Neighbour':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'Bernoulli Naive Bayes':
        classifier = BernoulliNB()   #Gaussian or Bernoulli 
    elif classifier_type == 'Gaussian Naive Bayes':
        classifier = GaussianNB()
    elif classifier_type == 'Decision Tree':
        classifier = DecisionTreeClassifier()
    elif classifier_type == 'Random Forest':
        classifier = RandomForestClassifier()
    elif classifier_type == 'Gradient Boost':
        classifier = HistGradientBoostingClassifier()
    elif classifier_type == 'True Gradient Boost':
        classifier = GradientBoostingClassifier()
    else:
        raise ValueError(f"Invalid classifier type: '{classifier_type}'.")

    param_grid = classifier_parameters(classifier_type)
    # Set up Stratified K-Fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)        #changed n_splits to 5 from 10 for speed
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=skf, n_jobs=-1, scoring=scoring)
    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)
    # Get the best parameters and best accuracy
    best_params = grid_search.best_params_
    print(f"\n\n{classifier_type} Best Parameters: {grid_search.best_params_}")
    best_score = grid_search.best_score_
    print(f"Best GridSearchCV Score: {grid_search.best_score_:.4f}")

    return best_params


def precision_score_for_normal(y_test, y_pred):     # determining precision of normal
    precision_labels = precision_score(y_test, y_pred, average=None, zero_division=1) # zero division to stop warning output
    normal_precision = round(precision_labels[5], 4)  # True Pos of Normal / True Pos of Normal + False Pos of Normal - Normal MUST be 6th label

    return normal_precision


def cross_validate(best_from_GridSearch, X_train, y_train, scoring):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)     #changed n_splits to 5 from 10 for speed
    #cross validate with accuracy
    cv_results = cross_val_score(best_from_GridSearch, X_train, y_train, cv=skf, n_jobs=-1, scoring=scoring)
    mean_accuracy = cv_results.mean()
    std_accuracy = cv_results.std()
    # cross validate with custom made scorer for Precision Normal
    normal_precision_scorer = make_scorer(precision_score_for_normal)
    cv_normal_precision = cross_val_score(best_from_GridSearch, X_train, y_train, cv=skf, n_jobs=-1, scoring=normal_precision_scorer)
    # Calculate mean and standard deviation for Normal Precision
    mean_normal_precision = cv_normal_precision.mean()
    std_normal_precision = cv_normal_precision.std()
    print(f"10 Fold Cross Validation: Mean of Accuracy: {mean_accuracy:.4f}")
    print(f"10 Fold Cross Validation: Standard Deviation of Accuracy: {std_accuracy:.4f}")
    print(f"10 Fold Cross Validation: Mean of Precision score for 'Normal': {mean_normal_precision:.4f}")
    print(f"10 Fold Cross Validation: Standard Deviation of Precision score for 'Normal': {std_normal_precision:.4f}")



#################################################################################################
# # Functions for Model Evaluation


def get_scores(y_test, y_pred, test_type, classifier):
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    fscore = round(f1_score(y_test, y_pred, average='weighted'), 2)
    precision_value = round(precision_score(y_test, y_pred, average='weighted', zero_division=1), 4) # zero division to stop warning output
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    # To determing Precision of Normal using custom scorer
    normal_precision_scorer = make_scorer(precision_score_for_normal)
    normal_precision = precision_score_for_normal(y_test, y_pred)
    print(f'\n{classifier} scoring on {test_type} set:')
    print('Accuracy: {}, F1 score: {}, Precision score for \'Normal\': {}'.format(accuracy, fscore, normal_precision))
    print('Weighted Averaged Precision: {}, Weighted Average Recall: {}'.format(precision_value, recall))

    return accuracy, fscore, normal_precision
    

def plot_colours(test_type):
    if test_type == "Training":
        return "Blues"
    else:
        return "Oranges"

def plot_Confusion(classifier_with_setted_params, X_test, y_test, accuracy, F1, normal_precision, test_type, classifier_name):
    colour = plot_colours(test_type)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(classifier_with_setted_params, X_test, y_test, ax=ax, cmap=colour)
    num_classes = len(disp.display_labels)
    ax.set_xticks(np.arange(num_classes))  
    ax.set_yticks(np.arange(num_classes)) 
    ax.set_xticklabels(disp.display_labels, rotation=90)  
    ax.set_yticklabels(disp.display_labels)
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(False)
    ax.set_title(f'{classifier_name} Confusion Matrix for {test_type} Dataset\nAccuracy: {accuracy:.2f}, F1: {F1:.2f} \n Precision for Normal: {normal_precision:.4f}')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()  
    plt.draw()      #for jupyter
    # plt.show()  #for .py
    
    return fig


def permutation_feature_importance(classifier, X, y, test): # Calculate permutation importance
    result = permutation_importance(classifier, X, y, n_repeats=3, random_state=42, n_jobs=-1)
    feat_imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title=f'Permutation Feature Importances for {test}')
    plt.ylabel('Importance Score')
    plt.show()


### Used only for Bernoulli due to large number of features - can choose to see only positive or negative scoring features
def POS_permutation_feature_importance(classifier, X, y, test):
    # Calculate permutation importance
    result = permutation_importance(classifier, X, y, n_repeats=3, random_state=42, n_jobs=-1)
    # Create a series with feature importance scores and filter positive values
    feat_imp = pd.Series(result.importances_mean, index=X.columns)
    positive_feat_imp = feat_imp[feat_imp > 0.001].sort_values(ascending=False)
    # Plot only positive feature importances
    if not positive_feat_imp.empty:
        positive_feat_imp.plot(kind='bar', title=f'Positive Permutation Feature Importances for {test}')
        plt.ylabel('Importance Score')
        plt.show()
    else:
        print("No positive feature importances to plot.")
        # Return list of features with positive importance
    if not positive_feat_imp.empty:
        positive_features = positive_feat_imp.index.tolist()
        print(f"Positive features for {test}: {positive_features}")
        return positive_features
    else:
        print("No positive feature importances.")
        return []
