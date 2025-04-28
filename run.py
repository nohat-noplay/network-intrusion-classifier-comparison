# Network Intrusion Classifier Comparison
# Run.py
# Author: Saf Flatters
# Year: 2024

# This is the main run code. 
# This code is divided by:
#           - Data cleaning and preprocessing
#           - Prepare Test Sets
#           - Training and evaluating multiple classification models:
#               - KNN Feature Selection and Transformation
#               - KNN Classication
#               - Naive Bayes Feature Selection and Transformation
#               - Naive Bayes Classication
#               - Decision Trees Feature Selection and Transformation
#               - Decision Trees Classication
#               - Random Forest Feature Selection and Transformation
#               - Random Forest Classication
#               - Gradient Boost Feature Selection and Transformation
#               - Gradient Boost Classication
#           - Compare and Export Best Predictors
#           - Model comparison based on accuracy, F1-score, and precision
#           - Export processed datasets, predictions, and confusion matrices



# !pip install arff
# !pip install -U pandas
# !pip install seaborn
# !pip install imbalanced-learn
# !pip install openpyxl
# !pip install python-Levenshtein

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


from Runpy_Functions import *


print('Hello!')
print('This code will clean training data, prepare test data, export data to .arff, classify with KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, present the top 2 predictions for both test sets and export predictions to csv.')
print('Please be patient....')
#################################################################################################
#################################################################################################
# # Clean Training Set

# Load data into Dataframe
df = pd.read_excel('Given_Datasets/Assignment-2024-training-data-set.xlsx')
pd.set_option('display.max_columns', None)

# Move Target attributes to the front 
current_columns = df.columns.tolist()
columns_to_move = ['label', 'attack_cat']
remaining_columns = [col for col in current_columns if col not in columns_to_move]
new_column_order = columns_to_move + remaining_columns
df = df[new_column_order]

# Preliminary cleaning
df = df[df['state'] != 'no']    # inconsistent entry
df = df.drop(columns=["ct_ftp_cmd"])    # due to duplication with is_ftp_login

# Binarise discrete Features - make_bins = 1
df['swin_bin'] = make_bins(df, 'swin', [255])
df['dwin_bin'] = make_bins(df, 'dwin', [255])
df['trans_depth_bin'] = make_bins(df, "trans_depth", [1])
df['is_ftp_login_bin'] = make_bins(df, "is_ftp_login", [1])
df['sttl_bin'] = make_bins(df, "sttl", [0, 254, 64])
df['dttl_bin'] = make_bins(df, "dttl", [0, 252, 60])

# change data type
df['is_sm_ips_ports'] = df['is_sm_ips_ports'].astype(bool)

# One Hot Encode Categorical Features (state, service, proto)
state_ohe = one_hot_encode(df, 'state')
service_ohe = one_hot_encode(df, 'service')
proto_ohe = one_hot_encode(df, 'proto')
            # filter for only relevant ohe features
proto_ohe = proto_ohe[['proto_udp', 'proto_unas', 'proto_tcp', 'proto_arp', 'proto_ospf', 'proto_rsvp', 'proto_cbt', 'proto_st2', 'proto_xnet', 'proto_hmp', 'proto_mux', 'proto_emcon', 
'proto_sat-mon', 'proto_iatp', 'proto_iso-ip', 'proto_tlsp', 'proto_sep', 'proto_tcf', 'proto_rvd', 'proto_pup', 'proto_sccopmce', 'proto_swipe', 
'proto_iso-tp4', 'proto_ipv6', 'proto_ippc', 'proto_pim', 'proto_netblt', 'proto_sun-nd', 'proto_aes-sp3-d', 'proto_micp', 'proto_mtp', 'proto_encap', 
'proto_larp', 'proto_mobile', 'proto_irtp', 'proto_ax.25', 'proto_egp', 'proto_igmp', 'proto_ipip', 'proto_nvp', 'proto_trunk-1', 'proto_argus', 'proto_mfe-nsp', 
'proto_gre', 'proto_etherip', 'proto_igp', 'proto_visa', 'proto_sps', 'proto_srp', 'proto_leaf-2', 'proto_bbn-rcc', 'proto_narp', 
'proto_ipcv', 'proto_vmtp', 'proto_mhrp']]  
service_ohe = service_ohe[['service_-', 'service_dns', 'service_http', 'service_smtp', 'service_ftp-data', 'service_pop3', 'service_ssh', 'service_ssl', 'service_snmp',  'service_dhcp',  'service_radius']]
state_ohe = state_ohe[['state_CON',]]
encoded_columns = pd.concat([state_ohe, service_ohe, proto_ohe], axis=1)
df = pd.concat([df, encoded_columns], axis=1)
df = df.drop(columns=['state', 'service', 'proto'])     # required to do here for SMOTE

# Engineer Features (edit_distance, rate_over_10000, ct_mean)
df['edit_distance'] = calc_edit_distance(df, "stcpb", "dtcpb")
df['rate_over_10000'] = make_boolean(df, "rate", 10000)  #?
df['rate_digit_count'] = count_digits(df, "rate")       #?
df['ct_mean'] = find_mean(df, 'ct', ['ct_state_ttl', 'ct_flw_http_mthd'])       # add the mean column to dataframe


######### EXPORT PROCESSED TRAINING SET ################

# export to arff
export_arff(df, 'Processed_Training_Set', 'attack_cat')
# export to csv
# df.to_csv("processed_training_data.csv")





#################################################################################################
#################################################################################################
# # Prepare Test Sets

# In[697]:

####### PREP TEST SET 1 ############
# Load data into Dataframe
t1 = pd.read_excel('Given_Datasets/Test-Data-Set-1-2024.xlsx')

# Binarise discrete Features - make_bins = 1
t1['swin_bin'] = make_bins(t1, 'swin', [255])
t1['dwin_bin'] = make_bins(t1, 'dwin', [255])
t1['trans_depth_bin'] = make_bins(t1, "trans_depth", [1])
t1['is_ftp_login_bin'] = make_bins(t1, "is_ftp_login", [1])
t1['sttl_bin'] = make_bins(t1, "sttl", [0, 254, 64])
t1['dttl_bin'] = make_bins(t1, "dttl", [0, 252, 60])
t1 = t1.drop(columns=["ct_ftp_cmd"])    # due to duplication with is_ftp_login

# change data type
t1['is_sm_ips_ports'] = t1['is_sm_ips_ports'].astype(bool)

# One Hot Encode Categorical Features (state, service, proto)
state_ohe = one_hot_encode(t1, 'state')
service_ohe = one_hot_encode(t1, 'service')
proto_ohe = one_hot_encode(t1, 'proto')
            # filter for only relevant ohe features
proto_ohe = proto_ohe[['proto_udp', 'proto_unas', 'proto_tcp', 'proto_arp', 'proto_ospf', 'proto_rsvp', 'proto_cbt', 'proto_st2', 'proto_xnet', 'proto_hmp', 'proto_mux', 'proto_emcon', 
'proto_sat-mon', 'proto_iatp', 'proto_iso-ip', 'proto_tlsp', 'proto_sep', 'proto_tcf', 'proto_rvd', 'proto_pup', 'proto_sccopmce', 'proto_swipe', 
'proto_iso-tp4', 'proto_ipv6', 'proto_ippc', 'proto_pim', 'proto_netblt', 'proto_sun-nd', 'proto_aes-sp3-d', 'proto_micp', 'proto_mtp', 'proto_encap', 
'proto_larp', 'proto_mobile', 'proto_irtp', 'proto_ax.25', 'proto_egp', 'proto_igmp', 'proto_ipip', 'proto_nvp', 'proto_trunk-1', 'proto_argus', 'proto_mfe-nsp', 'proto_gre', 'proto_etherip', 'proto_igp', 'proto_visa', 'proto_sps', 'proto_srp', 'proto_leaf-2', 'proto_bbn-rcc', 'proto_narp', 
'proto_ipcv', 'proto_vmtp', 'proto_mhrp']]  
service_ohe = service_ohe[['service_-', 'service_dns', 'service_http', 'service_smtp', 'service_ftp-data', 'service_ssl', 'service_pop3', 'service_ssh', 'service_snmp',  'service_dhcp',  'service_radius']]
state_ohe = state_ohe[['state_CON',]]
encoded_columns = pd.concat([state_ohe, service_ohe, proto_ohe], axis=1)
t1 = pd.concat([t1, encoded_columns], axis=1)
t1 = t1.drop(columns=['state', 'service', 'proto'])     # required to do here for SMOTE


# Engineer Features (edit_distance, rate_over_10000, ct_mean)
t1['edit_distance'] = calc_edit_distance(t1, "stcpb", "dtcpb")
t1['rate_over_10000'] = make_boolean(t1, "rate", 10000)  #?
t1['rate_digit_count'] = count_digits(t1, "rate")       #?
t1['ct_mean'] = find_mean(t1, 'ct', ['ct_state_ttl', 'ct_flw_http_mthd'])       # add the mean column to dataframe

######### EXPORT PROCESSED TEST SET 1 ################

# export to arff
export_arff(t1, 'Processed_Test_Set_1', 'attack_cat') 
# export to csv
# t1.to_csv("processed_test_set1.csv")

####### PREP TEST SET 2 ############
# Load data into Dataframe
t2 = pd.read_excel('Given_Datasets/Test-Data-Set-2-2024.xlsx')

# Binarise discrete Features - make_bins = 1
t2['swin_bin'] = make_bins(t2, 'swin', [255])
t2['dwin_bin'] = make_bins(t2, 'dwin', [255])
t2['trans_depth_bin'] = make_bins(t2, "trans_depth", [1])
t2['is_ftp_login_bin'] = make_bins(t2, "is_ftp_login", [1])
t2['sttl_bin'] = make_bins(t2, "sttl", [0, 254, 64])
t2['dttl_bin'] = make_bins(t2, "dttl", [0, 252, 60])
t2 = t2.drop(columns=["ct_ftp_cmd"])    # due to duplication with is_ftp_login

# change data type
t2['is_sm_ips_ports'] = t2['is_sm_ips_ports'].astype(bool)

# One Hot Encode Categorical Features (state, service, proto)
state_ohe = one_hot_encode(t2, 'state')
service_ohe = one_hot_encode(t2, 'service')
proto_ohe = one_hot_encode(t2, 'proto')
            # filter for only relevant ohe features
proto_ohe = proto_ohe[['proto_udp', 'proto_unas', 'proto_tcp', 'proto_arp', 'proto_ospf', 'proto_rsvp', 'proto_cbt', 'proto_st2', 'proto_xnet', 'proto_hmp', 'proto_mux', 'proto_emcon', 
'proto_sat-mon', 'proto_iatp', 'proto_iso-ip', 'proto_tlsp', 'proto_sep', 'proto_tcf', 'proto_rvd', 'proto_pup', 'proto_sccopmce', 'proto_swipe', 
'proto_iso-tp4', 'proto_ipv6', 'proto_ippc', 'proto_pim', 'proto_netblt', 'proto_sun-nd', 'proto_aes-sp3-d', 'proto_micp', 'proto_mtp', 'proto_encap', 
'proto_larp', 'proto_mobile', 'proto_irtp', 'proto_ax.25', 'proto_egp', 'proto_igmp', 'proto_ipip', 'proto_nvp', 'proto_trunk-1', 'proto_argus', 'proto_mfe-nsp', 
'proto_gre', 'proto_etherip', 'proto_igp', 'proto_visa', 'proto_sps', 'proto_srp', 'proto_leaf-2', 'proto_bbn-rcc', 'proto_narp', 
'proto_ipcv', 'proto_vmtp', 'proto_mhrp']]  
service_ohe = service_ohe[['service_-', 'service_dns', 'service_http', 'service_smtp', 'service_ssl', 'service_ftp-data', 'service_pop3', 'service_ssh', 'service_snmp',  'service_dhcp',  'service_radius']]
state_ohe = state_ohe[['state_CON',]]
encoded_columns = pd.concat([state_ohe, service_ohe, proto_ohe], axis=1)
t2 = pd.concat([t2, encoded_columns], axis=1)
t2 = t2.drop(columns=['state', 'service', 'proto'])     # required to do here for SMOTE


# Engineer Features (edit_distance, rate_over_10000, ct_mean)
t2['edit_distance'] = calc_edit_distance(t2, "stcpb", "dtcpb")
t2['rate_over_10000'] = make_boolean(t2, "rate", 10000)  #?
t2['rate_digit_count'] = count_digits(t2, "rate")       #?
t2['ct_mean'] = find_mean(t2, 'ct', ['ct_state_ttl', 'ct_flw_http_mthd'])       # add the mean column to dataframe


######### EXPORT PROCESSED TEST SET 2 ################

# export to arff
export_arff(t2, 'Processed_Test_Set_2', 'attack_cat') 
# export to csv
# t2.to_csv("processed_test_set2.csv")

# t1.info()


#################################################################################################
#################################################################################################
# Classification


#################################################################################################
# # KNN: Feature Selection & Transformation


#13 features version
features_df = df[['attack_cat',
 'sttl',
 'smean',
 'dmean',
 'service_-',
 'service_smtp',
 'service_dns',
 'proto_udp',
 'tcprtt',
 'ct_state_ttl',
 'ct_mean', 
 'sjit', 
 'dur', 
 'sload'
    ]] 


# Data Balance
# df_balanced = balance_w_SMOTE(features_df, target_col='attack_cat', target_size=6666)
df_balanced = features_df

# Split Sets
training_set = df_balanced
X = training_set.drop('attack_cat', axis=1)
y = training_set['attack_cat']  #for training to check type of attack

features = X.columns

X_test_1 = t1[features]
X_test_2 = t2[features]

y_test_1 = t1['attack_cat']
y_test_2 = t2['attack_cat']


# Standardise Training Set
scaler = standardise_data(X)

# Standardise Test Sets with same scaler
standardise_test_data(X_test_1, scaler)
standardise_test_data(X_test_2, scaler)




#################################################################################################
# # KNN Classification

#### k-Nearest Neighbour
hypertune = "skip" #show if want to tune hyper parameters more

# trainer

# Training Set split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) #80/20 split

# Initialize the KNN classifier and find best parameters
# knn = KNeighborsClassifier()

##### This is to reduce time now hyper parameters have been tuned!
if hypertune == "show":
    best_params = hypertune_w_CV("K-Nearest Neighbour", 'accuracy', X_train, y_train)
elif hypertune == "skip":     
    print("\n\nPre-set: K-Nearest Neighbour Best Parameters: 'metric': 'manhattan', 'n_jobs': -1, 'n_neighbors': 11, 'p': 1, 'weights': 'distance'")
    print("Best GridSearchCV Score: 0.9064")
    best_params = {'metric': 'manhattan', 'n_jobs': -1, 'n_neighbors': 13, 'p': 1, 'weights': 'distance'}
    

# Train the final KNN model with the best hyperparameters on the entire training set
set_knn = KNeighborsClassifier(**best_params)
set_knn.fit(X_train, y_train)

# Cross Validate
cross_validate(set_knn, X_train, y_train, scoring='accuracy')  # this scoring also includes normal_precision custom scorer

# Make predictions on the training (X_test) set
y_train_pred = set_knn.predict(X_test)

# Evaluate the model performance
# print("\nTRAINING DATA:")
accuracy_train, F1_train, normal_precision_train = get_scores(y_test, y_train_pred, "Training", "KNN")  
KNN_Conf_train = plot_Confusion(set_knn, X_test, y_test, accuracy_train, F1_train, normal_precision_train, 'Training', "KNN")

# tester

# Predictions and accuracy for the first test set
y_test_1_pred = set_knn.predict(X_test_1)

# Evaluate the model performance
# print("\nTEST 1 DATA:")
accuracy_t1, F1_t1, normal_precision_t1 = get_scores(y_test_1, y_test_1_pred, "Test 1", "KNN")  
KNN_Conf_t1 = plot_Confusion(set_knn, X_test_1, y_test_1, accuracy_t1, F1_t1, normal_precision_t1, 'Test 1', "KNN")

# Predictions and accuracy for the second test set
y_test_2_pred = set_knn.predict(X_test_2)

# Evaluate the model performance
# print("\nTEST 2 DATA:")
accuracy_t2, F1_t2, normal_precision_t2 = get_scores(y_test_2, y_test_2_pred, "Test 2", "KNN")  
KNN_Conf_t2 = plot_Confusion(set_knn, X_test_2, y_test_2, accuracy_t2, F1_t2, normal_precision_t2, 'Test 2', "KNN")

KNN_t1_accuracyscore = accuracy_t1
KNN_t2_accuracyscore = accuracy_t2
KNN_t1_normalprecision = normal_precision_t1
KNN_t2_normalprecision = normal_precision_t2
KNN_t1_predictions = y_test_1_pred
KNN_t2_predictions = y_test_2_pred


# # KNN Feature Importance Evaluation
# permutation_feature_importance(set_knn, X_test_1, y_test_1, "Test 1")
# permutation_feature_importance(set_knn, X_test_2, y_test_2, "Test 2")






#################################################################################################
# # NB: Feature Selection & Transformation

# Feature Selection

# #For bernoulli naive bayes - must comment out standardisation
#ALL BOOLEAN that give positive feature importance
features_df = df[['attack_cat', 'dttl_bin', 'sttl_bin', 'proto_udp', 'state_CON', 'service_dns', 'proto_arp', 'is_sm_ips_ports', 'service_-', 
'proto_ipv6', 'service_pop3', 'service_snmp', 'proto_rsvp', 'proto_cbt', 'proto_st2', 'proto_xnet', 'proto_hmp', 'proto_mux', 'proto_emcon', 
'proto_sat-mon', 'proto_iatp', 'proto_iso-ip', 'proto_tlsp', 'proto_sep', 'proto_tcf', 'proto_rvd', 'proto_pup', 'proto_sccopmce', 'proto_swipe', 
'proto_iso-tp4', 'service_dhcp', 'proto_ippc', 'proto_pim', 'proto_netblt', 'proto_sun-nd', 'proto_aes-sp3-d', 'proto_micp', 'proto_mtp', 'proto_encap', 
'proto_larp', 'proto_mobile', 'proto_irtp', 'proto_ax.25', 'proto_egp', 'proto_igmp', 'proto_ipip', 'proto_nvp', 'proto_trunk-1', 'proto_argus', 'proto_mfe-nsp', 
'service_ssl', 'proto_gre', 'proto_etherip', 'proto_igp', 'proto_visa', 'proto_sps', 'proto_srp', 'proto_leaf-2', 'service_radius', 'proto_bbn-rcc', 'proto_narp', 
'proto_ipcv', 'proto_vmtp', 'proto_mhrp'
    ]]

#for gaussian naive bayes
#14 features version
# features_df = df[['attack_cat',
#  'dur', 
#  'spkts',
#  'rate',
#  'sttl',
#  'sjit',
#  'smean',
#  'dmean',
#  'sttl_bin',
#  'dttl_bin',
#  'service_-',
#  'service_dns',
#  'service_smtp',
#  'proto_udp',
#  'proto_unas',
#     ]] 

# # Data Balance
# df_balanced = balance_w_SMOTE(features_df, target_col='attack_cat', target_size=6666)
df_balanced = features_df

# Split Sets
training_set = df_balanced
X = training_set.drop('attack_cat', axis=1)
y = training_set['attack_cat']  #for training to check type of attack

features = X.columns

X_test_1 = t1[features]
X_test_2 = t2[features]

y_test_1 = t1['attack_cat']
y_test_2 = t2['attack_cat']

#To be commented out if Bernoulli Naive Bayes
# # Standardise Training Set
# scaler = standardise_data(X)

# # Standardise Test Sets with same scaler
# standardise_test_data(X_test_1, scaler)
# standardise_test_data(X_test_2, scaler)

#################################################################################################
# # Naive Bayes Classification

#### Gaussian or Bernoulli Naive Bayes

# Training Set split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) #80/20 split

# trainer

# Initialise the NB classifier and find best hyperparameters with GridSearch
# best_params = hypertune_w_CV("Gaussian Naive Bayes", 'accuracy', X_train, y_train)
best_params = hypertune_w_CV("Bernoulli Naive Bayes", 'accuracy', X_train, y_train)   #gaussian or bernoulli

# Train the final KNN model with the best hyperparameters on the entire training set
# set_nb = GaussianNB(**best_params) #gaussian or bernoulli
set_nb = BernoulliNB(**best_params) #gaussian or bernoulli
set_nb.fit(X_train, y_train)

# Cross Validate
cross_validate(set_nb, X_train, y_train, scoring='accuracy')  # this scoring also includes normal_precision custom scorer

# Make predictions on the training (X_test) set
y_train_pred = set_nb.predict(X_test)

# Evaluate the model performance
# print("\nTRAINING DATA:")
accuracy_train, F1_train, normal_precision_train = get_scores(y_test, y_train_pred, "Training", "Naive Bayes")  
NB_Conf_train = plot_Confusion(set_nb, X_test, y_test, accuracy_train, F1_train, normal_precision_train, 'Training', "Naive Bayes")
# plot_permutation_importance(set_nb, X_test, y_test, 'training')

# tester

# Predictions and accuracy for the first test set
y_test_1_pred = set_nb.predict(X_test_1)

# Evaluate the model performance
# print("\nTEST 1 DATA:")
accuracy_t1, F1_t1, normal_precision_t1 = get_scores(y_test_1, y_test_1_pred, "Test 1", "Naive Bayes")  
NB_Conf_t1 = plot_Confusion(set_nb, X_test_1, y_test_1, accuracy_t1, F1_t1, normal_precision_t1, 'Test 1', "Naive Bayes")

# Predictions and accuracy for the first test set
y_test_2_pred = set_nb.predict(X_test_2)

# Evaluate the model performance
# print("\nTEST 2 DATA:")
accuracy_t2, F1_t2, normal_precision_t2 = get_scores(y_test_2, y_test_2_pred, "Test 2", "Naive Bayes")  
NB_Conf_t2 = plot_Confusion(set_nb, X_test_2, y_test_2, accuracy_t2, F1_t2, normal_precision_t2, 'Test 2', "Naive Bayes")

NB_t1_accuracyscore = accuracy_t1
NB_t2_accuracyscore = accuracy_t2
NB_t1_normalprecision = normal_precision_t1
NB_t2_normalprecision = normal_precision_t2
NB_t1_predictions = y_test_1_pred
NB_t2_predictions = y_test_2_pred


# # NB Feature Importance Evaluation
# permutation_feature_importance(set_nb, X_test_1, y_test_1, "Test 1")
# permutation_feature_importance(set_nb, X_test_2, y_test_2, "Test 2")





#################################################################################################
# # DT: Feature Selection & Transformation

# Feature Selection

#10 features version 
features_df = df[['attack_cat',
 'dur', 
 'spkts',
 'rate',
 'sttl',
 'sjit',
 'smean',
 'dmean',
 'service_-',
 'service_dns',
 'proto_udp',
 'proto_swipe',
 'proto_sun-nd',
 'proto_mobile'
    ]] 

# Data Balance
# df_balanced = balance_w_SMOTE(features_df, target_col='attack_cat', target_size=6666)
df_balanced = features_df

# Split Sets
training_set = df_balanced
X = training_set.drop('attack_cat', axis=1)
y = training_set['attack_cat']  #for training to check type of attack

features = X.columns

X_test_1 = t1[features]
X_test_2 = t2[features]

y_test_1 = t1['attack_cat']
y_test_2 = t2['attack_cat']


# Standardise Training Set
scaler = standardise_data(X)

# Standardise Test Sets with same scaler
standardise_test_data(X_test_1, scaler)
standardise_test_data(X_test_2, scaler)



#################################################################################################
# # Decision Trees Classification

#### Decision Trees
hypertune = "skip" #show if want to tune hyper parameters more

# Training Set split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) #80/20 split

# trainer


##### This is to reduce time now hyper parameters have been tuned!
if hypertune == "show":
    # Initialize the DT classifier and find best parameters
    best_params = hypertune_w_CV("Decision Tree", 'accuracy', X_train, y_train)
elif hypertune == "skip":     
    print("\n\nPre-set: Decision Tree Best Parameters: {'criterion': 'gini', 'max_depth': 30, 'splitter': 'best'}")
    print("Best GridSearchCV Score: 0.9117")
    best_params = {'criterion': 'gini', 'max_depth': 20, 'splitter': 'best'}


# Train the final DT model with the best hyperparameters on the entire training set
set_dt = DecisionTreeClassifier(**best_params)
set_dt.fit(X_train, y_train)

# Cross Validate
cross_validate(set_dt, X_train, y_train, scoring='accuracy')  # this scoring also includes normal_precision custom scorer

# Make predictions on the training (X_test) set
y_train_pred = set_dt.predict(X_test)

# Evaluate the model performance
# print("\nTRAINING DATA:")
accuracy_train, F1_train, normal_precision_train = get_scores(y_test, y_train_pred, "Training", "Decision Tree")  
DT_Conf_train = plot_Confusion(set_dt, X_test, y_test, accuracy_train, F1_train, normal_precision_train, 'Training', "Decision Tree")

# tester

# Predictions and accuracy for the first test set
y_test_1_pred = set_dt.predict(X_test_1)

# Evaluate the model performance
# print("\nTEST 1 DATA:")
accuracy_t1, F1_t1, normal_precision_t1 = get_scores(y_test_1, y_test_1_pred, "Test 1", "Decision Tree")  
DT_Conf_t1 = plot_Confusion(set_dt, X_test_1, y_test_1, accuracy_t1, F1_t1, normal_precision_t1, 'Test 1', "Decision Tree")

# Predictions and accuracy for the first test set
y_test_2_pred = set_dt.predict(X_test_2)

# Evaluate the model performance
# print("\nTEST 2 DATA:")
accuracy_t2, F1_t2, normal_precision_t2 = get_scores(y_test_2, y_test_2_pred, "Test 2", "Decision Tree")  
DT_Conf_t2 = plot_Confusion(set_dt, X_test_2, y_test_2, accuracy_t2, F1_t2, normal_precision_t2, 'Test 2', "Decision Tree")

DT_t1_accuracyscore = accuracy_t1
DT_t2_accuracyscore = accuracy_t2
DT_t1_normalprecision = normal_precision_t1
DT_t2_normalprecision = normal_precision_t2
DT_t1_predictions = y_test_1_pred
DT_t2_predictions = y_test_2_pred


# # DT Feature Importance Evaluation
# #SPlits
# importances = set_dt.feature_importances_
# # Create a DataFrame for visualization
# feature_importance_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
# # Plotting
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance from dt')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()




#################################################################################################
# # RF: Feature Selection & Transformation 

#14 features version
features_df = df[['attack_cat',
 'dur', 
 'spkts',
 'rate',
 'sttl',
 'sjit',
 'smean',
 'dmean',
 'sttl_bin',
 'dttl_bin',
 'service_-',
 'service_dns',
 'service_smtp',
 'proto_udp',
 'proto_unas',
    ]] 

# Data Balance
# df_balanced = balance_w_SMOTE(features_df, target_col='attack_cat', target_size=6666)
df_balanced = features_df

# Split Sets
training_set = df_balanced
X = training_set.drop('attack_cat', axis=1)
y = training_set['attack_cat']  #for training to check type of attack

features = X.columns

X_test_1 = t1[features]
X_test_2 = t2[features]

y_test_1 = t1['attack_cat']
y_test_2 = t2['attack_cat']

# Standardise Training Set
scaler = standardise_data(X)

# Standardise Test Sets with same scaler
standardise_test_data(X_test_1, scaler)
standardise_test_data(X_test_2, scaler)

#################################################################################################
# # Random Forest Classification

#### Random Forest
hypertune = "skip" #show if want to tune hyper parameters more

# Training Set split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) #80/20 split

# trainer

##### This is to reduce time now hyper parameters have been tuned!
if hypertune == "show":
    # Initialize the RF classifier and find best parameters
    best_params = hypertune_w_CV("Random Forest", 'accuracy', X_train, y_train)
elif hypertune == "skip":     
    print("\n\nPre-set: Random Forest Best Parameters: {'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 10, 'n_estimators': 200, 'n_jobs': -1}")
    print("Best GridSearchCV Score: 0.9114")
    best_params = {'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 10, 'n_estimators': 200, 'n_jobs': -1}

# Train the final RF model with the best hyperparameters on the entire training set
set_rf = RandomForestClassifier(**best_params)
set_rf.fit(X_train, y_train)

# Cross Validate
cross_validate(set_rf, X_train, y_train, scoring='accuracy')  # this scoring also includes normal_precision custom scorer

# Make predictions on the training (X_test) set
y_train_pred = set_rf.predict(X_test)

# Evaluate the model performance
# print("\nTRAINING DATA:")
accuracy_train, F1_train, normal_precision_train = get_scores(y_test, y_train_pred, "Training", "Random Forest")  
RF_Conf_train = plot_Confusion(set_rf, X_test, y_test, accuracy_train, F1_train, normal_precision_train, 'Training', "Random Forest")

# tester

# Predictions and accuracy for the first test set
y_test_1_pred = set_rf.predict(X_test_1)

# Evaluate the model performance
# print("\nTEST 1 DATA:")
accuracy_t1, F1_t1, normal_precision_t1 = get_scores(y_test_1, y_test_1_pred, "Test 1", "Random Forest")  
RF_Conf_t1 = plot_Confusion(set_rf, X_test_1, y_test_1, accuracy_t1, F1_t1, normal_precision_t1, 'Test 1', "Random Forest")

# Predictions and accuracy for the first test set
y_test_2_pred = set_rf.predict(X_test_2)

# Evaluate the model performance
# print("\nTEST 2 DATA:")
accuracy_t2, F1_t2, normal_precision_t2 = get_scores(y_test_2, y_test_2_pred, "Test 2", "Random Forest")  
RF_Conf_t2 = plot_Confusion(set_rf, X_test_2, y_test_2, accuracy_t2, F1_t2, normal_precision_t2, 'Test 2', "Random Forest")

RF_t1_accuracyscore = accuracy_t1
RF_t2_accuracyscore = accuracy_t2
RF_t1_normalprecision = normal_precision_t1
RF_t2_normalprecision = normal_precision_t2
RF_t1_predictions = y_test_1_pred
RF_t2_predictions = y_test_2_pred


# # RF: Feature Importance Evaluation
# #This is to see where the tree splits! Not feature importance after the test
# importances = set_rf.feature_importances_
# # Create a DataFrame for visualization
# feature_importance_rf = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
# # Plotting
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_rf)
# plt.title('Feature Importance from rf')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()





#################################################################################################
# # GB: Feature Selection & Transformation

#8 features
features_df = df[['attack_cat', 
 'spkts',
 'sttl',
 'smean',
 'dmean',
 'dttl_bin',
 'service_-',
 'service_dns',
 'proto_udp',
    ]]        #, 'label'

# Data Balance
# df_balanced = balance_w_SMOTE(features_df, target_col='attack_cat', target_size=6666)
df_balanced = features_df

# Split Sets
training_set = df_balanced
X = training_set.drop('attack_cat', axis=1)
y = training_set['attack_cat']  #for training to check type of attack

features = X.columns

X_test_1 = t1[features]
X_test_2 = t2[features]

y_test_1 = t1['attack_cat']
y_test_2 = t2['attack_cat']

# Standardise Training Set
scaler = standardise_data(X)

# Standardise Test Sets with same scaler
standardise_test_data(X_test_1, scaler)
standardise_test_data(X_test_2, scaler)



#################################################################################################
# # Gradient Boost Classification

#### Gradient Boost
hypertune = "skip" #show if want to tune hyper parameters more

# Training Set split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) #80/20 split

# trainer

##### This is to reduce time now hyper parameters have been tuned!
if hypertune == "show":
    # Initialize the GB classifier and find best parameters
    best_params = hypertune_w_CV("Gradient Boost", 'accuracy', X_train, y_train)
elif hypertune == "skip":     
    print("\n\nPre-set: Gradient Boost Best Parameters: {'class_weight': None, 'learning_rate': 0.01, 'max_depth': 8, 'max_iter': 200}")
    print("Best GridSearchCV Score: 0.9106")
    best_params = {'class_weight': None, 'learning_rate': 0.01, 'max_depth': 8, 'max_iter': 200}


# Train the final GB model with the best hyperparameters on the entire training set
set_gb = HistGradientBoostingClassifier(**best_params)  #Histogram
set_gb.fit(X_train, y_train)

# Cross Validate
cross_validate(set_gb, X_train, y_train, scoring='accuracy')  # this scoring also includes normal_precision custom scorer

# Make predictions on the training (X_test) set
y_train_pred = set_gb.predict(X_test)

# Evaluate the model performance
# print("\nTRAINING DATA:")
accuracy_train, F1_train, normal_precision_train = get_scores(y_test, y_train_pred, "Training", "Gradient Boost")  
GB_Conf_train = plot_Confusion(set_gb, X_test, y_test, accuracy_train, F1_train, normal_precision_train, 'Training', "Gradient Boost")

# tester

# Predictions and accuracy for the first test set
y_test_1_pred = set_gb.predict(X_test_1)

# Evaluate the model performance
# print("\nTEST 1 DATA:")
accuracy_t1, F1_t1, normal_precision_t1 = get_scores(y_test_1, y_test_1_pred, "Test 1", "Gradient Boost")  
GB_Conf_t1 = plot_Confusion(set_gb, X_test_1, y_test_1, accuracy_t1, F1_t1, normal_precision_t1, 'Test 1', "Gradient Boost")

# Predictions and accuracy for the first test set
y_test_2_pred = set_gb.predict(X_test_2)

# Evaluate the model performance
# print("\nTEST 2 DATA:")
accuracy_t2, F1_t2, normal_precision_t2 = get_scores(y_test_2, y_test_2_pred, "Test 2", "Gradient Boost")  
GB_Conf_t2 = plot_Confusion(set_gb, X_test_2, y_test_2, accuracy_t2, F1_t2, normal_precision_t2, 'Test 2', "Gradient Boost")

GB_t1_accuracyscore = accuracy_t1
GB_t2_accuracyscore = accuracy_t2
GB_t1_normalprecision = normal_precision_t1
GB_t2_normalprecision = normal_precision_t2
GB_t1_predictions = y_test_1_pred
GB_t2_predictions = y_test_2_pred


# # GB Feature Importance Evaluation
# permutation_feature_importance(set_gb, X_test_1, y_test_1, "Test 1")
# permutation_feature_importance(set_gb, X_test_2, y_test_2, "Test 2")





#################################################################################################
#################################################################################################
# # Compare & Export Best Predictors
# Top two 'accuracy' scores for each Test Data Set

print("\n\n TOP PREDICTIONS FOR EACH TEST SET...\n")

# Accuracy scores
test1_accuracy_scores = {
    'KNN_t1': KNN_t1_accuracyscore,
    'NB_t1': NB_t1_accuracyscore,
    'DT_t1': DT_t1_accuracyscore, 
    'RF_t1': RF_t1_accuracyscore, 
    'GB_t1': GB_t1_accuracyscore
}

test2_accuracy_scores = {
    'KNN_t2': KNN_t2_accuracyscore,
    'NB_t2': NB_t2_accuracyscore,
    'DT_t2': DT_t2_accuracyscore, 
    'RF_t2': RF_t2_accuracyscore, 
    'GB_t2': GB_t2_accuracyscore
}

precision_normal = {
    'KNN_t1': KNN_t1_normalprecision,
    'KNN_t2': KNN_t2_normalprecision,
    'NB_t1': NB_t1_normalprecision,
    'NB_t2': NB_t2_normalprecision,
    'DT_t1': DT_t1_normalprecision,
    'DT_t2': DT_t2_normalprecision, 
    'RF_t1': RF_t1_normalprecision, 
    'RF_t2': RF_t2_normalprecision, 
    'GB_t1': GB_t1_normalprecision, 
    'GB_t2': GB_t2_normalprecision
}

# Confusion matrices associated with each accuracy score
confusion_matrices = {
    'KNN_t1': KNN_Conf_t1,
    'KNN_t2': KNN_Conf_t2,
    'NB_t1': NB_Conf_t1,
    'NB_t2': NB_Conf_t2,
    'DT_t1': DT_Conf_t1,
    'DT_t2': DT_Conf_t2,
    'RF_t1': RF_Conf_t1,
    'RF_t2': RF_Conf_t2, 
    'GB_t1': GB_Conf_t1,
    'GB_t2': GB_Conf_t2
}

# Predictions associated with each accuracy score
predictions = {
    'KNN_t1': KNN_t1_predictions,
    'KNN_t2': KNN_t2_predictions,
    'NB_t1': NB_t1_predictions,
    'NB_t2': NB_t2_predictions,
    'DT_t1': DT_t1_predictions,
    'DT_t2': DT_t2_predictions,
    'RF_t1': RF_t1_predictions,
    'RF_t2': RF_t2_predictions, 
    'GB_t1': GB_t1_predictions,
    'GB_t2': GB_t2_predictions
}


# Sort accuracy scores to find the top two
t1_sorted_accuracies = sorted(test1_accuracy_scores.items(), key=lambda x: x[1], reverse=True)
test1_top_two = t1_sorted_accuracies[:2]

t2_sorted_accuracies = sorted(test2_accuracy_scores.items(), key=lambda x: x[1], reverse=True)
test2_top_two = t2_sorted_accuracies[:2]

# Print the scores for the top two accuracy scores
print("\nTest Data Set 1 - Top Two Accuracy Predictions:")
for model, acc in test1_top_two:
    precision_norm_score = precision_normal.get(model, "N/A")  # Get precision score, default to "N/A" if not found
    print(f"Model: {model}, Accuracy: {acc}, Precision Score for 'Normal': {precision_norm_score}")

print("\nTest Data Set 2 - Top Two Accuracy Predictions:")
for model, acc in test2_top_two:
    precision_norm_score = precision_normal.get(model, "N/A")  # Get precision score, default to "N/A" if not found
    print(f"Model: {model}, Accuracy: {acc}, Precision Score for 'Normal': {precision_norm_score}")



#################################################################################################
#################################################################################################
# Encoded attack_cat to CSV predict.csv files

# Dictionary to encode attack_cat categories
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

# Apply encoding to Test Data Set 1 predictions
encoded_test1_predictions = pd.DataFrame({
    'Test1_Predict1': pd.Series(predictions[test1_top_two[0][0]]).replace(attack_cat_encoding),
    'Test1_Predict2': pd.Series(predictions[test1_top_two[1][0]]).replace(attack_cat_encoding)
})

# Apply encoding to Test Data Set 2 predictions
encoded_test2_predictions = pd.DataFrame({
    'Test2_Predict1': pd.Series(predictions[test2_top_two[0][0]]).replace(attack_cat_encoding),
    'Test2_Predict2': pd.Series(predictions[test2_top_two[1][0]]).replace(attack_cat_encoding)
})

# Export Test Data Set 1 encoded predictions to CSV
encoded_test1_predictions.index = np.arange(1, len(encoded_test1_predictions) + 1)
encoded_test1_predictions.to_csv('predict_1.csv', index=True, index_label='ID')
print("\nEncoded top two prediction sets for Test Data Set 1 have been saved to 'predict_1.csv'.")

# Export Test Data Set 2 encoded predictions to CSV
encoded_test2_predictions.index = np.arange(1, len(encoded_test2_predictions) + 1)
encoded_test2_predictions.to_csv('predict_2.csv', index=True, index_label='ID')
print("\nEncoded top two prediction sets for Test Data Set 2 have been saved to 'predict_2.csv'.\n")


# NOT ENCODED
# # Create a DataFrame for the top two predictions
# test1_top_predictions = {
#     'Predict1': predictions[test1_top_two[0][0]],  # First highest
#     'Predict2': predictions[test1_top_two[1][0]]  # Second highest
# }

# test2_top_predictions = {
#     'Predict1': predictions[test2_top_two[0][0]],  # First highest
#     'Predict2': predictions[test2_top_two[1][0]]  # Second highest
# }

# # Export to CSV
# df_test1 = pd.DataFrame(test1_top_predictions)
# df_test1.index = np.arange(1, len(df_test1) + 1)
# df_test1.to_csv('predict_1.csv', index=True, index_label='ID')
# print("\nTop two prediction sets for Test Data Set 1 have been saved to 'predict_1.csv'.")

# df_test2 = pd.DataFrame(test2_top_predictions)
# df_test2.index = np.arange(1, len(df_test2) + 1)
# df_test2.to_csv('predict_2.csv', index=True, index_label='ID')
# print("\nTop two prediction sets for Test Data Set 2 have been saved to 'predict_2.csv'.\n")



#################################################################################################
#################################################################################################
# EXPORT Confusion Matrices


subfolder = "Confusion_Matrices"
os.makedirs(subfolder, exist_ok=True) # if it doesnt exist

for name, fig in confusion_matrices.items():
    file_path = os.path.join(subfolder, f"{name}_confusion_matrix.png")  
    fig.savefig(file_path)  

print('Exported all Test Confusion Matrices to Confusion_Matrices Folder\n')