# Fit Logistic Regression Classifier

##################################################################################################

# Packages
import pandas as pd
import sklearn
import os
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import time

##################################################################################################

# Start timer
start_time = time.time()

##################################################################################################

# Load Training Data
def combine_directory_parquets(directory_path):
    '''
    Combines all parquet files in a directory into a single dataframe.
    '''
    # If path does not end in a slash, add one
    if directory_path[-1] != '/':
        directory_path += '/'
    # list of files in directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    # read in all parquet files
    combined_df = pd.concat([pd.read_parquet(directory_path + f) for f in file_list])
    # Return combined dataframe
    return combined_df
training_data = combine_directory_parquets('../../../Data/Features/All Features/train')
# Drop Image Path, test_80_20
training_data = training_data.drop(columns=['Image Path', 'test_80_20'])
print('all training data')
print(training_data)

##################################################################################################

# Hyperparameter Settings
# Regularization: none, L1, L2
# Balanced Class Weights: yes, no
# Multi-Class Strategy: one-vs-rest, multinomial

# Create combinations of all hyperparameter settings
regularization = ['l1', 'l2', None]
class_weight = ['balanced', None]
multi_class_strategy = ['ovr', 'multinomial']
# Hyperparameter setting combinations dataframe
hyperparameters_setting_combos = pd.DataFrame(columns=['regularization', 'class_weight', 'multi_class_strategy'])
# Iterate through all hyperparameter settings
for reg in regularization:
    for cw in class_weight:
        for mcs in multi_class_strategy:
            hyperparameters_setting_combos = pd.concat([hyperparameters_setting_combos, pd.DataFrame({'regularization': reg, 'class_weight': cw, 'multi_class_strategy': mcs}, index=[0])])
print('hyperparameters_setting_combos')
print(hyperparameters_setting_combos)

##################################################################################################

# Create Hyperparameter Setting Validation Set
# Sample 10% of the training data for validation.
# Note still using cross validation for l1, l2, etc. choices

hps_validation_df = training_data.sample(frac=0.1, random_state=290)
X_hps_validation = hps_validation_df.drop(columns=['Class'])
y_hps_validation = hps_validation_df['Class']
hps_training_df = training_data.drop(hps_validation_df.index)
X_hps_training = hps_training_df.drop(columns=['Class'])
y_hps_training = hps_training_df['Class']

# Preprocess Data
# Use StandardScaler to scale the data
scaler = sklearn.preprocessing.StandardScaler()
X_hps_training = scaler.fit_transform(X_hps_training)
X_hps_validation = scaler.transform(X_hps_validation)

##################################################################################################

# Function to Evaluate Hyperparameter Settings
def eval_hyperparameter_settings(regularization, class_weight, multi_class_strategy):
    '''
    Evaluate hyperparameter settings using Logistic Regression. Returns validation accuracy.
    '''
    # Fit model
    if regularization in ['l1', 'l2']:
        # Perform grid search with 10 fold cross validation
        parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
              {'penalty':['none', 'elasticnet', 'l1', 'l2']},
              {'C':[0.001, 0.01, 0.1, 1, 10, 100]}]
        clf = GridSearchCV(LogisticRegression(random_state=290, penalty=regularization, class_weight=class_weight, multi_class=multi_class, solver='saga', n_jobs=-1), param_grid, cv=10).fit(X_hps_training, y_hps_training)
    else:
        clf = LogisticRegression(random_state=290, penalty=regularization, class_weight=class_weight, multi_class=multi_class_strategy, solver='saga', n_jobs=-1).fit(X_hps_training, y_hps_training)
    # Return validation accuracy
    return clf.score(X_hps_validation, y_hps_validation)

# Add column to hyperparameters_setting_combos for validation accuracy
hyperparameters_setting_combos['validation_accuracy'] = hyperparameters_setting_combos.apply(lambda row: eval_hyperparameter_settings(row['regularization'], row['class_weight'], row['multi_class_strategy']), axis=1)
print('hyperparameters_setting_combos and accuracies')
print(hyperparameters_setting_combos)

##################################################################################################

# Select Row With Highest Validation Score
highest_accuracy = hyperparameters_setting_combos['validation_accuracy'].max()
best_hyperparameters = dict(hyperparameters_setting_combos[hyperparameters_setting_combos['validation_accuracy'] == highest_accuracy])
print('best hyperparameters')
print(best_hyperparameters)

##################################################################################################

# Fit Model on Full Training Data

# Create matrices for training
X = training_data.drop(columns=['Class'])
y = training_data['Class']

# Preprocess with standard scalar
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Fit model
if best_hyperparameters['regularization'] in ['l1', 'l2']:
    clf = LogisticRegressionCV(cv=10, random_state=290, penalty=best_hyperparameters['regularization'], class_weight=best_hyperparameters['class_weight'], multi_class=best_hyperparameters['multi_class_strategy'], solver='saga', n_jobs=-1).fit(X, y)
else:
    clf = LogisticRegression(random_state=290, penalty=best_hyperparameters['regularization'], class_weight=best_hyperparameters['class_weight'], multi_class=best_hyperparameters['multi_class_strategy'], solver='saga', n_jobs=-1).fit(X, y)

# Save
joblib.dump(clf, "Best Logistic Regression Model.pkl") 

##################################################################################################

# End timer
end_time = time.time()

# Print runtime in minutes
print('runtime in minutes')
print((end_time - start_time) / 60)

# Print runtime in per image
print('runtime per image in minutes')
print(((end_time - start_time) / len(training_data)) / 60)
