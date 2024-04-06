# Fit Logistic Regression Classifier

##################################################################################################

# Packages
import pandas as pd
import sklearn
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import time

##################################################################################################

# Sample run or not
sample_run = False

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
print('all training data')
print(training_data)

# Keep where 'Class' is in 'SUV', 'Pickup', 'Sedan', 'Convertible'
training_data = training_data[training_data['Class'].isin(['SUV', 'Pickup', 'Sedan', 'Convertible'])]

# If sample run, take a 1% sample of the data
if sample_run:
    training_data = training_data.sample(frac=0.01)

##################################################################################################

# Hyperparameter Settings
hyperparameter_settings = [
    # Non-penalized
    {'solver': ['lbfgs'], 
     'penalty': [None], 
     'C': [1],  # C is irrelevant here but required as a placeholder
     'class_weight': [None, 'balanced'], 
     'multi_class': ['ovr', 'multinomial']},
    # ElasticNet penalty
    {'solver': ['saga'], 
     'penalty': ['elasticnet'], 
     'C': [0.001, 0.01, 0.1, 1, 10, 100], 
     'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0], 
     'class_weight': [None, 'balanced'], 
     'multi_class': ['ovr', 'multinomial']}
]
print('hyperparameter settings')
print(hyperparameter_settings)

##################################################################################################

# Create matrices for training
# X is all numeric columns, y is 'Class'
num_cols = training_data.select_dtypes(include=['float64', 'int64']).columns
X = training_data[num_cols]
y = training_data['Class']

##################################################################################################

# Preprocess with standard scalar
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)

##################################################################################################

# Fit model
# Perform grid search with 5 fold cross validation
lr = LogisticRegression(max_iter=1000) # higher to encourage convergence
gs = GridSearchCV(lr, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X, y)

print("tuned hyperparameters: ", gs.best_params_)
print("accuracy: ", gs.best_score_)
print("best model: ", gs.best_estimator_)

# Dump the best model to a file
joblib.dump(gs.best_estimator_, 'Best Logistic Regression Model.joblib')

##################################################################################################

# End timer
end_time = time.time()

# Print runtime in minutes
print('runtime in minutes')
print((end_time - start_time) / 60)

# Print runtime in per image
print('runtime per image in minutes')
print(((end_time - start_time) / len(training_data)) / 60)
