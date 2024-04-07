##################################################################################################
# Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import time
import os
import pandas as pd

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
# Drop Image Path, test_80_20
training_data = training_data.drop(columns=['Image Path', 'test_80_20'])
print('all training data')
print(training_data)

# If sample run, take a 1% sample of the data
if sample_run:
    training_data = training_data.sample(frac=0.01)

##################################################################################################

# Hyperparameter Settings
hyperparameter_settings = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False], 
    'oob_score': [True, False],
    'n_jobs': [-1]
}

print('hyperparameter settings')
print(hyperparameter_settings)

##################################################################################################

# Create matrices for training
X = training_data.drop(columns=['Class'])
y = training_data['Class']

##################################################################################################

# Fit model
# Perform grid search with 5 fold cross validation
rf = RandomForestClassifier() 
gs = GridSearchCV(rf, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X, y)

print("tuned hyperparameters: ", gs.best_params_)
print("accuracy: ", gs.best_score_)
print("best model: ", gs.best_estimator_)

# Dump the best model to a file
joblib.dump(gs.best_estimator_, 'Best Random Forest Model.joblib')

##################################################################################################

# End timer
end_time = time.time()

# Print runtime in minutes
print('runtime in minutes')
print((end_time - start_time) / 60)

# Print runtime per image
print('runtime per image in minutes')
print(((end_time - start_time) / len(training_data)) / 60)
