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
sample_run = True

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
# Penalty: none, L1, L2
# A bunch of C values
# Balanced Class Weights: yes, no
# Multi-Class Strategy: one-vs-rest, multinomial
# A bunch of solvers
hyperparameter_settings = [{'penalty':[None, 'elasticnet', 'l1', 'l2']},
                           {'C':[0.001, 0.01, 0.1, 1, 10, 100]},
                           {'class_weight':['balanced', None]},
                           {'multi_class':['ovr', 'multinomial']},
                           {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
print('hyperparameter settings')
print(hyperparameter_settings)

##################################################################################################

# Create matrices for training
X = training_data.drop(columns=['Class'])
y = training_data['Class']

##################################################################################################

# Preprocess with standard scalar
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)

##################################################################################################

# Fit model
# Perform grid search with 5 fold cross validation
clf = GridSearchCV(LogisticRegression(), hyperparameter_settings, cv=5).fit(X, y)

print("tuned hyperparameters: ", clf.best_params_)
print("accuracy: ", clf.best_score_)

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
