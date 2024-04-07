# Packages
import pandas as pd
import os
import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Start timer
start_time = time.time()

# Load Training Data
def combine_directory_parquets(directory_path):
    '''
    Combines all parquet files in a directory into a single dataframe.
    '''
    if directory_path[-1] != '/':
        directory_path += '/'
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(directory_path + f) for f in file_list])
    return combined_df

training_data = combine_directory_parquets('../../../Data/Features/All Features/train')
print('All training data:')
print(training_data)

# Filter data
training_data = training_data[training_data['Class'].isin(['SUV', 'Pickup', 'Sedan', 'Convertible'])]

# Sample data if needed
sample_run = False 
if sample_run:
    training_data = training_data.sample(frac=0.01)

# Hyperparameters for XGBoost
hyperparameter_settings = [
    {
        'learning_rate': [0.01, 0.1, 0.3], 
        'n_estimators': [100, 200],  
        'max_depth': [3, 6], 
        'subsample': [0.8, 1],  
        'colsample_bytree': [0.8, 1],
    }
]

# Create feature matrix (X) and target vector (y)
num_cols = training_data.select_dtypes(include=['float64', 'int64']).columns
X = training_data[num_cols]
y = training_data['Class']

# Encode class labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Grid search with 5-fold cross-validation
gs = GridSearchCV(xgb_clf, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X, y_encoded)
print("Tuned hyperparameters:", gs.best_params_)
print("Accuracy:", gs.best_score_)
print("Best model:", gs.best_estimator_)

# Save the best model
joblib.dump(gs.best_estimator_, 'Best_XGBoost_Model.joblib')

# End timer and print runtime
end_time = time.time()
print('Runtime in minutes:', (end_time - start_time) / 60)
print('Runtime per image in minutes:', ((end_time - start_time) / len(training_data)) / 60)
