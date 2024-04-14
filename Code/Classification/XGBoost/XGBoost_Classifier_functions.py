# Packages
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def combine_directory_parquets(directory_path):
    '''
    Combines all parquet files in a directory into a single dataframe.
    '''
    if directory_path[-1] != '/':
        directory_path += '/'
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(directory_path + f) for f in file_list])
    return combined_df

def prepare_matrices(data):
    '''
    Takes in a dataframe, preprocesses it, and returns X and y matrices.
    '''
    num_cols = data.select_dtypes(include=np.number).columns
    X = data[num_cols]
    y = data['Class']

    # Optionally skip scaling if preferred for XGBoost
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode class labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_encoded,label_encoder,scaler

def fit_xgboost_classifier(X_train, y_train, classifier_name):
    '''
    Fits an XGBoost classifier to the training data matrices.
    '''
    output_dir = '../../../Output/Classifier Fitting/XGBoost/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hyperparameter_grid =  {
            'n_estimators': [100, 200,300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1,3,5],
            'k_folds': 5,
    }         
    
    param_grid = {
        'n_estimators': hyperparameter_grid['n_estimators'],
        'learning_rate': hyperparameter_grid['learning_rate'],
        'max_depth': hyperparameter_grid['max_depth'],
        'min_child_weight': hyperparameter_grid['min_child_weight']
    }

    start_time = time.time()

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    gs = GridSearchCV(xgb_clf, param_grid, cv=hyperparameter_grid['k_folds'], scoring='accuracy', return_train_score=True, n_jobs=-1)
    
    gs.fit(X_train, y_train)
    
    fit_time = time.time() - start_time
    
    best_model = gs.best_estimator_
    
    joblib.dump(best_model, output_dir + classifier_name + ' Best Model.joblib', compress=True)
    
    
    runtime_minutes = fit_time / 60
    print("Training time in minutes: ", runtime_minutes)
    runtime_per_image = runtime_minutes / len(y_train)
    print("Training time per image in minutes: ", runtime_per_image)
    train_accuracy_best_model = gs.best_estimator_.score(X_train, y_train)
    print("Train accuracy of best model: ", train_accuracy_best_model)
    mean_cross_validated_accuracy = gs.best_score_
    print("Mean cross validated accuracy of best model: ", mean_cross_validated_accuracy)
    
    
    training_statistics_df = pd.DataFrame({
        'runtime_minutes': [runtime_minutes],
        'runtime_per_image': [runtime_per_image],
        'train_accuracy_best_model': [train_accuracy_best_model],
        'mean_cross_validated_accuracy': [mean_cross_validated_accuracy]
    })
    
    
    training_statistics_df.to_excel(output_dir + classifier_name + ' Training Statistics.xlsx')

    print("Hyperparameters searched: ", hyperparameter_grid)
    print("Tuned hyperparameters: ", gs.best_params_)

    joblib.dump(hyperparameter_grid, output_dir + classifier_name + ' Hyperparameter Settings.joblib')
    joblib.dump(gs.best_params_, output_dir + classifier_name + ' Tuned Hyperparameters.joblib')

  
def make_predictions(test_data, X_test, classifier_name,label_encoder):
    '''
    Makes predictions on the test data using the best XGBoost model.
    '''
    output_dir = '../../../Output/Classifier Fitting/XGBoost/'
    inference_dir = '../../../Output/Classifier Inference/XGBoost/'
    predictions_dir = '../../../Data/Predictions/XGBoost/'

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    best_model = joblib.load(output_dir + classifier_name + ' Best Model.joblib')

    start_time = time.time()

    predictions_encoded = best_model.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    

    end_time = time.time()

    runtime_minutes = (end_time - start_time) / 60
    print("Prediction time in minutes: ", runtime_minutes)
    runtime_per_image = runtime_minutes / len(test_data)
    print("Prediction time per image in minutes: ", runtime_per_image)

    prediction_statistics_df = pd.DataFrame({
        'runtime_minutes': [runtime_minutes],
        'runtime_per_image': [runtime_per_image]
    })

    prediction_statistics_df.to_excel(inference_dir + classifier_name + ' Prediction Statistics.xlsx', index=False)

    # Keep only string items in test_data
    limited_test_data = test_data.copy()
    #limited_test_data = limited_test_data[[col for col in limited_test_data.columns if col not in limited_test_data.select_dtypes(include=np.number).columns]]
    limited_test_data = limited_test_data[['Class', 'harmonized_filename', 'image_path_blur', 'image_path_no_blur']]
    print('limited cols in test data')

    # Add to test_data
    limited_test_data['XGBoost_Classification'] = predictions
    print('added to test data')

    limited_test_data.to_excel(predictions_dir + 'XGBoost_Classifier_Predictions_' + classifier_name + '.xlsx', index=False)

##################################################################################################

# Sample usage:
# training_data = combine_directory_parquets('path_to_training_data')
# X_train, y_train = prepare_matrices(training_data)
# fit_xgboost_classifier(X_train, y_train, 'XGBoost_Classifier')
# For predictions, prepare test data similarly and use the make_predictions function
