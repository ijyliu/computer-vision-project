# Logistic Regression Classifier Functions

##################################################################################################

# Packages
import pandas as pd
import sklearn
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import joblib
import time
import numpy as np

##################################################################################################

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

def prepare_matrices(data):
    '''
    Takes in a dataframe and returns X and y matrices.
    '''
    # Create matrices for training
    # X is all numeric columns, y is 'Class'
    num_cols = data.select_dtypes(include=np.number).columns
    X = data[num_cols]
    y = data['Class']

    # Preprocess with standard scalar
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def fit_logistic_regression_classifier(X_train, y_train, classifier_name):
    '''
    Fits a logistic regression classifier to the training data matrices.
    '''

    # Make output directories if they do not exist
    if not os.path.exists('../../../Output/Classifier Fitting/Logistic Regression/'):
        os.makedirs('../../../Output/Classifier Fitting/Logistic Regression/')

    # Hyperparameter Settings
    hyperparameter_settings = [
        # Non-penalized
        {'solver': ['saga'], 
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

    # Start timer
    start_time = time.time()

    # Fit model
    # Perform grid search with 5 fold cross validation
    lr = LogisticRegression(max_iter=1000) # higher to encourage convergence
    gs = GridSearchCV(lr, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X_train, y_train)

    # End timer
    end_time = time.time()

    # Dump the best model to a file
    joblib.dump(gs.best_estimator_, '../../../Output/Classifier Fitting/Logistic Regression/' + classifier_name + ' Best Model.joblib', compress=True)

    # Statistics
    runtime_minutes = (end_time - start_time) / 60
    print("training time in minutes: ", runtime_minutes)
    runtime_per_image = runtime_minutes / len(y_train)
    print("training time per image in minutes: ", runtime_per_image)
    train_accuracy_best_model = gs.best_estimator_.score(X_train, y_train)
    print("train accuracy of best model: ", train_accuracy_best_model)
    mean_cross_validated_accuracy = gs.best_score_
    print("mean cross validated accuracy of best model: ", mean_cross_validated_accuracy)
    # Create dataframe
    training_statistics_df = pd.DataFrame({
        'runtime_minutes': [runtime_minutes],
        'runtime_per_image': [runtime_per_image],
        'train_accuracy_best_model': [train_accuracy_best_model],
        'mean_cross_validated_accuracy': [mean_cross_validated_accuracy]
    })
    # Output to Excel
    training_statistics_df.to_excel('../../../Output/Classifier Fitting/Logistic Regression/' + classifier_name + ' Training Statistics.xlsx')

    # Hyperparameters
    print("hyperparameters searched: ", hyperparameter_settings)
    tuned_hyperparameters = gs.best_params_
    print("tuned hyperparameters: ", tuned_hyperparameters)
    # Output dictionaries
    joblib.dump(hyperparameter_settings, '../../../Output/Classifier Fitting/Logistic Regression/' + classifier_name + ' Hyperparameter Settings.joblib')
    joblib.dump(tuned_hyperparameters, '../../../Output/Classifier Fitting/Logistic Regression/' + classifier_name + ' Tuned Hyperparameters.joblib')

def make_predictions(test_data, X_test, classifier_name):
    '''
    Makes predictions on the test data using the best model.
    '''

    # Make output directories if they do not exist
    if not os.path.exists('../../../Output/Classifier Inference/Logistic Regression/'):
        os.makedirs('../../../Output/Classifier Inference/Logistic Regression/')
    if not os.path.exists('../../../Data/Predictions/Logistic Regression/'):
        os.makedirs('../../../Data/Predictions/Logistic Regression/')

    # Load model
    best_model = joblib.load('../../../Output/Classifier Fitting/Logistic Regression/' + classifier_name + ' Best Model.joblib')
    print('best model')
    print(best_model)

    # Start timer
    start_time = time.time()

    # Predictions
    predictions = best_model.predict(X_test)
    print('predictions head')
    print(predictions[:5])

    # End timer
    end_time = time.time()

    # Statistics
    runtime_minutes = (end_time - start_time) / 60
    print("prediction time in minutes: ", runtime_minutes)
    runtime_per_image = runtime_minutes / len(test_data)
    print("prediction time per image in minutes: ", runtime_per_image)
    # Create dataframe
    prediction_statistics_df = pd.DataFrame({
        'runtime_minutes': [runtime_minutes],
        'runtime_per_image': [runtime_per_image]
    })
    # Output to Excel
    prediction_statistics_df.to_excel('../../../Output/Classifier Inference/Logistic Regression/' + classifier_name + ' Prediction Statistics.xlsx', index=False)

    # Add to test_data
    test_data['Logistic_Regression_Classification'] = predictions

    # Keep only string items in test_data
    limited_test_data = test_data[[col for col in test_data.columns if col not in test_data.select_dtypes(include=np.number).columns]]

    # Save to excel
    limited_test_data.to_excel('../../../Data/Predictions/Logistic Regression/Logistic_Regression_Classifier_Predictions_' + classifier_name + '.xlsx', index=False)
