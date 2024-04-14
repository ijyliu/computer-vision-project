# Random Forest Classifier Functions

##################################################################################################

# Packages
import pandas as pd
import sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
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

def fit_random_forest_classifier(X_train, y_train, classifier_name):
    '''
    Fits a Random Forest classifier to the training data matrices.
    '''

    # Make output directories if they do not exist
    if not os.path.exists('../../../Output/Classifier Fitting/Random Forest/'):
        os.makedirs('../../../Output/Classifier Fitting/Random Forest/')


    # Hyperparameter Settings
    hyperparameter_settings = {
        'n_estimators': randint(100, 1000),  # Number of trees in random forest
        'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],  # Maximum number of levels in tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'bootstrap': [True, False]  # Method of selecting samples for training each tree
    }

    print('hyperparameter settings')
    print(hyperparameter_settings)

    # Start timer
    start_time = time.time()

    # Fit model
    # Perform grid search with 5 fold cross validation
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X_train, y_train)

    # End timer
    end_time = time.time()

    # Dump the best model to a file
    joblib.dump(gs.best_estimator_, '../../../Output/Classifier Fitting/Random Forest/' + classifier_name + ' Best Model.joblib', compress=True)

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
    training_statistics_df.to_excel('../../../Output/Classifier Fitting/Random Forest/' + classifier_name + ' Training Statistics.xlsx')


    # Hyperparameters
    print("hyperparameters searched: ", hyperparameter_settings)
    tuned_hyperparameters = gs.best_params_
    print("tuned hyperparameters: ", tuned_hyperparameters)
    # Output dictionaries
    joblib.dump(hyperparameter_settings, '../../../Output/Classifier Fitting/Random Forest/' + classifier_name + ' Hyperparameter Settings.joblib')
    joblib.dump(tuned_hyperparameters, '../../../Output/Classifier Fitting/Random Forest/' + classifier_name + ' Tuned Hyperparameters.joblib')

def make_predictions(test_data, X_test, classifier_name):
    '''
    Makes predictions on the test data using the best model.
    '''

    # Make output directories if they do not exist
    if not os.path.exists('../../../Output/Classifier Inference/Random Forest/'):
        os.makedirs('../../../Output/Classifier Inference/Random Forest/')
    if not os.path.exists('../../../Data/Predictions/Random Forest/'):
        os.makedirs('../../../Data/Predictions/Random Forest/')

    # Load model
    best_model = joblib.load('../../../Output/Classifier Fitting/Random Forest/' + classifier_name + ' Best Model.joblib')
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
    prediction_statistics_df.to_excel('../../../Output/Classifier Inference/Random Forest/' + classifier_name + ' Prediction Statistics.xlsx', index=False)
    print('saved prediction statistics')

    # Keep only string items in test_data
    limited_test_data = test_data.copy()
    #limited_test_data = limited_test_data[[col for col in limited_test_data.columns if col not in limited_test_data.select_dtypes(include=np.number).columns]]
    limited_test_data = limited_test_data[['Class', 'harmonized_filename', 'image_path_blur', 'image_path_no_blur']]
    print('limited cols in test data')

    # Add to test_data
    limited_test_data['Random_Forest_Classification'] = predictions
    print('added to test data')

    # Save to excel
    limited_test_data.to_excel('../../../Data/Predictions/Random Forest/Random_Forest_Classifier_Predictions_' + classifier_name + '.xlsx', index=False)
