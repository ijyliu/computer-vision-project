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

    return X, y_encoded

def fit_xgboost_classifier(X_train, y_train, classifier_name):
    '''
    Fits an XGBoost classifier to the training data matrices.
    '''
    output_dir = '../../../Output/Classifier Fitting/XGBoost/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hyperparameter_settings = [
        {
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
    ]

    start_time = time.time()

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    gs = GridSearchCV(xgb_clf, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1).fit(X_train, y_train)

    end_time = time.time()

    joblib.dump(gs.best_estimator_, output_dir + classifier_name + ' Best Model.joblib')

    runtime_minutes = (end_time - start_time) / 60
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

    print("Hyperparameters searched: ", hyperparameter_settings)
    print("Tuned hyperparameters: ", gs.best_params_)

    joblib.dump(hyperparameter_settings, output_dir + classifier_name + ' Hyperparameter Settings.joblib')
    joblib.dump(gs.best_params_, output_dir + classifier_name + ' Tuned Hyperparameters.joblib')

def make_predictions(test_data, X_test, classifier_name):
    '''
    Makes predictions on the test data using the best XGBoost model.
    '''
    inference_dir = '../../../Output/Classifier Inference/XGBoost/'
    predictions_dir = '../../../Data/Predictions/XGBoost/'

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    best_model = joblib.load(inference_dir + classifier_name + ' Best Model.joblib')

    start_time = time.time()

    predictions = best_model.predict(X_test)

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

    test_data['XGBoost_Classification'] = predictions

    limited_test_data = test_data[[col for col in test_data.columns if col not in test_data.select_dtypes(include=np.number).columns]]

    limited_test_data.to_excel(predictions_dir + 'XGBoost_Classifier_Predictions_' + classifier_name + '.xlsx', index=False)

##################################################################################################

# Sample usage:
# training_data = combine_directory_parquets('path_to_training_data')
# X_train, y_train = prepare_matrices(training_data)
# fit_xgboost_classifier(X_train, y_train, 'XGBoost_Classifier')
# For predictions, prepare test data similarly and use the make_predictions function