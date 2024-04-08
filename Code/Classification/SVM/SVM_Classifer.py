import pandas as pd
import numpy as np
import glob
import os
import joblib
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Prepare Training Data
def collect_parquets(train_path):
    parquet_files = glob.glob(f"{train_path}/*.parquet")
    dfs = [pd.read_parquet(file) for file in parquet_files]
    train_df = pd.concat(dfs, ignore_index = True)
    return train_df


##################################################################################################

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

##################################################################################################

def train_SVM(X_train, y_train, classifier_name):
    '''
    Fits an SVM classifier to the training data matrices.
    '''
    output_dir = '../../../Output/Classifier Fitting/SVM/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hyperparameter_grid = {
    'c_values' :  np.logspace(-1, 2, num = 4),
    'kernel_grid' :  ['rbf', 'poly'],
    'gamma_grid' :  np.logspace(-1, 2, num = 4),
    'degree_grid':  [2, 3, 5, 7],
    'k_folds' : 5
    }
    
    # Setup the hyperparameter grid
    param_grid = {
        'C': hyperparameter_grid['c_values'],
        'kernel': hyperparameter_grid['kernel_grid'],
        'gamma': hyperparameter_grid['gamma_grid'],
        'degree': hyperparameter_grid['degree_grid']
    }

    start_time = time.time()

    # Initialize the SVM classifier
    svm = SVC()
    
    # Setup GridSearchCV
    gs = GridSearchCV(svm, param_grid, cv = hyperparameter_grid['k_folds'], scoring = 'accuracy', return_train_score = True)
    
    # Fit the model
    gs.fit(X_train, y_train)

    # Training time
    fit_time = time.time() - start_time
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_model, output_dir + classifier_name + ' Best Model.joblib')

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

    joblib.dump(hyperparameter_settings, output_dir + classifier_name + ' Hyperparameter Settings.joblib')
    joblib.dump(gs.best_params_, output_dir + classifier_name + ' Tuned Hyperparameters.joblib')
    
    ##################################################################################################

def make_predictions(test_data, X_test, classifier_name):
    '''
    Makes predictions on the test data using the best SVM model.
    '''
    inference_dir = '../../../Output/Classifier Inference/SVM/'
    predictions_dir = '../../../Data/Predictions/SVM/'

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

    test_data['SVM_Classification'] = predictions

    limited_test_data = test_data[[col for col in test_data.columns if col not in test_data.select_dtypes(include = np.number).columns]]

    limited_test_data.to_excel(predictions_dir + 'SVM_Classifier_Predictions_' + classifier_name + '.xlsx', index = False)
    ##################################################################################################


