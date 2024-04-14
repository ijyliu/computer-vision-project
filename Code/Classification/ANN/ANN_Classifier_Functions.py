
# Import Required Libraries
import pandas as pd
import numpy as np
import glob
import os
import joblib
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

##################################################################################################

# Prepare Training Data
def combine_directory_parquets(directory_path):
    '''
    Combines all parquet files in a directory into a single dataframe.
    '''
    if directory_path[-1] != '/':
        directory_path += '/'
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(directory_path + f) for f in file_list])
    return combined_df


##################################################################################################

def prepare_matrices(data, classifier_name):
    '''
    Takes in a dataframe and returns X and y matrices.
    '''
    output_dir = '../../../Output/Classifier Fitting/ANN/'
    # Create matrices for training
    # X is all numeric columns, y is 'Class'
    num_cols = data.select_dtypes(include=np.number).columns
    X = data[num_cols]
    y = data['Class']

    # Preprocess with standard scalar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode class labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Save encoder for converting predictions
    joblib.dump(label_encoder, output_dir + classifier_name + 'label_encoder.joblib')

    return X, y

##################################################################################################

def fit_ann_classifier(X_train, y_train, classifier_name):
    output_dir = '../../../Output/Classifier Fitting/ANN/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hyperparameter_grid = {
        'learning_rate': [0.01, 0.001],
        'batch_size': [32, 64, 128],
        'optimizer': ['rmsprop', 'adam', 'SGD'],
        'epochs': [10, 50]
    }
    
    # Create the ANN model - Used Keras instead of Pytorch so that we could use GRIDSEARCHCV
    def build_model(optimizer = 'adam', learning_rate = 0.01):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape = (511,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        
        # Select optimizer
        optimizers = {
            'rmsprop': RMSprop(learning_rate=learning_rate),
            'adam': Adam(learning_rate=learning_rate),
            'SGD': SGD(learning_rate=learning_rate)
        }
        model.compile(optimizer=optimizers[optimizer], loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    # Create a KerasClassifier object
    model = KerasClassifier(build_fn = build_model, verbose = 0)

    # Print device
    print("Device: ", model.device)

    # Setup GridSearchCV
    param_grid = {
        'optimizer': hyperparameter_grid['optimizer'],
        'learning_rate': hyperparameter_grid['learning_rate'],
        'batch_size': hyperparameter_grid['batch_size'],
        'epochs': hyperparameter_grid['epochs']
    }
    
    start_time = time.time()

    # Initialize GridSearchCV
    gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
    
    # Fit the model
    gs.fit(X_train, y_train)

    # Training time
    fit_time = time.time() - start_time
    
    # Best model
    best_model = gs.best_estimator_
    
    # Save the model
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
        
    ##################################################################################################

def make_predictions(test_data, X_test, classifier_name):
    '''
    Makes predictions on the test data using the best ANN model.
    '''
    output_dir = '../../../Output/Classifier Fitting/ANN/'
    inference_dir = '../../../Output/Classifier Inference/ANN/'
    predictions_dir = '../../../Data/Predictions/ANN/'

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Load the best ANN model
    best_model = load_model(output_dir + classifier_name + ' Best Model.joblib')

    start_time = time.time()

    # Making predictions
    predictions = best_model.predict(X_test)
    
    # Convert predictions from probabilities to class labels
    predicted_classes = np.argmax(predictions, axis = 1)
    
    # Transform predictions to class labels
    label_encoder = joblib.load(output_dir + classifier_name + 'label_encoder.joblib')
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    end_time = time.time()

    runtime_minutes = (end_time - start_time) / 60
    print("Prediction time in minutes: ", runtime_minutes)
    runtime_per_image = runtime_minutes / len(test_data)
    print("Prediction time per image in minutes: ", runtime_per_image)

    prediction_statistics_df = pd.DataFrame({
        'runtime_minutes': [runtime_minutes],
        'runtime_per_image': [runtime_per_image]
    })

    prediction_statistics_df.to_excel(inference_dir + classifier_name + ' Prediction Statistics.xlsx', index = False)

    # Add predictions to test data
    test_data['ANN_Classification'] = predicted_classes

    # Filter out numerical columns if desired to limit data in final output
    limited_test_data = test_data[[col for col in test_data.columns if col not in test_data.select_dtypes(include = np.number).columns]]

    limited_test_data.to_excel(predictions_dir + 'ANN_Classifier_Predictions_' + classifier_name + '.xlsx', index = False)

    
    ##################################################################################################

# Sample usage:
# training_data = combine_directory_parquets('path_to_training_data')
# X_train, y_train = prepare_matrices(training_data, 'VGG')
# fit_ann_classifier(X_train, y_train, 'VGG')
# For predictions, prepare test data similarly and use the make_predictions function

