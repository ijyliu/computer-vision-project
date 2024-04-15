# Evaluate Autogluon Tabular Only SCF
# Use the Autogluon AutoML library 

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import numpy as np

##################################################################################################

# Make save directories if they do not exist
if not os.path.exists('../../../Data/Predictions/Autogluon'):
    os.makedirs('../../../Data/Predictions/Autogluon')
if not os.path.exists('../../../Output/Classifier Evaluation/Autogluon'):
    os.makedirs('../../../Output/Classifier Evaluation/Autogluon')
if not os.path.exists('../../../Output/Classifier Fitting/Autogluon'):
    os.makedirs('../../../Output/Classifier Fitting/Autogluon')

##################################################################################################

# Load test data
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

# Create test data
test_df = combine_directory_parquets('../../../Data/Features/Top3_Features/test')
print('all test data')
print(test_df)

# Print out column names
print('column names')
for col in test_df.columns:
    print(col)

##################################################################################################

# Limit to numeric columns + 'Class'
test_df_num_cols = list(test_df.select_dtypes(include=np.number).columns)
test_df_cols_to_keep = test_df_num_cols.copy()
test_df_cols_to_keep.append('Class')
test_df_for_prediction = test_df[test_df_cols_to_keep].reset_index(drop=True)

# Print out column names
print('column names after limiting')
for col in test_df_for_prediction.columns:
    print(col)

##################################################################################################

# Load Model
predictor = TabularPredictor.load('../../../Output/Classifier Fitting/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_Top3')
print('loaded predictor')
print(predictor)

##################################################################################################

# Make Predictions

# Convert from pandas to autogluon
test_data = TabularDataset(test_df_for_prediction)
print('converted from pandas')

# Apply test
predictions = predictor.predict(test_data)
# Concatenate with test df string columns
print('test df num cols')
print(test_df_num_cols)
test_df_string_cols = [col for col in test_df.columns if col not in test_df_num_cols]
print('test df string cols')
print(test_df_string_cols)
test_df_string_cols_part = test_df[test_df_string_cols]
#test_df_string_cols_part['Autogluon_Tabular_Best_Model_Prediction'] = predictions
# Use index values to line up
print('predictions')
print(predictions)
print('predictions length')
print(len(predictions))
print('predictions type')
print(type(predictions))
#predictions = pd.concat([test_df_string_cols_part, predictions], axis=1)
predictions = pd.concat([test_df_string_cols_part, predictions.to_frame().T], ignore_index=True)
print('concatenated predictions')
print(predictions)
# Save to Excel
predictions.to_excel('../../../Data/Predictions/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_predictions_Top3.xlsx', index=False)

##################################################################################################

# Evaluation and Leaderboard

# Evaluation
predictor.evaluate(test_data, silent=True)

# Leaderboard of models
leaderboard = predictor.leaderboard(test_data)
# Save to Excel
leaderboard.to_excel('../../../Output/Classifier Evaluation/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_leaderboard_Top3.xlsx', index=False)
print(leaderboard)

##################################################################################################

# Hyperparameters

# Model info including hyperparameters
pred_info = predictor.info()
# Get model hyperparameters
list_of_models = pred_info['model_info'].keys()
# List of dataframes to fill
list_of_dfs = []
# Iterate over models
for model in list_of_models:
    # Get hyperparameters
    hyperparameters = pred_info['model_info'][model]['hyperparameters']
    # Convert to dataframe
    df = pd.DataFrame.from_dict(hyperparameters, orient='index')
    # Add model name
    df['model'] = model
    # Append to list
    list_of_dfs.append(df)
# Concatenate all dataframes
hyperparameters_df = pd.concat(list_of_dfs).reset_index().rename(columns={'index': 'hyperparameter', 0: 'value'})[['model', 'hyperparameter', 'value']]
# Save to Excel
hyperparameters_df.to_excel('../../../Output/Classifier Fitting/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_hyperparameters_Top3.xlsx', index=False)
print(hyperparameters_df)

##################################################################################################

# Feature Importance via Permutation
fi = predictor.feature_importance(test_data)

# Save to Excel
fi.to_excel('../../../Output/Classifier Evaluation/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_feature_importance_Top3.xlsx')

# Print entire df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(fi)
