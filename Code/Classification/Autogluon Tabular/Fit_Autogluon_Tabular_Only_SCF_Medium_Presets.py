# Fit Autogluon Tabular Only SCF
# Use the Autogluon AutoML library

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import numpy as np

##################################################################################################

# Load train data
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

# Create training data
train_df = combine_directory_parquets('../../../Data/Features/All Features/train')
print('all training data')
print(train_df)

# Print out column names
print('column names')
for col in train_df.columns:
    print(col)

##################################################################################################

# Limit to numeric columns + 'Class'
train_df_num_cols = list(train_df.select_dtypes(include=np.number).columns)
train_df_cols_to_keep = train_df_num_cols
train_df_cols_to_keep.append('Class')
train_df = train_df[train_df_cols_to_keep]

# Print out column names
print('column names')
for col in train_df.columns:
    print(col)

##################################################################################################

# Fit AutoGluon

# Convert from pandas to autogluon
train_data = TabularDataset(train_df)

# Create model save directory if it doesn't exist
os.makedirs('../../../Output/Classifier Fitting/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets', exist_ok=True)

# Fit models
# Set seed to try to encourage stability
import numpy as np
np.random.seed(222)
# Run predictor
predictor = TabularPredictor(label='Class', path='../../../Output/Classifier Fitting/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets').fit(train_data=train_data, presets='medium_quality', excluded_model_types = ['KNN'])
