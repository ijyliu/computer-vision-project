# VGG SCF

####################################################################################################

# Flag for whether this is a sample run or not
sample_run = True

####################################################################################################

# Start timer
import time
start_time = time.time()

####################################################################################################

# Packages
#from PIL import Image
import os
import shutil
import warnings
warnings.filterwarnings('ignore')
import pandas
#import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
from compute_vgg16 import *

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('set up device')
print('device: ', device)

####################################################################################################

# Load model

# Load the pretrained VGG16 model
vgg_model = models.vgg16(pretrained=True).features.to(device)
print('loaded model')

print('printing architecture')
print(vgg_model)

# Set the model to evaluation mode
vgg_model.eval()

####################################################################################################

# train images

# Load Non-Blurred Images
train_image_paths = ['../../../Images/train/No Blur/' + img_path for img_path in os.listdir('../../../Images/train/No Blur')]
if sample_run:
    train_image_paths = train_image_paths[:2]
print('first train image path')
print(train_image_paths[0])

# Extract embeddings
train_embeddings = [compute_vgg16(vgg_model, device, img_path) for img_path in train_image_paths]
print('first train embedding')
print(train_embeddings[0])

# Put train_image_paths and train_embeddings into a pandas dataframe
# Get train_embeddings as lists
train_embeddings_lists = [embedding.flatten().tolist() for embedding in train_embeddings]
print('first train embedding list')
print(train_embeddings_lists[0])
print('length of first train embedding list')
print(len(train_embeddings_lists[0]))
print('length train_embeddings_lists')
print(len(train_embeddings_lists))
# Create dataframe for image path and embeddings
train_embeddings_df = pandas.DataFrame(train_embeddings_lists)
train_embeddings_df.insert(0, 'Image Path', train_image_paths)
print('head of te df')
print(train_embeddings_df.head())
print('shape of te df')
print(train_embeddings_df.shape)
# Rename all columns except for 'Image Path'
train_embeddings_df.columns = ['Image Path'] + ['VGG_Embedding_Element_' + str(i) for i in range(len(train_embeddings_lists[0]))]
# Add column test_80_20 with value 0
train_embeddings_df['test_80_20'] = 0
print('train embeddings dataframe')
print(train_embeddings_df)

####################################################################################################

# test images

# Load Non-Blurred Images
test_image_paths = ['../../../Images/test/No Blur/' + img_path for img_path in os.listdir('../../../Images/test/No Blur')]
if sample_run:
    test_image_paths = test_image_paths[:2]
print('first test image path')
print(test_image_paths[0])

# Extract embeddings
test_embeddings = [compute_vgg16(vgg_model, device, img_path) for img_path in test_image_paths]
print('first test embedding')
print(test_embeddings[0])

# Put test_image_paths and test_embeddings into a pandas dataframe
# Get test_embeddings as lists
test_embeddings_lists = [embedding.flatten().tolist() for embedding in test_embeddings]
print('first test embedding list')
print(test_embeddings_lists[0])
# Create dataframe for image path and embeddings
test_embeddings_df = pandas.DataFrame(test_embeddings_lists)
test_embeddings_df.insert(0, 'Image Path', test_image_paths)
# Rename all columns except for 'Image Path'
test_embeddings_df.columns = ['Image Path'] + ['VGG_Embedding_Element_' + str(i) for i in range(len(test_embeddings_lists[0]))]
# Add column test_80_20 with value 1
test_embeddings_df['test_80_20'] = 1
print('test embeddings dataframe')
print(test_embeddings_df)

####################################################################################################

# Concatenate dataframes
vgg_embeddings_df = pandas.concat([train_embeddings_df, test_embeddings_df], ignore_index=True)
print('concatenated embeddings dataframe')
print(vgg_embeddings_df)

# Split dataset into pieces
num_pieces = 8
total_len_pieces = 0
# Create folders if they don't exist
if not os.path.exists(r'../../../Data/Features/VGG'):
    os.makedirs(r'../../../Data/Features/VGG')
if not os.path.exists(r'../../../Data/Features/VGG_Sample'):
    os.makedirs(r'../../../Data/Features/VGG_Sample')
# Delete previous pieces, all contents of '../../../Data/Features/VGG'
folder = r'../../../Data/Features/VGG'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
# Save pieces
if not sample_run:
    for i in range(num_pieces):
        # start index for piece rows
        start_index = i * len(vgg_embeddings_df) // num_pieces
        # end index for piece rows
        end_index = (i + 1) * len(vgg_embeddings_df) // num_pieces
        # get piece
        piece = vgg_embeddings_df[start_index:end_index]
        piece.to_parquet(r'../../../Data/Features/VGG/VGG Embeddings_piece_' + str(i) + '.parquet', index=False)
        print(len(piece))
        total_len_pieces += len(piece)
    # save 100 row sample
    vgg_embeddings_df.sample(100).to_parquet(r'../../../Data/Features/VGG_Sample/VGG Embeddings_sample.parquet', index=False)

# check total piece length and length of vgg_embeddings_df
print('length check')
print(total_len_pieces)
print(len(vgg_embeddings_df))

####################################################################################################

# End timer
end_time = time.time()

# Print time taken in minutes
ttm = (end_time - start_time) / 60
print('Time taken (in minutes):', ttm)

# Time per image
print('Time per image (in minutes):', ttm / len(vgg_embeddings_df))
