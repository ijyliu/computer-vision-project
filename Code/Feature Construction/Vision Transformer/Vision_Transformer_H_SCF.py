# Vision Transformer H SCF
# Create embeddings with vision transformer huge model on SCF

####################################################################################################

# Start timer
import time
start_time = time.time()

####################################################################################################

# Packages
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
from transformers import ViTImageProcessor
from transformers import ViTModel
import torch
import pandas

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('set up device')

####################################################################################################

# Load model
vision_transformer = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
vision_transformer.to(device)
print('loaded model')

# Load processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print('loaded processor')

# Function to extract embedding given an image path
def get_embedding(image_path):
    # Load image
    image = Image.open(image_path)
    # Pre-process image
    preprocessed_image = processor(images=image, return_tensors="pt")
    # Move the image to the device
    image_on_device = preprocessed_image.to(device)
    # Run the image through the model
    with torch.no_grad():
        outputs = vision_transformer(**image_on_device)
    # Extract the pooler output
    return outputs.pooler_output

####################################################################################################

# train images

# Load Non-Blurred Images
train_image_paths = ['../../../Images/train/No Blur/' + img_path for img_path in os.listdir('../../../Images/train/No Blur')]
print('first train image path')
print(train_image_paths[0])

# Extract embeddings
train_embeddings = [get_embedding(img_path) for img_path in train_image_paths]
print('first train embedding')
print(train_embeddings[0])

# Put train_image_paths and train_embeddings into a pandas dataframe
# Get train_embeddings as lists
train_embeddings_lists = [embedding.flatten().tolist() for embedding in train_embeddings]
print('first train embedding list')
print(train_embeddings_lists[0])
# Create dataframe for image path and embeddings
train_embeddings_df = pandas.DataFrame(train_embeddings_lists)
train_embeddings_df.insert(0, 'Image Path', train_image_paths)
# Rename all columns except for 'Image Path'
train_embeddings_df.columns = ['Image Path'] + ['ViT_Embedding_Element_' + str(i) for i in range(len(train_embeddings_lists))]
# Add column test_80_20 with value 0
train_embeddings_df['test_80_20'] = 0
print('train embeddings dataframe')
print(train_embeddings_df)

####################################################################################################

# test images

# Load Non-Blurred Images
test_image_paths = ['../../../Images/test/No Blur/' + img_path for img_path in os.listdir('../../../Images/test/No Blur')]
print('first test image path')
print(test_image_paths[0])

# Extract embeddings
test_embeddings = [get_embedding(img_path) for img_path in test_image_paths]
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
test_embeddings_df.columns = ['Image Path'] + ['ViT_Embedding_Element_' + str(i) for i in range(len(test_embeddings_lists))]
# Add column test_80_20 with value 1
test_embeddings_df['test_80_20'] = 1
print('test embeddings dataframe')
print(test_embeddings_df)

####################################################################################################

# Concatenate dataframes
vit_embeddings_df = pandas.concat([train_embeddings_df, test_embeddings_df], ignore_index=True)
print('concatenated embeddings dataframe')
print(vit_embeddings_df)

# Save dataframe
vit_embeddings_df.to_parquet('../../../Data/Features/Vision Transformer/Vision Transformer Embeddings.parquet')

####################################################################################################

# End timer
end_time = time.time()

# Print time taken in minutes
ttm = (end_time - start_time) / 60
print('Time taken (in minutes):', ttm)

# Time per image
print('Time per image (in minutes):', ttm / len(vit_embeddings_df))
