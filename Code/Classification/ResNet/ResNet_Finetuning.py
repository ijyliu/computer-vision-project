# ResNet Fine Tuning
# Fine tune an end-to-end ResNet model on the cars dataset.

##################################################################################################

# Whether this is a sample run or not
sample_run = True

# Packages
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('set up device')
print('device: ', device)

##################################################################################################

# Set Up Images and Labels
# Non-blurred

# Load data
directory_path = '../../../Data/Features/All Features/train/'
# list of files in directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
# read in all parquet files
combined_df = pd.concat([pd.read_parquet(directory_path + f, columns=['Class', 'harmonized_filename']) for f in file_list])
# replace .jpg with _no_blur.jpg
combined_df['harmonized_filename'] = combined_df['harmonized_filename'].str.replace('.jpg', '_no_blur.jpg')
# reset index
combined_df.reset_index(drop=True, inplace=True)
# create a dictionary mapping to encode the labels
label_mapping = {cl: num for num, cl in enumerate(combined_df['Class'].unique())}
print(label_mapping)
# save label mapping
pd.DataFrame(label_mapping.items(), columns=['Class', 'Encoded']).to_excel('../../../Output/Classifier Fitting/ResNet/label_mapping.xlsx', index=False)
# encode the labels
combined_df['Class'] = combined_df['Class'].map(label_mapping)
print(combined_df)

# Split into training and validation
# 80% training, 20% validation
# Use seed 290
train_df = combined_df.sample(frac=0.8, random_state=290)
val_df = combined_df.drop(train_df.index)
print(train_df)
print(val_df)

# Get image paths and labels
if sample_run:
    train_image_paths = ['../../../Images/train/No Blur/' + hf for hf in train_df['harmonized_filename'][:2]]
    train_image_labels = list(train_df['Class'][:2])
    val_image_paths = ['../../../Images/train/No Blur/' + hf for hf in val_df['harmonized_filename'][:2]]
    val_image_labels = list(val_df['Class'][:2])
else:
    train_image_paths = ['../../../Images/train/No Blur/' + hf for hf in train_df['harmonized_filename']]
    train_image_labels = list(train_df['Class'])
    val_image_paths = ['../../../Images/train/No Blur/' + hf for hf in val_df['harmonized_filename']]
    val_image_labels = list(val_df['Class'])

# Check path and label
print(train_image_paths[0])
print(train_image_labels[0])
print(val_image_paths[0])
print(val_image_labels[0])

# Check counts
print(len(train_image_paths))
print(len(train_image_labels))
print(len(val_image_paths))
print(len(val_image_labels))

##################################################################################################

# PyTorch Dataset and Preprocessing

# Define the custom dataset
class CustomImageDataset(Dataset):

    # Initialize the dataset object with paths, labels and transform
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    # Get the length of the dataset
    def __len__(self):
        return len(self.image_paths)

    # Get an item from the dataset
    def __getitem__(self, idx):

        # Read the image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Apply the transform
        if self.transform:
            image = self.transform(image)
        
        # Get the label
        label = self.labels[idx]

        # Return the image
        return image, label

# Standard ResNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the training dataset
train_dataset = CustomImageDataset(image_paths=train_image_paths, labels=train_image_labels, transform=preprocess)
print(train_dataset[0])

# Create the validation dataset
val_dataset = CustomImageDataset(image_paths=val_image_paths, labels=val_image_labels, transform=preprocess)
print(val_dataset[0])

# Print length of train and val datasets
print(len(train_dataset))
print(len(val_dataset))

# Set up the data loaders
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

##################################################################################################

# Load Model

# Load the model
model = models.resnet50(pretrained=True)
# Modify the final layer for our classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(combined_df['Class'].unique()))
# Send the model to the device
model = model.to(device)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the number of epochs
num_epochs = 10

##################################################################################################

# Training

# Train the model
for epoch in range(num_epochs):

    print('Starting Epoch:', epoch + 1, 'of', num_epochs)

    # Start timer
    start_time = time.time()

    # Train the model on the training set
    model.train()
    train_loss = 0.0
    train_correct = 0
    # Iterate over batches
    for inputs, labels in train_data_loader:
        
        # Move the data to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + loss calc + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update the training loss and accuracy
        # Multiply loss.item() (average loss in the batch) by batch size to get total loss
        train_loss += loss.item() * inputs.size(0)
        # Get the predicted class as the column index with the highest score
        _, preds = torch.max(outputs, 1)
        # Add the number of correct predictions to the total correct
        train_correct += torch.sum(preds == labels.data).item()

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    # No gradients
    with torch.no_grad():
        # Iterate over batches
        for inputs, labels in val_data_loader:

            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the test loss and accuracy
            # Multiply loss.item() (average loss in the batch) by batch size to get total loss
            val_loss += loss.item() * inputs.size(0)
            # Get the predicted class as the column index with the highest score
            _, preds = torch.max(outputs, 1)
            # Add the number of correct predictions to the total correct
            val_correct += torch.sum(preds == labels.data).item()

    # Print the training and test loss and accuracy
    # Renormalize the loss
    train_loss = train_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    # Calculate accuracy
    train_acc = train_correct / len(train_dataset)
    val_acc = val_correct / len(val_dataset)
    # End timer
    end_time = time.time()
    # Print
    print(f"Epoch {epoch + 1} of {num_epochs} | Time Taken: {str(round(end_time - start_time, 2)) + 's'} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

##################################################################################################

# Save model
torch.save(model.state_dict(), '../../../Output/Classifier Fitting/ResNet/resnet.pth')
