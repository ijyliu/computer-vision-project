
# Import libraries
import os
import time
import pandas as pd
import shutil
from Image_Processing_Functions import *

# Load "~/Box/INFO 290T Project/Intermediate Data/resized_cars_annos.xlsx"
# This file contains the paths to the original and resized images
resized_cars_annos = pd.read_excel('~/Box/INFO 290T Project/Intermediate Data/resized_cars_annos.xlsx')

# Clear full directory
shutil.rmtree(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/"), ignore_errors=True)
# Make subdirectories
os.makedirs(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/train"), exist_ok=True)
os.makedirs(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/test"), exist_ok=True)

# Start timer
full_start_time = time.time()

# Iterate through the full set and process each image
for _, row in resized_cars_annos.iterrows():
    process_image(row['source_file_path'], row['destination_file_path'])

# End timer
full_end_time = time.time()

# Full time taken in minutes
print('Full time taken: ', (full_end_time - full_start_time) / 60, ' minutes')

# Check success of operation - number of files in the source and destination directories
print('number of files to resize: ', len(resized_cars_annos))
print('number of resized files: ', len(os.listdir(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/train"))) + len(os.listdir(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/test"))))
print('items above 260 w and h: ', len(resized_cars_annos.query('width > 260 and height > 260')))
