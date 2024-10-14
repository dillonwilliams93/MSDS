# pytorch works best with a folder structure that is already categorized into classes
# this function will categorize the images into folders
import pandas as pd
import os
from pathlib import Path

# establish root directory
project_root = '/Users/dillonwilliams/pycharmprojects/Cancer Detection/'

# read in the labels
labels = pd.read_csv(project_root + 'data/train_labels.csv')

# create a folder for each class
for i in range(2):
    os.makedirs(f'{project_root}data/train/{i}',
                exist_ok=True)

# move the images to the correct folder
for i in range(len(labels)):
    image = labels.iloc[i][0]
    label = labels.iloc[i][1]
    os.rename(f'{project_root}data/train/{image}.tif',
              f'{project_root}data/train/{label}/{image}.tif')
