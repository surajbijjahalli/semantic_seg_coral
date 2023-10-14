#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
import seaborn as sns
import glob
import random
from pathlib import Path
from Fast_MSS import *
from MLC_utils import *
import pathlib
from datasets import load_dataset

from huggingface_hub import login
'''
login(token = "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")

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("segments/sidewalk-semantic")

#%%

example_label = np.array(dataset["train"][1]["label"])
print("Unique semantic labels in this image: ", np.unique(example_label))
fig,axs = plt.subplots()
axs.imshow(example_label)
'''
#%%

parser = argparse.ArgumentParser("Visualize sparse labels")
parser.add_argument("--path_img_dir",type=str,help="path to image directory")
parser.add_argument("--path_annotations",type=str,help="path to sparse annotations")


args = parser.parse_args()
# Path to the directory of images
path_to_img_dir = args.path_img_dir

# Path to the sparse annotation csv file
path_to_annotation_file = args.path_annotations


# List of paths for each file in the image directory
list_imgs = glob.glob(path_to_img_dir+"*.jpg")

# Pick a random file path from the list
rand_sample = random.choice(list_imgs)

# extract the file name without the extension and convert to an integer - this is necessary since we use the file name as a key for grabbing the sparse annotations
img_name = int(Path(rand_sample).stem)





# Read in the entire annotation file
label_dataframe = pd.read_csv(path_to_annotation_file)

# grab all unique labels in the entire annotation file
unique_labels = list(label_dataframe['label'].unique())

# Create dictionary keys for each unique label
dict_keys = list(np.array(range(len(unique_labels))).astype(str))

# Create a dictionary of labels
dict_of_labels = dict(zip(dict_keys,unique_labels))

# Grab the annotations for the sampled random image
labels = label_dataframe[label_dataframe['quadratid']==img_name].copy()

# Grab the x-y coordinates of each annotation
labels_coordinates = labels[['x','y']]

# Path to image - same as the randomly sampled image path
path_to_img = rand_sample #path_to_img_dir+str(img_name)+".jpg"

# Read in the image
img = plt.imread(path_to_img)

def rescale_image(img,labels,new_width):
    # Rescale image maintaining aspect ratio and adjust annotation coordinates
    
    #Get aspect ratio
    old_h,old_w = img.shape[:2]
    aspect_ratio = old_h/old_w

    new_height = aspect_ratio*new_width
    rescaled_image = cv2.resize(img,(int(new_height),int(new_width)))
    #old_x = labels['x'].to_numpy()
    #old_y = labels['y'].to_numpy()
    #new_x = old_x * new_width/old_w
    #new_y = old_y * new_height/old_h
    labels['x'] = labels['x'].multiply(new_width/old_w)
    labels['y'] = labels['y'].multiply(new_height/old_h)
    
    return rescaled_image,labels



rescaled_img,labels = rescale_image(img,labels,1030)



# Create a new dataframe by renaming the columns to be consistent with Fast_mss library
new_labels = labels.rename(columns={"x": "X", "y": "Y","label": "Label"})


# Call fast_mss to generate groung truth from sparse annotations
mask = fast_mss(rescaled_img, new_labels[['X','Y','Label']], unique_labels, 
                start_iter = 2500, end_iter = 20, num_iter = 30, method = 'mode')  

# convert to uint8
mask = mask.astype(np.uint8)

# Display the image, sparse annotations, and the generated ground truth mask
display_overlay(rescaled_img,colorize_prediction(mask, unique_labels),new_labels)
'''
print("Number of images in the dataset: ", len(label_dataframe))
print("number of unique labels in the entire dataset: ",len(unique_labels))
print("unique labels: ", unique_labels)


# Save mask
cv2.imwrite(str(img_name)+'_mask.png',mask)

'''


# Create an ID to label mapping - maps integer values to labels
import json
# simple example
id2label = dict_of_labels
with open('id2label.json', 'w') as fp:
    json.dump(id2label, fp)