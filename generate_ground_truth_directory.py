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
from tqdm import tqdm
from huggingface_hub import login
'''
login(token = "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")

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("segments/sidewalk-semantic")



example_label = np.array(dataset["train"][1]["label"])
print("Unique semantic labels in this image: ", np.unique(example_label))
fig,axs = plt.subplots()
axs.imshow(example_label)
'''


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

#%%

parser = argparse.ArgumentParser("Visualize sparse labels")
parser.add_argument("--path_img_dir",type=str,help="path to image directory")
parser.add_argument("--path_annotations",type=str,help="path to sparse annotations")
parser.add_argument("--path_output",type=str,help="directory to store generated ground truth masks")
parser.add_argument("--path_resized_imgs",type=str,help="directory to store resized images")

args = parser.parse_args()
# Path to the directory of images
path_to_img_dir = args.path_img_dir

# Path to the sparse annotation csv file
path_to_annotation_file = args.path_annotations

output_dir = args.path_output

path_to_resized_imgs = args.path_resized_imgs

# Read in the entire annotation file
label_dataframe = pd.read_csv(path_to_annotation_file)

# grab all unique labels in the entire annotation file
unique_labels = list(label_dataframe['label'].unique())

# Create dictionary keys for each unique label
dict_keys = list(np.array(range(len(unique_labels))))

# Create a dictionary of labels
dict_of_labels = dict(zip(dict_keys,unique_labels))

# List of paths for each file in the image directory
list_imgs = glob.glob(path_to_img_dir+"*.jpg")
pbar = tqdm(list_imgs)

for path_index, path in enumerate(pbar):
    # Pick a random file path from the list
    #sample = random.choice(list_imgs)
    sample = path

    # extract the file name without the extension and convert to an integer - this is necessary since we use the file name as a key for grabbing the sparse annotations
    img_name = int(Path(sample).stem)







    # Grab the annotations for the sampled random image
    labels = label_dataframe[label_dataframe['quadratid']==img_name].copy()

    


    # Grab the x-y coordinates of each annotation
    labels_coordinates = labels[['x','y']]

    # Path to image - same as the randomly sampled image path
    path_to_img = sample #path_to_img_dir+str(img_name)+".jpg"

    # Read in the image
    img = plt.imread(path_to_img)

    




    rescaled_img,labels = rescale_image(img,labels,200)

    # Create a new dataframe by renaming the columns to be consistent with Fast_mss library
    new_labels = labels.rename(columns={"x": "X", "y": "Y","label": "Label"})


    # Call fast_mss to generate groung truth from sparse annotations
    mask = fast_mss(rescaled_img, new_labels[['X','Y','Label']], unique_labels, 
                start_iter = 2500, end_iter = 20, num_iter = 30, method = 'mode')  

    # convert to uint8
    mask = mask.astype(np.uint8)



    # Save mask
    cv2.imwrite(output_dir + '/' +str(img_name)+'_mask.png',mask)

    cv2.imwrite(path_to_resized_imgs + '/' +str(img_name)+'_resized.png',rescaled_img)