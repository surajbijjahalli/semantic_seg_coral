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


parser = argparse.ArgumentParser("Visualize sparse labels")
parser.add_argument("--path_img_dir",type=str,help="path to image directory")
parser.add_argument("--path_annotations",type=str,help="path to sparse annotations")


args = parser.parse_args()
path_to_img_dir = args.path_img_dir
path_to_annotation_file = args.path_annotations


list_imgs = glob.glob(path_to_img_dir+"*.jpg")

rand_sample = random.choice(list_imgs)
img_name = int(Path(rand_sample).stem)






label_dataframe = pd.read_csv(path_to_annotation_file)

# grab all unique labels
unique_labels = list(label_dataframe['label'].unique())

labels = label_dataframe[label_dataframe['quadratid']==img_name]
labels_coordinates = labels[['x','y']]
path_to_img = path_to_img_dir+str(img_name)+".jpg"


img = plt.imread(path_to_img)



new_labels = labels.rename(columns={"x": "X", "y": "Y","label": "Label"})



mask = fast_mss(img, new_labels[['X','Y','Label']], unique_labels, 
                start_iter = 2500, end_iter = 20, num_iter = 20, method = 'mode')  

                


display_overlay(img,colorize_prediction(mask, unique_labels),labels)


