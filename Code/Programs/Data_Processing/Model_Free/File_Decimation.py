from __future__ import print_function

#Standard
import cv2
import re
from PIL import Image, ImageOps
import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import shutil

#Local files
import Programs.Data_Recording.JetsonYolo_Main.models.JetsonYolo as JetsonYolo
from Programs.Data_Processing.Model_Free.Utilities import numericalSort
import torch

#Torch and SKlearn
from torchvision.transforms import ToTensor, Lambda
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os.path

#Function to check if more than 1 person in any frames, and deleting if so
def check_human_count(images):
    lesser_count = 0
    greater_count = 0
    for image in images:
        objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
        if len(objs) > 1:
          greater_count += 1
        elif len(objs) < 1:
            lesser_count += 1
    print("human detected: ", greater_count, lesser_count, len(images))
    if greater_count > (len(images) * 0.5) or lesser_count > (len(images) * 0.5):
        return False
    return True

#Function to check person traverses the width of the image, else dump the files
def check_human_traversal(images):
    #Highest possbile pixel value is 240
    min_x = 1000
    max_x = -1000

    for image in images:
        objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
        #box coords takes the format [xmin, ymin, xmax, ymax]
        debug_img, box_coords = JetsonYolo.plot_obj_bounds(objs, np.asarray(image))
        #print("box co-ords: ", box_coords)
        #Co-ordinates will only be present in images with a human present
        if len(box_coords) > 0:
            if box_coords[0] < min_x:
                min_x = box_coords[0]
            if box_coords[0] > max_x:
                max_x = box_coords[0]

    #If the difference in bounding boxes is equal to 50% of the frame width
    if abs(max_x - min_x) > (images[0].shape[0] * 0.5):
        return True

    return False

#Function to send proof-read instances to a g-mail account or google drive
def decimate(path = './Images/CameraTest'):

    #Instead of deleting, flag the indices to see which ones would get decimated to ascertain if it's acceptable.
    #Load in images
    instances = []
    image_names = []
    folder_names = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if iterator == 0:
            folder_names = dirs
        images = []
        im_names = []
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                # load the input image and associated mask from disk
                image = cv2.imread(os.path.join(subdir, file))
                #print("file name: ", file)
                images.append(image)
                im_names.append(os.path.join(subdir, file))
            image_names.append(im_names)
            instances.append(images)



    for i, ims in enumerate(instances):
        #Pass images through check human count
        contains_1_human = check_human_count(ims)
        #Pass images through check human traversal
        human_traverses_fully = check_human_traversal(ims)

        #Send images to google drive
        if contains_1_human and human_traverses_fully:
          print("image appropriate, retaining")
        else:
            #delete folder
            delete_folder(folder_names[i])
            

def delete_folder(index, path = './Images/CameraTest/'):
    deletion_folder = str(path + str(index))
    #This will delete the folder and its contents recursively
    shutil.rmtree(deletion_folder, ignore_errors=True)