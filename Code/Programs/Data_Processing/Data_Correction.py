'''
This file contains all the additional functions for correcting the raw data during the pre-processing phase
'''
#imports
import numpy as np
import math
from tqdm import tqdm
#dependencies
from Programs.Data_Processing.Image_Processor import *   
from Programs.Data_Processing.Render import * 

def calculate_distance(x1, y1, x2, y2):
    '''
    Calculates the 2D distance between two co-ordinates

    Arguments
    ---------
        x1, y1, x2, y2: float
            The 2D portions of 2 co-ordinate sets 1 and 2

    Returns
    -------
        List(float, float)
            The 2D distance between points 1 and 2.
    '''
    # Calculate the square of the differences
    dx = x2 - x1
    dy = y2 - y1
    dx_squared = dx ** 2
    dy_squared = dy ** 2
    # Calculate the sum of the squared differences and take the square root
    distance = math.sqrt(dx_squared + dy_squared)
    return distance

def smooth_unlikely_values(joint_data, image_data = None, render = False, threshold = 100):
    '''
    This function finds any values that between times t and t+1, has moved an unusual amount (default set to 100 pixels)
    and reset it to the prior value

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        render: (optional, default = False) bool
            Indicates whether to show debug images
        threshold: (optional, default = 100) int
            Threshold for joint difference before being reset.

    Returns
    -------
        List(List())
            The original dataset with any corrections made
    '''
    consta = copy.deepcopy(joint_data)
    for i, frame in enumerate(joint_data):
        #Ignore first frame as there's no t-1
        if i > 0:
            if render:
                tmp = copy.deepcopy(image_data[i])
                render_joints(tmp, joint_data[i], delay = True, use_depth=False)
            for j, coord in enumerate(frame):
                #Ignore metadata and head co-ordinates
                if j > 5:
                    if calculate_distance(coord[0], coord[1], joint_data[i - 1][j][0], joint_data[i-1][j][1]) > threshold:
                        #Just reset any odd values to its t-1 value
                        joint_data[i][j] = consta[i-1][j]
            if render:
                render_joints(image_data[i], joint_data[i], delay = True, use_depth=False)
    return joint_data

def normalize_joint_scales(joint_data, image_data = None, meta = 5, width = 424, height = 240):
    '''
    This function is a bespoke scaler to compress all skeleton joints into an area scaled to each skeletons maximum size and width

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        width: (optional, default = 424) int
            Width of source images
        height: (optional, default 240) intr
            Height of source images
    Returns
    -------
        List(List())
            The original dataset with normalizations made
    '''
    norm_joints = []
    for i, instance in enumerate(tqdm(joint_data)):
        #Add metadata
        render_joints(image_data[i], joint_data[i], True)
        norm_joint_row = instance[0:meta + 1]
        for j, joint in enumerate(instance):
            #Ignore metadata
            if j <= meta:
                continue
            
            #2D Calculations for when avoiding depth sensor data
            all_x = [item[0] for j, item in enumerate(instance) if j > meta]
            all_y = [item[1] for j, item in enumerate(instance) if j > meta]
            min_x = min(all_x)
            max_x = max(all_x)
            min_y = min(all_y)
            max_y = max(all_y)
            
            #If not using depth sensor
            norm_joint = [round(((width * 2) * (joint[0] - min_x)/(max_x - min_x) )/10, 2),
                          round(((height * 2) * (joint[1] - min_y)/(max_y - min_y))/10, 2),
                          round(joint[2], 2)]
            norm_joint_row.append(norm_joint)
        norm_joints.append(norm_joint_row)
        render_joints(image_data[i], norm_joints[i], True)
    return norm_joints
    
def trim_frames(joint_data, image_data = None, trim = 5):
    '''
    A simple utility function to trim any excess frames to avoid the noise caused by individuals walking onto/out of the FOV of the camera.

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        trim: (optional, default = 5) int
            The number of frames from the stand and end of each sequence to cut
    Returns
    -------
        List(List()), List(List())
            The modified joints and images without the cut frames
    '''
    trimmed_joints = []
    trimmed_images = []
    for i, row in enumerate((pbar:= tqdm(joint_data))):
        pbar.set_postfix_str(i)
        found_end = False
        #find the last frame in each sequence denoted by their sequence metadata attribute
        if i < len(joint_data) - trim:
            for j in range(trim):
                if joint_data[i][1] > joint_data[i+j][1]:
                    found_end = True
                elif joint_data[i][1] < joint_data[i-j][1]:
                    found_end = True            
        else:
            found_end = True
        #Only append those not within the trim range
        if found_end == False:
            trimmed_joints.append(row)
            trimmed_images.append(image_data[i])
    return trimmed_joints, trimmed_images

def remove_empty_frames(joint_data, image_data = None, meta_data = 5):
    '''
    Utility function to find any frames with no people in it and cut them.

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        meta_data: (optional, default = 5)
            the amount of meta data in the rows, this may change in future
    Returns
    -------
        List(List()), List(List())
            The modified joints and images without the cut frames
    '''
    cleaned_joints = []
    cleaned_images = []
    for i, row in enumerate(tqdm(joint_data)):
        empty_coords = 0
        for j, coord in enumerate(row):
            if j > meta_data:
                if all(v == 0 for v in coord) == True:
                    empty_coords += 1
        #Remove all empty coord frames to hopefully catch some outliers at the cost of some frames
        if empty_coords < 1:
            cleaned_joints.append(row)
            cleaned_images.append(image_data[i])
    return cleaned_joints, cleaned_images

def normalize_outlier_values(joint_data, image_data, tolerance = 100, meta = 5):
    '''
    This function finds outliers outside of the standard range of the graphs and resets them to the median

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        tolerance: (optional, default = 100)
            The number of pixels that a joint can be an outlier before it is considered unacceptable
        meta_data: (optional, default = 5)
            the amount of meta data in the rows, this may change in future
    Returns
    -------
        List(List())
            The modified joints with the outliers normalized
    '''
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    for i, row in enumerate(tqdm(joint_data)):
        #Get row median to distinguish which of joint pairs are the outlier
        x_coords = [coord[0] for j, coord in enumerate(row) if j > meta]
        y_coords = [coord[1] for k, coord in enumerate(row) if k > meta]
        med_coord = [np.median(x_coords), np.median(y_coords)]
        for l, coord in enumerate(row):
            #Ignore metadata
            if l > meta:
                for j_index in joint_connections:
                    outlier_reassigned = False
                    #Found connection
                    joint_0_coord = [row[j_index[0] + meta + 1][0], row[j_index[0] + meta + 1][1]]
                    joint_1_coord = [row[j_index[1] + meta + 1][0], row[j_index[1] + meta + 1][1]]
                    if l - meta - 1 == j_index[0] or l - meta - 1 == j_index[1]:
                        if math.dist(joint_0_coord, joint_1_coord) > tolerance:
                            #Work out which of the two is the outlier
                            if math.dist(med_coord, joint_0_coord) > math.dist(med_coord, joint_1_coord):
                                #Just set outlier to it's neighbour to reduce damage done by outlier without getting rid of frame
                                #I could replace this in future with the ground truth relative distance for a better approximation
                                joint_data[i][j_index[0] + meta + 1] = [joint_1_coord[0], joint_1_coord[1], row[j_index[1] + meta + 1][2]]
                                outlier_reassigned = True
                            else:
                                joint_data[i][j_index[1] + meta + 1] = [joint_0_coord[0], joint_0_coord[1], row[j_index[0] + meta + 1][2]]
                                outlier_reassigned = True
                    #Stop looping after first re-assignment: some joints have multiple connections.
                    if outlier_reassigned:
                        break            
    return joint_data
    
def normalize_outlier_depths(joints_data, image_data = None, meta = 5):
    '''
    Utility function to normalize the z-co-ordinates to deal with cases of occlusion and other noise

    Arguments
    ---------
        joint_data: List(List())
            Passes the whole dataset where each row represents a frame
        image_data: (optional, default = None) List(List())
            Image data corresponding to the joints, optional for debugging
        meta_data: (optional, default = 5)
            the amount of meta data in the rows, this may change in future
    Returns
    -------
        List(List())
            The modified joints with outliers reset to the median
    '''
    for i, row in enumerate(tqdm(joints_data)):
        depth_values = []
        #Get average depth of row
        for j, joints in enumerate(row):
            if j > meta:
                depth_values.append(joints_data[i][j][2])

        #Work out if quartile extent is even a problem
        #Do this by examining if there is a huge gap between the median and the extremities
        quartiles = np.quantile(depth_values, [0,0.15,0.5,0.9,1])
        q1_problem = False
        q4_problem = False
        if quartiles[3] > quartiles[2] + 50:
            q4_problem = True
        if quartiles[1] < quartiles[2] - 50:
            q1_problem = True

        #Find all the instances with an inconsistent depth
        incorrect_depth_indices = []
        for k, d in enumerate(depth_values):
            #Check if an outlier
            if d > quartiles[3] and q4_problem == True or d <= quartiles[1] and q1_problem == True or d <= 15.0:
                incorrect_depth_indices.append(k + 3)
        k = 0
        #Set incorrect depths to the depths of their neighbours if they fall inside the range
        for k, indice in enumerate(incorrect_depth_indices):
            for connection_pair in joint_connections:
                if connection_pair[0] + meta + 1 == indice:
                    connection_joint = joints_data[i][connection_pair[1] + meta + 1][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                        continue
                elif connection_pair[1] + meta + 1 == indice:
                    connection_joint = joints_data[i][connection_pair[0] + meta + 1][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                        continue   
    return joints_data


    