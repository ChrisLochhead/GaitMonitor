import numpy as np
import math
from tqdm import tqdm
from Programs.Data_Processing.Model_Based.Utilities import load, load_images, get_3D_coords
from Programs.Data_Processing.Model_Based.Demo import *   
from Programs.Data_Processing.Model_Based.Render import * 

from sklearn.preprocessing import StandardScaler
#normalization function
#[x′ i, y′ i ] = [ imgwidth 2 ∗ xi − xmin xmax − xmin , imgheight 2 ∗ yi − ymin ymax − ymin ]


def calculate_distance(x1, y1, x2, y2):
    # Calculate the difference in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the square of the differences
    dx_squared = dx ** 2
    dy_squared = dy ** 2

    # Calculate the sum of the squared differences and take the square root
    distance = math.sqrt(dx_squared + dy_squared)

    return distance

def smooth_unlikely_values(joint_data, image_data):
    for i, frame in enumerate(joint_data):
        #Ignore first frame
        if i > 0:
            #tmp = copy.deepcopy(image_data[i])
            #render_joints(tmp, joint_data[i], delay = True, use_depth=False)
            for j, coord in enumerate(frame):
                #Ignore metadata and head co-ordinates
                if j > 5:
                    if calculate_distance(coord[0], coord[1], joint_data[i - 1][j][0], joint_data[i-1][j][1]) > 100:
                        #Just reset any odd values to its previous value
                        joint_data[i][j] = joint_data[i-1][j]
            #render_joints(image_data[i], joint_data[i], delay = True, use_depth=False)
    
    return joint_data

def normalize_data(data):
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    scaler.fit(data)
    # apply transform
    standardized = scaler.transform(data)
    # inverse transform
    inverse = scaler.inverse_transform(standardized)

def normalize_joint_scales(joints, images, meta = 5):
    norm_joints = []
    #Photo dimensions in pixels
    width = 424
    height = 240
 
    for i, instance in enumerate(tqdm(joints)):
        #Add metadata
        #norm_joint_row = [instance[0], instance[1], instance[2]]
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

    return norm_joints
        
'''
Example dataset

0, 1, 0 [pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         angle_0, angle_1, angle_2] * 18
'''
  
def trim_frames(joint_data, image_data, trim):
    trimmed_joints = []
    trimmed_images = []

    for i, row in enumerate((pbar:= tqdm(joint_data))):
        pbar.set_postfix_str(i)
        #General case
        found_end = False
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

def remove_empty_frames(joint_data, image_data, meta_data = 5):
    cleaned_joints = []
    cleaned_images = []
    for i, row in enumerate(tqdm(joint_data)):
        print("Image: ", i, "of ", len(joint_data), len(image_data))
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

def get_connected_joint(joint_index):
    for j_connection in joint_connections:
        if joint_index == j_connection[0]:
            return j_connection[1]
        elif joint_index == j_connection[1]:
            return j_connection[0]

def normalize_outlier_values(joint_data, image_data, tolerance = 100, meta = 5):
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)

    for i, row in enumerate(tqdm(joint_data)):

        #Get row median to distinguish which of joint pairs are the outlier
        x_coords = [coord[0] for j, coord in enumerate(row) if j > meta]
        y_coords = [coord[1] for k, coord in enumerate(row) if k > meta]
        med_coord = [np.median(x_coords), np.median(y_coords)]

        #render_joints(image_data[i], joint_data[i], delay = True, use_depth=False)
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
                            #print("one above changes: ", math.dist(joint_0_coord, joint_1_coord))
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

        #render_joints(image_data[i], joint_data[i], delay = True, use_depth=False)               
    return joint_data
    
def normalize_outlier_depths(joints_data, image_data, plot3d = True, meta = 5):
    for i, row in enumerate(tqdm(joints_data)):

        #print("before")
        #render_joints(image_data[i], row, delay = True)
        #plot3D_joints(row)
        depth_values = []
        #Get average depth of row
        for j, joints in enumerate(row):
            if j > meta:
                depth_values.append(joints_data[i][j][2])

        quartiles = np.quantile(depth_values, [0,0.15,0.5,0.9,1])

        #Work out if quartile extent is even a problem
        q1_problem = False
        q4_problem = False

        #Do this by examining if there is a huge gap between the median and the extremities
        if quartiles[3] > quartiles[2] + 50:
            q4_problem = True
        if quartiles[1] < quartiles[2] - 50:
            q1_problem = True

        #Find all the instances with an inconsistent depth
        incorrect_depth_indices = []
        for k, d in enumerate(depth_values):
            #Check if an outlier
            #print("depth value is: ", d, "threshold is: ", quartiles[3], quartiles[1])
            if d > quartiles[3] and q4_problem == True or d <= quartiles[1] and q1_problem == True or d <= 15.0:
                #print("found incorrect depth: ", k, d, "coord: ", joints_data[i][3 + k])
                incorrect_depth_indices.append(k + 3)

        k = 0
        #Set incorrect depths to the depths of their neighbours if they fall inside the range
        #print("number of corrections to be made: ", len(incorrect_depth_indices))
        for k, indice in enumerate(incorrect_depth_indices):
            for connection_pair in joint_connections:
                #print("resetting indice: ", indice, joints_data[i][indice])
                if connection_pair[0] + meta + 1 == indice:
                    connection_joint = joints_data[i][connection_pair[1] + meta + 1][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                       # print("resetting connection joint depth", indice)
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                      #  print("resetting connection joint depth with median", indice)
                        continue
                elif connection_pair[1] + meta + 1 == indice:
                    connection_joint = joints_data[i][connection_pair[0] + meta + 1][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                       # print("resetting connection joint depth", indice)
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                     #   print("resetting connection joint depth with median", indice)
                        continue
        #print("after corrections")
        #render_joints(image_data[i], row, delay = True)
        #plot3D_joints(row)
    
    return joints_data