import cv2
import numpy as np
import csv
import os 
import csv
import copy
import pandas as pd

from Programs.Data_Processing.Model_Based.Utilities import load, load_images, get_3D_coords
from Programs.Data_Processing.Model_Based.Demo import *   
from Programs.Data_Processing.Model_Based.Render import * 

#normalization function
#[x′ i, y′ i ] = [ imgwidth 2 ∗ xi − xmin xmax − xmin , imgheight 2 ∗ yi − ymin ymax − ymin ]
def normalize_joint_scales(joints, images, save = True, save_name = "normalized_pixel_data_absolute"):
    norm_joints = []
    #Photo dimensions in pixels
    width = 424
    height = 240

 
    for i, instance in enumerate(joints):
        #Add metadata
        norm_joint_row = [instance[0], instance[1], instance[2]]

        for j, joint in enumerate(instance):
            #Ignore metadata
            if j <= 2:
                continue
            
            #2D Calculations for when avoiding depth sensor data
            all_x = [item[0] for j, item in enumerate(instance) if j > 2]
            all_y = [item[1] for j, item in enumerate(instance) if j > 2]
            min_x = min(all_x)
            max_x = max(all_x)
            min_y = min(all_y)
            max_y = max(all_y)

            #Using depth sensor
            #norm_joint = [(joint[0] * joint[2])/255,# 
            #              (joint[1] * joint[2])/255,# this works poorly with bad depth data.
            #             joint[2]]
            
            #If not using depth sensor
            norm_joint = [((width * 2) * (joint[0] - min_x)/(max_x - min_x) )/10,
                          ((height * 2) * (joint[1] - min_y)/(max_y - min_y))/10,
                          joint[2]]
            
            norm_joint_row.append(norm_joint)
        norm_joints.append(norm_joint_row)
        if i == 7:
            [print("\n n_coord : ", c, coord) for c, coord in enumerate(norm_joint_row)]

            [print("\n coord : ", c, coord) for c, coord in enumerate(instance)]
        #print("normal: ", instance)
        render_joints(images[i], norm_joint_row, delay=True, use_depth=True)
        render_joints(images[i], instance, delay=True, use_depth=True)
    new_dataframe = pd.DataFrame(norm_joints, columns = colnames)

    if save:
        #Convert to dataframe 
        print("SAVING")
        new_dataframe.to_csv("../EDA/Finished_Data/" + save_name + ".csv",index=False, header=False)     


        


def create_joint_angle_dataset():
    pass

#TODO
#convert normalized data into relative
#build angle dataset (with absolute)
#rebuild velocity with normalized absolute dataset
#Combine all 3 into one super dataset
#Construct region-based datasets (2 and 5 step)
#Test all in EDA 
#Test all in GCN

#Back to work on GAN

#Bone angle dataset (behavioural)
#Position data (spatial)
#Velocity (temporal)
#Regional (2 and 5 tier hierarchy)

'''
Example dataset

0, 1, 0 [pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         angle_0, angle_1, angle_2] * 18
'''
def create_conmbined_dataset():
    pass

def apply_joint_occlusion(file, save = False, debug = False):
    dataset = load(file)
    occluded_data = compensate_depth_occlusion(occlusion_boxes, dataset, debug=debug) 

    new_dataframe = pd.DataFrame(occluded_data, columns = colnames)
    if save:
        #Convert to dataframe 
        print("SAVING")
        new_dataframe.to_csv("./EDA/Gait_Pixel_Dataset_Occluded.csv",index=False, header=False)      

def remove_empty_frames(joint_data, image_data):
    cleaned_joints = []
    cleaned_images = []
    for i, row in enumerate(joint_data):
        print("joint: ", i, "of : ", len(joint_data))
        empty_coords = 0
        for j, coord in enumerate(row):
            if j > 2:
                if all(v == 0 for v in coord) == True:
                    empty_coords += 1
        
        #Remove all empty coord frames to hopefully catch some outliers at the cost of some frames
        if empty_coords < 1:
            cleaned_joints.append(row)
            cleaned_images.append(image_data[i])
    
    return cleaned_joints, cleaned_images

#Remember to remove this function after all done
def correct_joints_data(image_file, joint_file, save = False, pixels = True):

    #Get raw images and corresponding joint info
    joint_data = load(joint_file)
    #print("joint data: ", joint_data)
    image_data = load_images(image_file, ignore_depth=True)

    finished_joint_data = []
    
    print("first lens: ", len(image_data), len(joint_data))
    #Remove last n frames to avoid weird behaviour
    #n_removed_frames = 6

    print("running frame occlusion", len(joint_data))
    #Occlude area where not walking properly
    occlusion_box_pixels = [[0,320,240,424]]

    #Create blank depth image
    tmp_depth = np.zeros((240,424))
    for i, row in enumerate(tmp_depth):
        for j, col in enumerate(row):
            tmp_depth[i][j] = 15

    _, occlusion_box_metres = get_3D_coords([[0,320,15],[240,424,15]], tmp_depth, meta_data=0)
    print("occlusion: ", occlusion_box_metres)
    occlusion_box_metres = [[occlusion_box_metres[0][0],occlusion_box_metres[0][1],occlusion_box_metres[1][0],occlusion_box_metres[1][1]]]
    
    #Occlusion function too sensitive: also removes empty frames
    #if pixels:
    #    image_data, joint_data = occlude_area_in_frame(occlusion_box_pixels, joint_data, image_data)
    #else:
    #    image_data, joint_data = occlude_area_in_frame(occlusion_box_metres, joint_data, image_data)

    #Fix any incorrect depth data
    print("detecting outlier values ", len(joint_data))
    joint_data = normalize_outlier_values(joint_data, image_data)

    print("Removing empty frames", len(joint_data))
    joint_data, image_data = remove_empty_frames(joint_data, image_data)
    print("removed lens: ", len(joint_data), len(image_data))

    for i, row in enumerate(joint_data):
        finished_joint_data.append(copy.deepcopy(row))

        #Save images as you go
        if save:
            print("saving instance: ", str(float(row[0])))
            if row[1] < 10:
                file_no = str(0) + str(row[1])
            else:
                file_no = str(row[1])

            directory = "./EDA/Finished_Data/Images/Instance_" + str(float(row[0]))
            print("i is: ", i , len(image_data), len(joint_data))
            os.makedirs(directory, exist_ok = True)
            cv2.imwrite(directory + "/" + file_no + ".jpg", image_data[i])

    #Fix head coordinates on what's left
    normalize_head_coords(joint_data, image_data)

    print("finished length of joints: ", len(finished_joint_data))
    #Finally finish joints
    if save:
        with open("./EDA/Finished_Data/pixel_data_absolute.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(finished_joint_data)

def check_interception(occlusion_boxes, joint):
    #X-axis collision
    for occlusion_box in occlusion_boxes:
        #print("joint: ", joint[0], "box: ", occlusion_box[0])
        if joint[0] > occlusion_box[0] and joint[0] < occlusion_box[2]:
            #Y-axis collision
            if joint[1] > occlusion_box[1] and joint[1] < occlusion_box[3]:
                return True
    
    return False

def get_connected_joint(joint_index):
    for j_connection in joint_connections:
        if joint_index == j_connection[0]:
            #print("found connection: 0", j_connection,  joint_index )
            return j_connection[1]
        elif joint_index == j_connection[1]:
            #print("found connection: 1", j_connection,  joint_index )
            return j_connection[0]

#Get background to find box co-ordinates of couch for occlusion
def compensate_depth_occlusion(occlusion_boxes, joints, folder ='./Images', debug = False):
        
    #Load images: 
    if debug:
        image_data = load_images(folder)

    for index, row in enumerate(joints):
        intercept_check_count = 0
        #print("instance: ", index)
        if debug:
            #print("before correcting joints")
            render_joints(image_data[index], row, delay = True, use_depth = True)      
        for joint_index, joint in enumerate(row):
            #Ignore metadata
            if joint_index >= 3:
                #Check interception between this x coord and the y coord next in line
                if check_interception(occlusion_boxes, joint) == True:
                    if debug:
                        intercept_check_count += 1
                        #print("this joint is connected to: ", row[get_connected_joint(joint_index-3)], "with index: ", get_connected_joint(joint_index-3), "connects to: ", joint_index-3)
                    
                    #Reassign this depth value to nearest connected joint
                    joints[index][joint_index][2] = copy.deepcopy(row[get_connected_joint(joint_index-3)][2]) 
        #New after all occluded joints corrected
        if debug:
            #print("after correcting joints: ", intercept_check_count)
            render_joints(image_data[index], row, delay = True, use_depth = True)
        
    return joints


def occlude_area_in_frame(occlusion_box, joint_data, image_data):

    refined_joint_data = []
    refined_image_data = []

    occlusion_begun = False
    has_been_in_frame = False

    for i, row in enumerate(joint_data):
        print("joint: ", i, "of : ", len(joint_data))
        occlusion_counter = 0
        #Reset for the start of a new instance
        if i > 0:
            if row[1] < joint_data[i - 1][1]:
                #render_joints(image_data[i], row, delay=True)
                occlusion_begun = False
                has_been_in_frame = False

        empty_coords = 0
        for j, coord in enumerate(row):
            if j > 2:
                if all(v == 0 for v in coord) == True:
                    empty_coords += 1

        for j, coord in enumerate(row):
            if j >= 3:
                if check_interception(occlusion_box, coord) == True and empty_coords < 15:
                    occlusion_counter += 1
        
        #If more than 5 joints are in the occlusion zone, get rid of these joints and corresponding images
        if occlusion_begun == False:
            if occlusion_counter < 3 and empty_coords < 15:
                refined_joint_data.append(row)
                refined_image_data.append(image_data[i])
                has_been_in_frame = True
                #print("num joints occluded = ", occlusion_counter)
                #print("keeping frame:")
                #render_joints(image_data[i], row, delay=True)
            else:
                if has_been_in_frame == True:
                    occlusion_begun = True
                #print("removing frame: ", occlusion_begun)
                #render_joints(image_data[i], row, delay=True)

    return refined_image_data, refined_joint_data


def normalize_outlier_values(joints_data, image_data, plot3d = True):
    for i, row in enumerate(joints_data):

        print("before")
        #render_joints(image_data[i], row, delay = True)
        #plot3D_joints(row)
        depth_values = []
        #Get average depth of row
        for j, joints in enumerate(row):
            if j > 2:
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
                if connection_pair[0] + 3 == indice:
                    connection_joint = joints_data[i][connection_pair[1] + 3][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                       # print("resetting connection joint depth", indice)
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                      #  print("resetting connection joint depth with median", indice)
                        continue
                elif connection_pair[1] + 3 == indice:
                    connection_joint = joints_data[i][connection_pair[0] + 3][2]
                    if connection_joint < quartiles[3] and connection_joint > quartiles[1]:
                        joints_data[i][indice][2] = connection_joint
                       # print("resetting connection joint depth", indice)
                        continue
                    else:
                        joints_data[i][indice][2] = quartiles[2]
                     #   print("resetting connection joint depth with median", indice)
                        continue
        print("after corrections")
        #render_joints(image_data[i], row, delay = True)
        #plot3D_joints(row)
    
    return joints_data
#3: nose
#4: left eye
#5: right eye
#6: left ear
#7: right ear
def normalize_head_coords(joints_data, image_data):

    for i, row in enumerate(joints_data):

        #print("before")
        #render_joints(image_data[i], row, delay = True)

        face_joints = []
        null_counter = 0
        face_joints_depth_mean = 0
        null_positions = [0,0,0,0,0]
        for j, coord in enumerate(row):
            if j > 2 and j < 8:
                face_joints.append(coord)
                for v in face_joints:
                    #print("coords here: ", v)
                    if v[0] <= 5 and v[1] <= 5:
                        #print("finding null here")
                        null_counter += 1
                        null_positions[j - 3] = 1
                        face_joints_depth_mean += v[2]
                face_joints_depth_mean = face_joints_depth_mean / 5                  

        #If more than one is out of place but it's not every one of them, correct the broken ones
        if null_counter > 0 and null_counter < 5:
            normalize_scalar = 4# len(face_joints) - null_counter
            mean = [0,0,0]
            for f in face_joints:
                #print("adding mean: ", mean, f)
                mean = add_lists(mean, f)

            #print("dividing result", mean, normalize_scalar)
            mean = divide_list(mean, normalize_scalar) 
            #print("result", mean) 

            #print("getting here?")
            #Re-iterate through correcting the broken face joints
            for k, coord in enumerate(row):
                if k > 2 and k < 8:
                    if null_positions[k - 3] == 1:
                        #print("applying new joints data value", joints_data[i][k])
                        joints_data[i][k] = mean
                        #print("value now", joints_data[i][k])

        
        #print("after", null_counter)
        #render_joints(image_data[i], joints_data[i], delay = True)

