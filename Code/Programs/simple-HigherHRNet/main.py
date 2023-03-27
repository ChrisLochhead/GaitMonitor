import cv2
import numpy as np
from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import check_video_rotation, draw_points_and_skeleton
import csv
import os 
import csv
import copy
import pandas as pd
from array import array
import ast
import operator
import math
from ast import literal_eval
import pyrealsense2 as rs

from DatasetBuilder.utilities import *
from DatasetBuilder.demo import *      

#Hand crafted features
'''        cadence (number of steps),

        stride length,

        height of feet above the ground,

        time left leg off ground,

        time right leg off ground,

        speed, stride length variance,

        time with both feet on the floor (gait freezing),

        difference in left and right leg stride length(detect limping),
'''

def make_intrinsics(intrinsics_file = "depth_intrinsics.csv"):
        '''
        Avoid having to read a bagfile to get the camera intrinsics
        '''

        with open(intrinsics_file, newline='') as csvfile:
            data_struct = list(csv.reader(csvfile))

        data = data_struct[0]
        for i, d in enumerate(data):
            #avoid converting what is already in the correct format (distortion type)
            if i != 6:
                data[i] = literal_eval(data[i])


        # Copied from a bagfile's intrinsics
        intrinsics = rs.intrinsics()
        intrinsics.coeffs = data[7]
        intrinsics.fx = data[4]
        intrinsics.fy = data[5]
        intrinsics.height = data[1]
        intrinsics.ppx = data[2]
        intrinsics.ppy = data[3]
        intrinsics.width=data[0]
        return intrinsics

loaded_intrinsics = make_intrinsics()

def get_3D_coords(coords_2d, dep_img, pts3d_net = True, dilate = True, meta_data = 3):

    pts_3D = []
    pts_3D_metres = []
    orig_dep_img = copy.deepcopy(dep_img)
    kernel = np.ones((6,6), np.uint8)
    if dilate == True:
        dep_img = cv2.dilate(dep_img, kernel, cv2.BORDER_REFLECT)

    #orig_dep_img = np.transpose(orig_dep_img)
    for i in range(meta_data, len(coords_2d)):
        
        x = copy.deepcopy(int(coords_2d[i][0])); y = copy.deepcopy(int(coords_2d[i][1]))
        
        if pts3d_net == True:
            
            #Clamp joints within frame size
            if x >= 240:
                x= 239
            if y >= 424:
                y= 423

            result = rs.rs2_deproject_pixel_to_point(loaded_intrinsics, [x, y], orig_dep_img[(x, y)])
            pts_3D_metres.append([-result[0], -result[1], result[2]])
            pts_3D.append([x, y, result[2]])
        else:
            pts_3D.append([x,y, dep_img[(y,x)]])
            pts_3D_metres.append([x,y, dep_img[(y,x)]])

    return pts_3D, pts_3D_metres

def run_image(image_name, single = True, save = False, directory = None, model= None, image_no = 0):
    if model == None:
        model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)

    #apply joints to image for visualisation
    image, joints = get_joints_from_frame(model, image, anonymous=True)


    loop = True
    while loop == True:
        if single == True:
            cv2.imshow('Example', image)
            cv2.waitKey(0) & 0xff

        loop = False
    
    if save and directory != None:
        cv2.imwrite(directory + "/" + str(image_no) + ".jpg", image)

    return image, joints

def run_depth_sample(folder_name, joints_info):
    #get joints info: 
    joint_dataframe = pd.read_csv(joints_info, names=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    #Drop metadata
    joint_dataframe = joint_dataframe.iloc[:,3:]

    #Transform into array 
    depth_array = joint_dataframe.to_numpy()
    #Convert to ints from strings
    for i, row in enumerate(depth_array):
        for j, value in enumerate(row):
            depth_array[i, j] = literal_eval(depth_array[i, j])

    #Run in realtime for debugging joint co-ordinates
    model = SimpleHigherHRNet(48, 17, "./weights/pose_higher_hrnet_w48_640.pth")

    directory = os.fsencode(folder_name)
    for subdir, dirs, files in os.walk(directory):
        dirs.sort(key=numericalSort)
        for i, file in enumerate(files):
            #Ignore depth images which are second half
            if i >= len(files)/2:
                break

            file_name = os.fsdecode(file)
            sub_dir = os.fsdecode(subdir)
            
            #display with openCV original image, overlayed with corresponding joints
            raw_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_COLOR)
            depth_image = cv2.imread(sub_dir + "/" + os.fsdecode(files[int(i + len(files)/2)]), cv2.IMREAD_ANYDEPTH)

            image, joints = run_image(sub_dir + "/" + file_name, single=False, save = False, directory=None, model=model, image_no = i)

            joint_set = joints[0]
            
            initial_joint_image = draw_joints_on_frame(raw_image, joint_set)
            cv2.imshow('image with raw joints',initial_joint_image)
            #cv2.imshow('depth image',depth_image)
            cv2.setMouseCallback('image with raw joints', click_event, initial_joint_image)
            cv2.waitKey(0) & 0xff

            #Apply depth changes to data, firstly with 2d images using z as colour
            #Need corresponding depth image not raw
            refined_joint_set, rjs_metres = get_3D_coords(joint_set, depth_image)
            refined_joint_image = draw_joints_on_frame(raw_image, refined_joint_set, use_depth_as_colour=True)
            cv2.imshow('image with refined joints (excl 2D)',refined_joint_image)
            cv2.setMouseCallback('image with refined joints (excl 2D)', click_event, refined_joint_image)
            cv2.waitKey(0) & 0xff


            #display again with 3d positions including altered X and Y
            final_joint_set, fjs_metres = get_3D_coords(joint_set, depth_image)
            final_joint_image = draw_joints_on_frame(raw_image, fjs_metres, use_depth_as_colour=True)
            cv2.imshow('image with refined joints (incl 2D)',final_joint_image)
            cv2.setMouseCallback('image with refined joints (incl 2D)', click_event, final_joint_image)
            cv2.waitKey(0) & 0xff


#Exclude 2D depth coord calculations to just get pixel data and depth value, otherwise it will return actual distances good for ML but
#not possible to properly display for debugging without re-projecting back to pixel data.
def run_images(folder_name, exclude_2D = False, write_mode = "w+", start_point = 0):
    directory = os.fsencode(folder_name)
    model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    file_iter = 0
    subdir_iter = -1
    data_class = 0
    #Format for the joints file
    #Instance Number: Sequence number: Class : Joint positions 1 - 17
    joints_file = []
    joints_file_metres = []

    for directory_iter, (subdir, dirs, files) in enumerate(os.walk(directory)):
        dirs.sort(key=numericalSort)

        #Skip any instances already recorded
        if directory_iter < start_point:
            subdir_iter += 1
            continue

        if subdir_iter % 5 == 0:
            data_class += 1
            if data_class > 2:
                data_class = 0

        first_depth = True
        count_in_directory = 0

        for f, file in enumerate(files):

            file_name = os.fsdecode(file)
            if file_name[0] == 'd':
                continue
            sub_dir = os.fsdecode(subdir)
            print("Sub directory: ", sub_dir, " Instance: ", file_iter - count_in_directory)
            print("subdir iter: ", subdir_iter)

            out_directory = "./example_imgs/"
            
            os.makedirs(out_directory, exist_ok=True)

             #Get joints from initial image
            image, joints = run_image(sub_dir + "/" + file_name, single=False, save = True, directory=out_directory + sub_dir, model=model, image_no = file_iter)

            # Get corresponding depth image, this is always same index += half the length of the instance
            dep_image = cv2.imread(sub_dir + "/" + os.fsdecode(files[int(f + (len(files)/2))]), cv2.IMREAD_ANYDEPTH)
            corr_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_ANYCOLOR )

            if len(joints) > 0:
                if len(joints[0]) < 1:
                    joints = [ [0,0,0] for _ in range(17) ]
            
            if len(joints) > 0:
                refined_joints, refined_joints_metres = get_3D_coords(joints[0], dep_image, meta_data=0)
            else:
                refined_joints = [ [0,0,0] for _ in range(17) ]
                refined_joints_metres = [ [0,0,0] for _ in range(17) ]

            new_entry = [subdir_iter, file_iter, data_class]
            new_metres_entry = [subdir_iter, file_iter, data_class]
            # 0 is instance, 1 is num in sequence, 2 is class, 3 is array of joints
            for i, joint in enumerate(refined_joints):
                #Convert so it saves with comma delimiters within the joint-sets
                tmp = [joint[0], joint[1], joint[2]]
                tmp_metres = [refined_joints_metres[i][0], refined_joints_metres[i][1],refined_joints_metres[i][2] ]
                new_entry.append(tmp)
                new_metres_entry.append(tmp_metres)


            print("full completed depth joints: ", len(new_entry))
            #render_joints(corr_image, new_entry[3:], delay=True, use_depth=True, metadata = 0)
            
            joints_file.append(new_entry)
            joints_file_metres.append(new_metres_entry)

            file_iter += 1
        subdir_iter +=1

        print("new subdirectory??")

        #Save after every folder
        if os.path.exists("./EDA/gait_dataset_pixels.csv"):
            write_mode = "a"
        else:
            write_mode = "w+"

        #Save to .csv
        print("SAVING")

        with open("./EDA/gait_dataset_pixels.csv",write_mode, newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file)
            joints_file = []
        with open("./EDA/gait_dataset_metres.csv",write_mode, newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file_metres)
            joints_file_metres = []

        #Debug
        #if subdir_iter >= 4:
        #    print("BREAKING")
        #    break
        file_iter = 0



def get_unit_vector(vector):

    magnitude = np.sqrt((vector[0]**2) + (vector[1]**2) + (vector[2]**2))
    if magnitude == 0:
        magnitude = 1

    unit_vector = [vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude]
    return unit_vector

def add_lists(list1, list2, change_type = False):
    return list((np.array(list1) + np.array(list2)).astype(int))

def list_to_arrays(my_list):
    tmp_list = copy.deepcopy(my_list)
    for i, l in enumerate(tmp_list):
        if isinstance(l, list):
            tmp_list[i] = np.array(l)

    return tmp_list

def subtract_lists(list1, list2):
    result = np.subtract(list1, list2)
  #  result = map(int, result)
    return list(result)#

def midpoint(list1, list2):
    result = (np.array(list1) + np.array(list2)).astype(int)
    result = result / 2
    return list(result)

def divide_lists(list1, list2, n):
    return list((np.array(list1) + np.array(list2))/n)

def divide_list(list1, n):
    return list((np.array(list1))/n)

def plot_velocity_vectors(image, joints_previous, joints_current, joints_next, debug = False):
    
    #Add meta data to the start of the row
    joint_velocities = [joints_current[0], joints_current[1], joints_current[2]]

    #Convert to numpy arrays instead of lists
    joints_previous = list_to_arrays(joints_previous)
    joints_current = list_to_arrays(joints_current)
    joints_next = list_to_arrays(joints_next)

    #-3 to account for metadata at front
    for i in range(3, len(joints_current)):
        if len(joints_next) > 1:
            direction_vector_after = subtract_lists(joints_next[i], joints_current[i])
            #Unit vector
            unit_vector_after = get_unit_vector(direction_vector_after)
            end_point_after = [joints_current[i][0] + (unit_vector_after[0]), joints_current[i][1] + (unit_vector_after[1]), joints_current[i][2] + (unit_vector_after[2])]
            direction_after = subtract_lists(end_point_after, [joints_current[i][0], joints_current[i][1], joints_current[i][2]])

        else:
            direction_after = [0,0,0]

        if len(joints_previous) > 1:
            direction_vector_before = subtract_lists(joints_previous[i], joints_current[i])
            #Unit vector
            unit_vector_before = get_unit_vector(direction_vector_before)
            end_point_before = [joints_current[i][0] + (unit_vector_before[0]), joints_current[i][1] + (unit_vector_before[1]), joints_current[i][2] + (unit_vector_before[2])]
            direction_before = subtract_lists([joints_current[i][0], joints_current[i][1], joints_current[i][2]], end_point_before)
        else:
            direction_before = [0,0,0]



        if len(joints_previous) > 0 and len(joints_next) > 0:
            smoothed_direction = divide_lists(direction_after, direction_before, 2) 
        else:
            smoothed_direction = direction_after + direction_before

            
        #Remove noise:
        for j in range(0, 3):
            if smoothed_direction[j] < 0.01 and smoothed_direction[j] > -0.01:
                smoothed_direction[j] = np.float64(0.0)
        print("smoothed direction: ", smoothed_direction, type(smoothed_direction[0]))

        if debug:
            x = int((smoothed_direction[1] * 40) + joints_current[i][1])
            y = int((smoothed_direction[0] * 40) + joints_current[i][0])
            image_direction = [int((smoothed_direction[1] * 40) + joints_current[i][1]),  int((smoothed_direction[0] * 40) + joints_current[i][0])]

            image = cv2.arrowedLine(image, [int(joints_current[i][1]), int(joints_current[i][0])] , image_direction,
                                            (0,255,0), 4) 

        joint_velocities.append(smoothed_direction)

    if debug:
        # Displaying the image 
        cv2.imshow("velocity vector", image) 
        cv2.setMouseCallback("velocity vector", click_event, image)
        cv2.waitKey(0) & 0xff

    return joint_velocities

def create_dataset_with_chestpoint(jointfile, folder, save = True, debug = False):
    joint_data = load(jointfile)
    image_data = load_images(folder)

    chest_dataset = []

    for i, joints in enumerate(joint_data):
        chest_dataset_row = list(joints)
        chest_datapoint = midpoint(joints[9], joints[8])
        chest_dataset_row.append(chest_datapoint)
        chest_dataset.append(chest_dataset_row)

        if debug:
            render_joints(image_data[i], chest_dataset_row, delay=True, use_depth=True)
    
    chest_colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17', 'Joint_18'] 

    new_dataframe = pd.DataFrame(chest_dataset, columns = chest_colnames)
    if save:
            #Convert to dataframe 
        new_dataframe.to_csv("./EDA/Chest_Dataset.csv",index=False, header=False)

def run_velocity_debugger(folder, jointfile, save = False, debug = False):
        
        joint_data = load(jointfile)
        image_data = load_images(folder, ignore_depth=False)
        velocity_dataset = []
        for i, joints in enumerate(joint_data):
            if i+1 < len(joint_data) and i > 0:
                velocity_dataset.append(plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, joint_data[i + 1], debug=debug))
            elif i+1 < len(joint_data) and i <= 0:
                velocity_dataset.append(plot_velocity_vectors(image_data[i], [0], joints, joint_data[i + 1], debug=debug))
            elif i+1 >= len(joint_data) and i > 0:
                velocity_dataset.append(plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, [0], debug=debug))
        
        new_dataframe = pd.DataFrame(velocity_dataset, columns = colnames)
        if save:
                #Convert to dataframe 
            new_dataframe.to_csv("./EDA/Finished_Data/pixel_velocity_relative.csv",index=False, header=False)
    
############################################################ Draw velocity vectors here ##################
## couch coords topleft.x, topleft.y, 2 boxes


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

def apply_joint_occlusion(file, save = False, debug = False):
    dataset = load(file)
    occluded_data = compensate_depth_occlusion(occlusion_boxes, dataset, debug=debug) 

    new_dataframe = pd.DataFrame(occluded_data, columns = colnames)
    if save:
        #Convert to dataframe 
        print("SAVING")
        new_dataframe.to_csv("./EDA/Gait_Pixel_Dataset_Occluded.csv",index=False, header=False)         

def correct_joints_data(image_file, joint_file, save = False, pixels = True):

    #Get raw images and corresponding joint info
    joint_data = load(joint_file)
   #print("joint data: ", joint_data)
    image_data = load_images(image_file, ignore_depth=True)

    finished_joint_data = []
    
    #Remove last n frames to avoid weird behaviour
    n_removed_frames = 6

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
    
    if pixels:
        image_data, joint_data = occlude_area_in_frame(occlusion_box_pixels, joint_data, image_data)
    else:
        image_data, joint_data = occlude_area_in_frame(occlusion_box_metres, joint_data, image_data)

    #Fix any incorrect depth data
    print("detecting outlier values ", len(joint_data))
    joint_data = normalize_outlier_values(joint_data, image_data)

    print("Removing empty frames", len(joint_data))

    for i, row in enumerate(joint_data):
        finished_joint_data.append(copy.deepcopy(row))

        #Save images as you go
        if save:
            print("saving instance: ", str(float(row[0])))
            if row[1] < 10:
                file_no = str(0) + str(row[1])
            else:
                file_no = str(row[1])

            #directory = "./EDA/Finished_Data/Images/Instance_" + str(float(row[0]))
            #os.makedirs(directory, exist_ok = True)
            #cv2.imwrite(directory + "/" + file_no + ".jpg", image_data[i])

    #Fix head coordinates on what's left
    normalize_head_coords(joint_data, image_data)

    print("finished length of joints: ", len(finished_joint_data))
    #Finally finish joints
    if save:
        with open("./EDA/Finished_Data/metres_data_absolute.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(finished_joint_data)


def occlude_area_in_frame(occlusion_box, joint_data, image_data):

    refined_joint_data = []
    refined_image_data = []

    occlusion_begun = False
    has_been_in_frame = False

    for i, row in enumerate(joint_data):
        print("joint: ", i, "of : ", len(joint_data))
        occlusion_counter = 0
        #Ignore metadata

        #Reset for the start of a new instance
        if i > 0:
            if row[1] < joint_data[i - 1][1]:
                #print("RESETTING TO FALSE")
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


def main():

    #Process data from EDA into perfect form
    #correct_joints_data("./Images", "./EDA/gait_dataset_metres.csv", save=True, pixels=False)
    #Test:
    #print("Correction processing sucessfully completed, testing resulting images and joints...")
    #load_and_overlay_joints(directory="./EDA/Finished_Data/Images/", joint_file="./EDA/Finished_Data/pixel_data_absolute.csv", ignore_depth=False, plot_3D=True)

    #Experimental creating hand crafted features
    #get_gait_cycles(load("./EDA/unravelled_relative_data.csv"))
    create_hcf_dataset("./EDA/Finished_Data/pixel_data_absolute.csv", "./EDA/gait_dataset_pixels.csv", "./Images")
    #Create dataset with chest joints
    #create_dataset_with_chestpoint("./EDA/gait_dataset_pixels.csv", "./Images")
    
    #Visualize joints overlaying (2D and 3D)
    #load_and_overlay_joints()

    #Demonstrate occlusion fixing
    #apply_joint_occlusion("./EDA/gait_dataset_pixels.csv", save = True, debug=True)

    #Draw calculated velocities
    #run_velocity_debugger("./EDA/Finished_Data/Images/", "./EDA/Finished_Data/pixel_data_relative.csv", save= True, debug=False)

    #run_images("./Images", exclude_2D=False, start_point=0)

    #run_depth_sample("./DepthExamples", "depth_examples.csv")
    #run_video()
if __name__ == '__main__':
    #Main menu
    main()