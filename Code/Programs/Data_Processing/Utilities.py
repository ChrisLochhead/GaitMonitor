'''
This file contains all utility and auxillary functions for the files contained in the Data Processing directory. Other utilities exist in the
Machine learning directory, explicitly for utilities for the GCNs.
'''
#imports
import cv2
import csv
import os 
import copy
import ast
import re
import math
import pandas as pd
import numpy as np
from ast import literal_eval
import pyrealsense2 as rs
from tqdm import tqdm
#dependencies
from Programs.Data_Processing.Render import render_joints, draw_joints_on_frame

colnames=['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 

colnames_midhip = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', "M_hip"] 

colnames_default = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot'] 

colnames_nohead = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose', 'L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', "M_hip"] 

hcf_colnames = ["Instance", "No_In_Sequence", "Class", 'Freeze', 'Obstacle', 'Person', "Feet_Height_0", "Feet_Height_1",
                 "Time_LOG_0", "Time_LOG_1", "Time_No_Movement", "Speed", "Stride_Gap", "Stride_Length", "Max_Gap", 'l_co 1',
                 'l_co 2', 'l_co 3', 'l_co 4', 'l_co 5', 'l_co 6', 'l_co 7', 'r_co 1', 'r_co 2', 'r_co 3', 'r_co 4', 'r_co 5', 'r_co 6', 'r_co 7']

fused_colnames = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Head','L_arm','R_arm','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', "M_hip"] 

                  #Head goes to hip, all other head joints go to nothing because they will be dropped
bone_connections =[[0, 17], [1, 0], [2, 1], [3, 0], [4, 3],
                    #Right arm (extremities like hands and feet go to 0)
                    [5, 7], [7, 9], [9, -1],
                    #Left arm
                    [6, 8], [8, 10], [10, -1],
                    #Legs (mid hip included)
                    [17, 11], 
                    [11, 13], [13, 15], [15, -1],
                    [12, 14], [14, 16], [16, -1],
                   ]


def mean_var(data_list):
    '''
    Calculates the mean and total variance of all folds in a test series

    Arguments
    ---------
    data_list : List(Float)
        List of results from each fold
    
    Returns
    -------
    float, float
        Mean and variance across all folds

    '''
    for i, d in enumerate(data_list):
        data_list[i] *= 100
    mean = sum(data_list)/len(data_list)
    var_enum = 0
    for d in data_list:
        var_enum += (d - mean) ** 2
    total_var = var_enum / len(data_list)
    return mean, total_var

def ang(point_1, point_2, point_3):
    '''
    Calculates the bone angles from between 3 joints

    Arguments
    ---------
    point_1, point_2, point_3: List(Float)
        3D co-ordinates to calculate the angle between with joint 2 being the joint that connects to both
    
    Returns
    -------
    Float
        Returns the angle between the 3 joints
    '''
    #p12 is the length from 1 to 2, given by = sqrt((P1x - P2x)2 + (P1y - P2y)2)   
    p12 = np.sqrt(((point_1[0] - point_2[0]) ** 2) + ((point_1[1] - point_2[1]) ** 2))
    p13 = np.sqrt(((point_1[0] - point_3[0]) ** 2) + ((point_1[1] - point_3[1]) ** 2))
    p23 = np.sqrt(((point_2[0] - point_3[0]) ** 2) + ((point_2[1] - point_3[1]) ** 2))
    top = ((p12 ** 2) + (p13 ** 2) - (p23 ** 2))
    denominator = (2 * p12 * p13)
    #Prevent division by 0
    if denominator == 0:
        denominator = 0.01
    result = top/denominator
    #Keep within Acos range
    if result > 1:
        result = 0.99
    elif result < -1:
        result = -0.99
    #Calculate the angle given the 3 points
    ang = math.degrees(math.acos(result))
    return 180 - ang


def build_knee_joint_data(gait_cycles, images):
    '''
    This function builds a dataset of knee joint angles for HCF dataset construction.

    Arguments
    ---------
    gait cycles: List(List())
        Joints segmented by gait cycle
    images: List(List())
        Images corresponding to the joints for debugging
    
    Returns
    -------
    [List(List), List(List)]
        Returns a list of datasets, one for each knee

    '''
    l_angle_dataset = []
    r_angle_dataset = []
    image_iter = 0
    for i, gait_cycle in enumerate(gait_cycles):
        l_angles = []
        r_angles = []
        for j, frame in enumerate(gait_cycle):
            l_angles.append(ang(frame[19], frame[17], frame[21]))
            r_angles.append(ang(frame[20], frame[18], frame[22]))
            #print("angles left: ", ang(frame[16], frame[14], frame[18]), " and right: ", ang(frame[17], frame[15], frame[19]))
            #Render.render_joints(images[image_iter], frame, delay = True)
            image_iter += 1
        l_angle_dataset.append(l_angles)
        r_angle_dataset.append(r_angles)
    return [l_angle_dataset, r_angle_dataset]

def interpolate_knee_data(x, y, scale = 5000):
    '''
    Interpolates between data points of the knees to produce a smoother graph

    Arguments
    ---------
    x: List()
        list of x co-ordinates of the knee joint across a sequence
    y: List()
        list of y co-ordinates of the knee joint across a sequence
    scale: int (optional, default = 5000)
        the number of interpolated values to include
    
    Returns
    -------
    List(List), List(List)
        Returns the interpolated data and corresponding indices

    '''
    curr_length = len(x)
    inter_length = (curr_length -1) * scale
    inter_data = []
    inter_indices = [i for i in range(inter_length + curr_length)]
    for i, instance in enumerate(x):
        inter_data.append(instance)
        #Don't do it for final instance
        if i < len(x) - 1:
            angle_change = abs(instance - x[i + 1])
            for j in range(1, scale + 1):
                if instance < x[i+1]:
                    inter_data.append(instance + ((angle_change / scale) * j))
                else:
                    inter_data.append(instance - ((angle_change / scale) * j))
    return inter_data, inter_indices

def plot_velocity_vectors(image, joints_previous, joints_current, joints_next, debug = False, meta = 5):
    '''
    Creates velocity vectors for the velocity dataset generator

    Arguments
    ---------
    image: List()
        corresponding image to the joints being processed
    joints_previous: List()
        the joints of the frame t-1
    joints_current: List()
        the joints of frame t
    joints_next: List()
        the joints for frame t+1
    debug: bool (optional, default = False)
        indicates whether to output debug info
    meta: int (optional, default = 5)
        indicates how many metadata values to expect per-frame

    Returns
    -------
    List(List)
        Returns the velocity frame
    '''
    #Add meta data to the start of the row
    joint_velocities = joints_current[0:meta+1]

    #Convert to numpy arrays instead of lists
    joints_previous = list_to_arrays(joints_previous)
    joints_current = list_to_arrays(joints_current)
    joints_next = list_to_arrays(joints_next)

    #-3 to account for metadata at front
    for i in range(meta + 1, len(joints_current)):
        if len(joints_next) > 1:
            direction_vector = subtract_lists(joints_next[i], joints_current[i])
            end_point_after = [joints_current[i][0] + (direction_vector[0]), joints_current[i][1] + (direction_vector[1]), joints_current[i][2] + (direction_vector[2])]
            direction_after = subtract_lists(end_point_after, [joints_current[i][0], joints_current[i][1], joints_current[i][2]])
        else:
            direction_after = [0,0,0]

        if len(joints_previous) > 1:
            direction_vector = subtract_lists(joints_previous[i], joints_current[i])
            end_point_before = [joints_current[i][0] + (direction_vector[0]), joints_current[i][1] + (direction_vector[1]), joints_current[i][2] + (direction_vector[2])]
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

        if debug and image != None:
            x = int((smoothed_direction[1] * 40) + joints_current[i][1])
            y = int((smoothed_direction[0] * 40) + joints_current[i][0])
            image_direction = [int((smoothed_direction[1] * 40) + joints_current[i][1]),  int((smoothed_direction[0] * 40) + joints_current[i][0])]

            image = cv2.arrowedLine(image, [int(joints_current[i][1]), int(joints_current[i][0])] , image_direction,
                                            (0,255,0), 4) 
        joint_velocities.append(smoothed_direction)
    return joint_velocities

def process_data_input(joint_source, image_source, ignore_depth = True, cols = colnames_midhip):
    '''
    Utility for optionally loading both images and joint data in one function call

    Arguments
    ---------
    joint_source: str or List(List())
        file or object containing joint info
    image_source: str or List(List())
        root folder or object containing the images
    ignore_depth: bool (optional, default = True)
        indicates whether to include depth map frames when loading the image data
    cols: str (optional, default = colnames_midhip)
        indicates which column titles to use for the joint data
    
    Returns
    -------
    List(List), List(List)
        Returns the loaded joint and image datasets

    '''
    if isinstance(joint_source, str):
        joints = load(joint_source, colnames=cols)
    else:
        joints = joint_source
    
    if isinstance(image_source, str):
        images = load_images(image_source, ignore_depth=ignore_depth)
    else:
        images = image_source

    return joints, images

def save_dataset(data, name, colnames = colnames):
    '''
    Saves joint file objects into csv files

    Arguments
    ---------
    data: List(List())
        joint data to be saved
    name: str
        output file name
    colnames: List(str)
        list of column names for the dataset
    
    Returns
    -------
    None

    '''
    print("Saving joints")
    #Check for dataset type
    if len(data[0]) == 24:
        colnames = colnames_midhip
    elif len(data[0]) == 29:
        colnames = hcf_colnames
    elif len(data[0]) == 16:
        colnames = fused_colnames
    elif len(data[0]) == 20:
        colnames = colnames_nohead
    elif len(data[0]) == 23:
        colnames = colnames_default

    new_dataframe = pd.DataFrame(data, columns = colnames)
    file_name = name.split("/")
    file_name = file_name[-1]
    os.makedirs(name + "/raw/" ,exist_ok=True)
    for i, data in new_dataframe.iterrows():
        for j, unit in enumerate(data):
            if isinstance(unit, list):
                new_unit = []
                for number in unit:
                    new_unit.append(round(number, 4))
                new_dataframe.iat[i, j] = new_unit
    new_dataframe.to_csv(name + "/raw/" + file_name + ".csv",index=False, header=False, float_format='%.3f')    
        
def numericalSort(value):
    '''
    Utility function used for properly sorting image and joint folders, so that thew sequence is 1,2,3...9,10,11 instead of 1,2,10,11,12.. etc.

    Arguments
    ---------
    value: str
        input value to be considered, this is usually a file or folder name
    
    Returns
    -------
    str
        Returns the cut string with the actual number in the sequence to be considered.

    '''
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_images(folder, ignore_depth = True):
    '''
    Function to load a series of images from a root folder.

    Arguments
    ---------
    folder: str
        Root folder path for the images
    ignore_depth: bool (optional, default = True)
        indicates whether to include depth map images
    
    Returns
    -------
    List(List())
        returns a list of image objects

    '''
    image_data = []
    directory = os.fsencode(folder)
    for subdir_iter, (subdir, dirs, files) in enumerate(os.walk(directory)):
        dirs.sort(key=numericalSort)
        #Ignore base folder and instance 1 (not in dataset)
        if subdir_iter >= 1:
            for i, file in enumerate((pbar:= tqdm(files))):
                pbar.set_postfix_str(file)
                if i >= len(files) / 2 and ignore_depth:
                    break
                file_name = os.fsdecode(file)
                sub_dir = os.fsdecode(subdir)
                #display with openCV original image, overlayed with corresponding joints
                raw_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_COLOR)
                image_data.append(raw_image)
    return image_data

def save_images(joint_data, image_data, directory, include_joints = False, aux_joints = None):
    '''
    Function to save images into a series of folders

    Arguments
    ---------
    joint_data: List(List())
        joint value dataset
    image_data: List(List())
        dataset object of all the images
    directory: str
        root directory to begin saving the images
    include_joints: bool (optional, default = False)
        indicates whether to save the images with the joints overlaid on top of them
    aus_joints: List(List) (optional, default = None)
        indicates if there are any other joints to overlay
    
    Returns
    -------
    None
    '''
    print("Saving images")
    for i, row in enumerate((pbar := tqdm(joint_data))):

        pbar.set_postfix_str("Instance_" + str(float(row[0])))
        #Make sure it saves numbers properly (not going 0, 1, 10, 11 etc...)
        if row[1] < 10:
            file_no = str(0) + str(row[1])
        else:
            file_no = str(row[1])

        if include_joints:
            if aux_joints != None:
                image_data[i] = draw_joints_on_frame(image_data[i], joint_data[i], aux_joints=aux_joints[i])
            else:
                print("len: ", len(image_data), len(joint_data))
                image_data[i] = draw_joints_on_frame(image_data[i], joint_data[i], aux_joints=None)


        folder = str(directory) + "Instance_" + str(float(row[0]))
        #print("saving: ", folder + "/" + file_no + ".jpg")
        os.makedirs(folder, exist_ok = True)
        cv2.imwrite(folder + "/" + file_no + ".jpg", image_data[i])

def load(file, metadata = True, colnames = colnames_midhip):
    '''
    Function to load and process the raw joint data into a manipulable format.

    Arguments
    ---------
    file: str
        file path to the joints data
    metadata: bool (optional, default = True)
        indicates whether the source file contains metadata
    colnames: List(str) (optional, default = colnames_midhip)
        indicates which column titles to use for the data (not currently used)
    
    Returns
    -------
    List(List())
        Returns the loaded joints
    '''
    joints = []
    #Load in as a pandas dataset
    dataset = pd.read_csv(file, names=None, header=None)
    #Convert all data to literals
    dataset = convert_to_literals(dataset, metadata)
    #Convert to 2D array 
    joints = dataset.values.tolist()#to_numpy()
    #Print array to check
    return joints

def convert_to_literals(data, metadata = True, m = 5):
    '''
    This function interprets raw string data from csv files into numbers, arrays and lists.

    Arguments
    ---------
    data: Pd.DataFrame
        raw loaded joint data file
    metadata: bool (optional, default = True)
        indicates whether metadata is included in the dataset
    m: int (optional, default = 5)
        if metadata included, this indicates how much to expect
    
    Returns
    -------
    List(List())
        The raw dataset with the individual values inside processed into their proper datatypes.
    '''
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index > m and metadata == True or metadata == False:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = float(data.iat[i, col_index])

    return data

def convert_to_sequences(abs_data):
    '''
    Converts the structure of the joint data to be segmented by full video sequences

    Arguments
    ---------
    abs_data: List(List())
        unsegmented original joint data
    
    Returns
    -------
    List(List)
        returns the original data segmented into sub-lists denoting sequences

    '''
    #Iterate through abs_data, get head coords at 5 after first frame and 5 before last
    sequences = []
    sequence = []
    for i, joint in enumerate(abs_data):
        if i < len(abs_data) - 1:
            #If this number in sequence is a higher number than the next, then the next is a new
            #Instance
            if joint[1] > abs_data[i + 1][1]:
                sequence.append(joint)
                sequences.append(sequence)
                sequence = []
            else:        
                sequence.append(joint)
        else:
            #Just add last one to sequence
            sequence.append(joint)
    #Add last sequence which is the remainder
    sequences.append(sequence)
    return sequences

def get_3D_coords(coords_2d, dep_img, pts3d_net = True, dilate = True, meta_data = 5):
    '''
    Takes the depth map to calculate the 3D co-ordinates for the joint data

    Arguments
    ---------
    coords_2d: List()
        The original 2D pose data from HigherHRNet
    dep_img: List()
        The corresponding depth image to the pose data
    pts3d_net: bool (optional, default = True)
        indicates whether pts3d_net is available for 3D joint inference
    dilate: bool (optional, default = True)
        indicates whether to dilate the depth image
    metadata: int (optional, default = 5)
        indicates how much metadata to expect per frame
    
    Returns
    -------
    List(List), List(List)
        returns the original pixel data with 3D co-ordinates alongside another with the same thing but in metres.

    '''
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


def make_intrinsics(intrinsics_file = "Code/depth_intrinsics.csv"):
    '''
    Just creates the 3D intrinsics file with hard coding as it's always the same for the camera being used.

    Arguments
    ---------
    instrinsics_file: str (optional, default = "Code/depth_intrinsics.csv")
        path to the file containing camera intrinsics info
    
    Returns
    -------
    List()
        List of intrinsics values

    '''
    data_struct = [['212',	'120',	'107.02976989746094',
        '61.70092010498047',	'154.89523315429688',	'154.63319396972656',	
        'distortion.inverse_brown_conrady',	'[0.0, 0.0, 0.0, 0.0, 0.0]']]

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

#Series of simple functions for manipulating lists
#adding
def add_lists(list1, list2, change_type = False):
    return list((np.array(list1) + np.array(list2)).astype(int))
#converting to numpy arrays
def list_to_arrays(my_list):
    tmp_list = copy.deepcopy(my_list)
    for i, l in enumerate(tmp_list):
        if isinstance(l, list):
            tmp_list[i] = np.array(l)
    return tmp_list
#subtraction
def subtract_lists(list1, list2):
    result = np.subtract(list1, list2)
    return list(result)
#finding the midpoint between 2 co-ordinates
def midpoint(list1, list2):
    result = (np.array(list1) + np.array(list2)).astype(int)
    result = result / 2
    return list(result)
#Division
def divide_lists(list1, list2, n):
    return list((np.array(list1) + np.array(list2))/n)
#anonymizes a frame
def blacken_frame(frame):
    dimensions = (len(frame), len(frame[0]))
    blank_frame = np.zeros((dimensions[0],dimensions[1], 3), dtype= np.uint8)
    return blank_frame

def set_gait_cycles(data, preset_cycle):
    '''
    Sets an unsegmented set of joints to the same shape as an already segmented set of gait cycles

    Arguments
    ---------
    data: List(List)
        unsegmented joint data
    preset_cycle: List(List)
        an already segmented set of gait cycles
    
    Returns
    -------
    List(List())
        The original data in the shape of the preset gait cycles.
    '''
    new_cycles = []
    data_iter = 0
    for i, cycle in enumerate(preset_cycle):
        new_cycle = []
        for j, frame in enumerate(cycle):
            new_cycle.append(data[data_iter])
            data_iter += 1
        new_cycles.append(new_cycle)
    return new_cycles

def interpolate_gait_cycle(data_cycles, joint_output, step = 5, restrict_cycle = False):
    '''
    Interpolates gait cycles to expand or contract them all to a uniform length

    Arguments
    ---------
    data_cycles: List(List)
        gait cycles to interpolate
    joint_output: str
        file path for the output
    step: int (optional, default = 5)
        the number of extra frames to interpolate between frames
    restrict_cycle: bool (optional, default = False)
        indicates whether to restrict the maximum size of the gait cycle
    
    Returns
    -------
    List(List())
        The interpolated gait cyles
    '''
    inter_cycles = []
    min_cycle_count = min(len(sub_list) for sub_list in data_cycles) - 1
    for a, cycle in enumerate(data_cycles):
        inter_cycle = []
        #print("original cycle length: ", len(cycle))
        for i, frame in enumerate(cycle):
            if i < min_cycle_count or restrict_cycle == False:
                #Add the frame first
                inter_cycle.append(frame)

                #Ignore the last frame for interpolation
                if i < len(cycle) - 1:
                    inter_frames = interpolate_coords(frame, cycle[i + 1], step)
                    #Unwrap and add to full cycle 
                    for j in range(step):
                        inter_cycle.append(inter_frames[j])

        inter_cycles.append(inter_cycle)

    save_cycles = []
    for c in inter_cycles:
        for f in c:
            save_cycles.append(f)
    if joint_output != None:
        save_dataset(save_cycles, joint_output)
    return inter_cycles

def interpolate_coords(start_frame, end_frame, step):
    '''
    Interpolate between two frames step times.

    Arguments
    ---------
    start_frame: List(List)
        the start joints state for the interpolation
    end_frame: List(List)
        the end joint state for the interpolation
    step: int
        the number of intermediate frames to produce in the interpolation
    
    Returns
    -------
    List(List())
        A series of frames with start, the interpolated frames and the end frame.
    '''
    # Calculate the step size for interpolation
    inter_frames = []
    for i in range(1, step + 1):
        inter_frame = copy.deepcopy(start_frame)
        for j, coord in enumerate(start_frame):
            if j > 5:
                step_size = (np.array(end_frame[j]) - np.array(coord)) / (step + 1)
                # Perform interpolation and create the new lists
                interpolated_coord = coord + i * step_size
                #print("interpolated coord 1: ", type(interpolated_coord), interpolated_coord)   
                listed = list(interpolated_coord)
                #print("\ninterpolated coord 2: ", type(listed), listed)
                inter_frame[j] = listed
        inter_frames.append(inter_frame)
    return inter_frames

def check_within_radius(point1, point2, radius):
    '''
    Utility to check if a point 2 lies within a radius of point 1

    Arguments
    ---------
    point_1: List()
        the centre co-ordinate
    point_2: List()
        the query co-ordinate
    radius: the radius to check from point_1 as a centre
    
    Returns
    -------
    bool
        true or false to indicate whether the point lies within the radius
    '''
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance <= radius

def get_average_sequence(data):
    '''
    Get the average gait cycle for masking.

    Arguments
    ---------
    data: List(List())
        list of gait cycle of joints
    
    Returns
    -------
    List(List())
        The average gait cycle from the original data
    '''
    #Get the first sequence and divide it by the number of sequences
    result = data[0]
    for i , frame in enumerate(result):
        for j, coords in enumerate(frame):
            if j > 5:
                result[i][j] = [val / len(data) for val in coords]

    #Add every subsequent frame weighted by the number of sequences in the data
    for i, sequence in enumerate(data):
        if i > 0:
            for j, frame in enumerate(sequence):
                for k, coords in enumerate(frame):
                    if k > 5:
                        new_addition = [val / len(data) for val in coords]
                        try:
                            result[j][k] = [x + y for x, y in zip(result[j][k], new_addition)]
                        except:
                            pass
    return result

def assign_person_number(data_to_append, data, joint_output, no, start_instance, save = True):
    '''
    Used when stitched multiple single-person datasets together to ensure everyone has a unique person-ID

    Arguments
    ---------
    data_to_append: List(List())
        existing single or multi-person dataset to append to
    data: List(List())
        single person dataset to add the the full dataset
    joint_output: str
        path to the output file
    no: int
        number for the person being newly added
    start_instance: int
        starting instance for the new person so every instance has a sequential value in the full dataset
    save: bool (optional, default = True)
        indicates whether to save the intermediate dataset produced each time this function is called
    
    Returns
    -------
    List(List())
        The appended dataset now with n + 1 people
    '''
    current_instance = start_instance + 1
    for i, row in enumerate(data):
        data[i][5] = no
        if i > 0:
            if row[1] < data[i-1][1]:
                current_instance +=1
        
        data[i][0] = current_instance

    if data_to_append != None:
        #Append to a master dataset
        for d in data:
            data_to_append.append(d)
    if save:
        save_dataset(data_to_append, joint_output)
    return current_instance, data_to_append 

def convert_to_video(image_folder, output, file, depth = False):
    '''
    converts a series of images into a video

    Arguments
    ---------
    image_folder: str
        path to the image folder 
    output: str
        output path
    file: str
        output path name
    depth: bool (optional, default = False)
        indicates whether to include depth frames
    
    Returns
    -------
    None
    '''
    # Get the list of image files in the directory
    images = [img for img in os.listdir(image_folder) if str(img).split(".")[-1] == "jpg"]

    os.makedirs(output, exist_ok = True)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the output video file name and codec
    video_name = output + '/' + file + '.mp4'
    print("video name: ", video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 7, (width, height))
    d_video = None
    if depth:
        d_video_name = output + '/' + file + '_depth.mp4'
        print("video name: ", d_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        d_video = cv2.VideoWriter(d_video_name, fourcc, 7, (width, height))

    for iter,  image in enumerate(images):
        if iter < len(images) / 2 or depth == False:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
        else:
            #add frames to a depth video to go into the same folder destination
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            d_video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    if d_video:
        d_video.release()

def convert_to_9_class(data, joint_output):
    '''
    Converts a dataset from 3-class to 9-class

    Arguments
    ---------
    data: str or List(List())
        list of datasets or the path to their file location
    joint_output: str
        output file path

    Returns
    -------
    None
    '''
    data, _ = process_data_input(data, None)
    new_data = []
    for i, row in enumerate(data):
        #check row 2, 3 and 4 and place answer in row 2
        if row[2] == 0 and row[3] == 0 and row[4] == 0:
            row[2] = 0
        elif row[2] == 0 and row[3] == 1 and row[4] == 0:
            row[2] = 1
        elif row[2] == 0 and row[3] == 0 and row[4] == 1:
            row[2] = 2
        
        elif row[2] == 1 and row[3] == 0 and row[4] == 0:
            row[2] = 3
        elif row[2] == 1 and row[3] == 1 and row[4] == 0:
            row[2] = 4
        elif row[2] == 1 and row[3] == 0 and row[4] == 1:
            row[2] = 5

        elif row[2] == 2 and row[3] == 0 and row[4] == 0:
            row[2] = 6
        elif row[2] == 2 and row[3] == 1 and row[4] == 0:
            row[2] = 7
        elif row[2] == 2 and row[3] == 0 and row[4] == 1:
            row[2] = 8

        new_data.append(row)
    
    save_dataset(new_data, joint_output)