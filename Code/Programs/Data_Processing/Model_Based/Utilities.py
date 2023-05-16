import cv2
import numpy as np
import csv
import os 
import csv
import copy
import pandas as pd
import ast
from ast import literal_eval
import pyrealsense2 as rs
import re
from tqdm import tqdm
import math
#from Render import render_joints, plot3D_joints 
import Programs.Data_Processing.Model_Based.Render as Render
'''order of joints:

0,1,2 are meta data:

actual: Format
0: nose
1: left eye
2: right eye
3: left ear
4: right ear
5: left shoulder
6: right shoulder
7: left elbow
8: right elbow
9: left hand
10: right hand
11: left hip
12: right hip
13: left knee 
14: right knee 
15: left foot 18
16: right foot 19

No chest
'''   

colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 

#Occlusion co-ordinates for home recording dataset
occlusion_boxes = [[140, 0, 190, 42], [190, 0, 236, 80] ]

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg

def build_knee_joint_data(gait_cycle):
    l_angles = []
    r_angles = []

    for i, frame in enumerate(gait_cycle):
        #Left hip to left knee, left knee to left foot
        lh_lk = [frame[14], frame[16]]
        lk_lf = [frame[16], frame[18]]

        #Right hip to right knee, right knee to right foot
        rh_rk = [frame[15], frame[17]]
        rk_rf = [frame[17], frame[19]]

        l_angles.append(ang(lh_lk, lk_lf))
        r_angles.append(ang(rh_rk, rk_rf))

    return [l_angles, r_angles]

def interpolate_knee_data(x, y, scale = 5000):
    curr_length = len(x)
    inter_length = (curr_length -1) * scale
    inter_data = []
    inter_indices = [i for i in range(inter_length + curr_length)]
    for i, instance in enumerate(x):
        #print("appending initial: ", instance)
        inter_data.append(instance)
        #Don't do it for final instance
        if i < len(x) - 1:
            angle_change = abs(instance - x[i + 1])
            #inter_changes = np.logspace(np.log(x[i+1]), np.log(instance), scale, base=np.exp(1))

            for j in range(1, scale + 1):
                #print("interpolating from ", instance, " to ", x[i + 1])
                #print("current: ", j, " of 4. Value is: ", inter_changes[j-1])
                if instance < x[i+1]:
                    inter_data.append(instance + ((angle_change / scale) * j))
                    #print("added value: ", instance + ((angle_change/scale) * j))
                else:
                    inter_data.append(instance - ((angle_change / scale) * j))
                    #print("added value: ", instance - ((angle_change/scale) * j))
    
    print("final lens: ", len(inter_data), len(inter_indices))
    return inter_data, inter_indices

def split_by_class_and_instance(data):
    norm, limp, stag = split_by_class(data)
    normal_instances = split_class_by_instance(norm)
    limp_instances = split_class_by_instance(limp)
    stagger_instances = split_class_by_instance(stag)

    return normal_instances, limp_instances, stagger_instances

#Split classes into instances, work out how to know where one ends and next begins (using sequencing)
def split_class_by_instance(data):
    filtered_data = []
    instance = []
    current_frame = 0
    last_frame = 0
    print("length of data: ", len(data))
    for i, row in enumerate(data):
        last_frame = current_frame
        current_frame = data[i][1]
        
        if current_frame < last_frame:
            filtered_data.append(copy.deepcopy(instance))
            instance = []

        instance.append(row)
            
    return filtered_data

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

        if debug:
            x = int((smoothed_direction[1] * 40) + joints_current[i][1])
            y = int((smoothed_direction[0] * 40) + joints_current[i][0])
            image_direction = [int((smoothed_direction[1] * 40) + joints_current[i][1]),  int((smoothed_direction[0] * 40) + joints_current[i][0])]

            image = cv2.arrowedLine(image, [int(joints_current[i][1]), int(joints_current[i][0])] , image_direction,
                                            (0,255,0), 4) 
        joint_velocities.append(smoothed_direction)

    return joint_velocities

def generate_relative_sequence_data(sequences, rel_data):
    rel_sequence_data = []
    iterator = 0
    for i, sequence in enumerate(sequences):
        rel_sequence = []
        for j, frame in enumerate(sequence):
            rel_sequence.append(rel_data[iterator])
            iterator += 1
        rel_sequence_data.append(rel_sequence)
    return rel_sequence_data

#Split data into the three classes
def split_by_class(data):
    regular = []
    limp = []
    stagger = []
    for index, row in enumerate(data):
        if data[index][2] == 0:
            regular.append(copy.deepcopy(row))
        if data[index][2] == 1:
            limp.append(copy.deepcopy(row))
        if data[index][2] == 2:
            stagger.append(copy.deepcopy(row))

    return regular, limp, stagger
           

def process_data_input(joint_source, image_source):
    if isinstance(joint_source, str):
        joints = load(joint_source)
    else:
        joints = joint_source
    
    if isinstance(image_source, str):
        images = load_images(image_source)
    else:
        images = image_source

    return joints, images
        
def convert_to_sequences(abs_data):
    pass
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

def save_dataset(data, name, colnames = colnames):
    print("Saving joints")
    new_dataframe = pd.DataFrame(data, columns = colnames)
    #Convert to dataframe 
    new_dataframe.to_csv(name,index=False, header=False)     


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

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

#Add function to outline velocity change by drawing arrow based on velocity of -1 and +1 frame
def load_images(folder, ignore_depth = True):
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

def make_intrinsics(intrinsics_file = "Code/depth_intrinsics.csv"):
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
    return list(result)

def midpoint(list1, list2):
    result = (np.array(list1) + np.array(list2)).astype(int)
    result = result / 2
    return list(result)

def divide_lists(list1, list2, n):
    return list((np.array(list1) + np.array(list2))/n)

def divide_list(list1, n):
    return list((np.array(list1))/n)

def blacken_frame(frame):
    dimensions = (len(frame), len(frame[0]))
    blank_frame = np.zeros((dimensions[0],dimensions[1], 3), dtype= np.uint8)
    return blank_frame

def apply_class_labels(num_switches, num_classes, joint_data):
    switch_iterator = 0
    current_class = 0
    for i, row in enumerate(joint_data):
        if switch_iterator == num_switches:
            current_class += 1
            if current_class > num_classes:
                current_class = 0
            
            switch_iterator = 0

        joint_data[i][2] = current_class

        #if there is at least one more datapoint next
        if i < len(joint_data) - 1:
            if joint_data[i][1] > joint_data[i+1][1]:
                switch_iterator  += 1
    
    return joint_data

def save_images(joint_data, image_data, directory):

    print("Saving images")
    for i, row in enumerate((pbar := tqdm(joint_data))):

        pbar.set_postfix_str("Instance_" + str(float(row[0])))
        #Make sure it saves numbers properly (not going 0, 1, 10, 11 etc...)
        if row[1] < 10:
            file_no = str(0) + str(row[1])
        else:
            file_no = str(row[1])

        folder = str(directory) + "Instance_" + str(float(row[0]))
        os.makedirs(folder, exist_ok = True)
        cv2.imwrite(folder + "/" + file_no + ".jpg", image_data[i])

def save(joints, name):
    # open the file in the write mode
    with open(name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        #Save the joints as a CSV
        for j, pt in enumerate(joints):
            for in_joint in pt:
                list = in_joint.flatten().tolist()
                row = [ round(elem, 4) for elem in list ]
                writer.writerow(row)

def load(file = "image_data.csv"):
    joints = []
    #Load in as a pandas dataset
    colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
          'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 
    dataset = pd.read_csv(file, names=colnames, header=None)

    #Convert all data to literals
    dataset = convert_to_literals(dataset)

    #Convert to 2D array 
    joints = dataset.to_numpy()
    #Print array to check
    return joints


def convert_to_literals(data):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index >= 3:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = int(data.iat[i, col_index])

    return data
