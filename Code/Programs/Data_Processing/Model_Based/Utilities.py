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
0: nose 3
1: left eye 4
2: right eye 5
3: left ear 6
4: right ear 7 
5: left shoulder 8 
6: right shoulder 9 
7: left elbow 10
8: right elbow 11
9: left hand 12
10: right hand 13
11: left hip 14
12: right hip 15
13: left knee 16
14: right knee 17
15: left foot 18
16: right foot 19
17: mid-hip 20

No chest
'''   

colnames=['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 

colnames_midhip = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', "M_hip"] 

colnames_top = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand'] 

colnames_bottom = ['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', "M_hip"] 


hcf_colnames = ["Instance", "No_In_Sequence", "Class", 'Freeze', 'Obstacle', 'Person', "Feet_Height_0", "Feet_Height_1",
                 "Time_LOG_0", "Time_LOG_1", "Time_No_Movement", "Speed", "Stride_Gap", "Stride_Length", "Max_Gap", 'l_co 1',
                 'l_co 2', 'l_co 3', 'l_co 4', 'l_co 5', 'l_co 6', 'l_co 7', 'r_co 1', 'r_co 2', 'r_co 3', 'r_co 4', 'r_co 5', 'r_co 6', 'r_co 7']

                #Head goes to hip, all other head joints go to nothing because they will be dropped
bone_connections =[[0, 17], [1, -1], [2, -1], [3, -1], [4, -1],
                    #Right arm (extremities like hands and feet go to 0)
                    [5, 7], [7, 9], [9, -1],
                    #Left arm
                    [6, 8], [8, 10], [10, -1],
                    #Legs (mid hip included)
                    [17, 11], 
                    [11, 13], [13, 15], [15, -1],
                    [12, 14], [14, 16], [16, -1],
                   ]

#Occlusion co-ordinates for home recording dataset
occlusion_boxes = [[140, 0, 190, 42], [190, 0, 236, 80] ]

# explicit function to normalize array
def normalize_1D(arr, t_min = 0, t_max = 1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def mean_var(data_list):
    for i, d in enumerate(data_list):
        data_list[i] *= 100

    mean = sum(data_list)/len(data_list)
  
    var_enum = 0
    for d in data_list:
        var_enum += (d - mean) ** 2

    total_var = var_enum / len(data_list)

    return mean, total_var

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def slope(x1, y1, x2, y2): # Line slope given two points:
    denom = x2 - x1
    if denom == 0:
        denom = 0.01
    return (y2-y1)/denom


#p12 is the length from 1 to 2, given by = sqrt((P1x - P2x)2 + (P1y - P2y)2)
#arccos((P12^2 + P13^2 - P23^2) / (2 * P12 * P13))
#Where point 1 is the angle point and 2 and 3 are the extremities. (so 1 is the knee)
def ang(point_1, point_2, point_3):
    #p12 is the length from 1 to 2, given by = sqrt((P1x - P2x)2 + (P1y - P2y)2)   
    p12 = np.sqrt(((point_1[0] - point_2[0]) ** 2) + ((point_1[1] - point_2[1]) ** 2))
    p13 = np.sqrt(((point_1[0] - point_3[0]) ** 2) + ((point_1[1] - point_3[1]) ** 2))
    p23 = np.sqrt(((point_2[0] - point_3[0]) ** 2) + ((point_2[1] - point_3[1]) ** 2))

    #print("point 1 {}, point 2: {}, point 3: {}".format(point_1, point_2, point_3))
    #print("p12, p13, p23: ", p12, p13, p23)
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

    #print(" top: {},  \ndenominator: {} \nand result: {}".format(top, denominator, result))
    #Calculate the angle given the 3 points
    ang = math.degrees(math.acos(result))
    return 180 - ang


def build_knee_joint_data(gait_cycles, images):

    l_angle_dataset = []
    r_angle_dataset = []
    image_iter = 0
    for i, gait_cycle in enumerate(gait_cycles):
        l_angles = []
        r_angles = []
        for j, frame in enumerate(gait_cycle):
            #Left hip to left knee, left knee to left foot
            lh_lk = [frame[17], frame[19]]
            lk_lf = [frame[19], frame[21]]

            #Right hip to right knee, right knee to right foot
            rh_rk = [frame[18], frame[20]]
            rk_rf = [frame[20], frame[22]]

            l_angles.append(ang(frame[19], frame[17], frame[21]))
            r_angles.append(ang(frame[20], frame[18], frame[22]))

            #print("angles left: ", ang(frame[16], frame[14], frame[18]), " and right: ", ang(frame[17], frame[15], frame[19]))
            #Render.render_joints(images[image_iter], frame, delay = True)
            image_iter += 1
        
        l_angle_dataset.append(l_angles)
        r_angle_dataset.append(r_angles)

    return [l_angle_dataset, r_angle_dataset]

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
    
    return inter_data, inter_indices

def split_by_class_and_instance(data):
    print("data before split: ", data[0])
    norm, limp, stag = split_by_class(data)
    print("data after split 1: ", norm[0])
    normal_instances = split_class_by_instance(norm)
    print("data after split 2: ", normal_instances[0])
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

#Returns a list of average values in a dataset
def create_average_data_sample(data, meta = 5):
    dataset_size = len(data)
    running_total = data[0]
    for i, row in enumerate(data):
        if i > 0:
            running_total = [x + y if j > meta else x for j, (x, y) in enumerate(zip(running_total, row))]
    
    average = [x / dataset_size for x in running_total[0]]
    return average

def plot_velocity_vectors(image, joints_previous, joints_current, joints_next, debug = False, meta = 5):
    
    #Add meta data to the start of the row
    #joint_velocities = [joints_current[0], joints_current[1], joints_current[2]]
    joint_velocities = joints_current[0:meta+1]

    #Convert to numpy arrays instead of lists
    joints_previous = list_to_arrays(joints_previous)
    joints_current = list_to_arrays(joints_current)
    joints_next = list_to_arrays(joints_next)

    #-3 to account for metadata at front
    for i in range(meta + 1, len(joints_current)):
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
           

def process_data_input(joint_source, image_source, ignore_depth = True, cols = colnames_midhip):
    if isinstance(joint_source, str):
        joints = load(joint_source, colnames=cols)
    else:
        joints = joint_source
    
    if isinstance(image_source, str):
        images = load_images(image_source, ignore_depth=ignore_depth)
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
    #Check for dataset type
    if len(data[0]) == 24:
        colnames = colnames_midhip
    elif len(data[0]) == 13:
        colnames = colnames_bottom
    elif len(data[0]) == 17:
        colnames = colnames_top
    elif len(data[0]) == 29:
        colnames = hcf_colnames


    new_dataframe = pd.DataFrame(data, columns = colnames)
    
    file_name = name.split("/")
    file_name = file_name[-1]

    if len(data[0]) == 26:
        print("size: ", new_dataframe.shape)
        print(new_dataframe.head())
    #Convert to dataframe 
    os.makedirs(name + "/raw/" ,exist_ok=True)
    new_dataframe.to_csv(name + "/raw/" + file_name + ".csv",index=False, header=False)     


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
        #print("saving: ", folder + "/" + file_no + ".jpg")
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

def load(file = "image_data.csv", metadata = True, colnames = colnames_midhip):
    joints = []
    #Load in as a pandas dataset
    dataset = pd.read_csv(file, names=colnames, header=None)

    print("dataset head: ", dataset.head())
    #Convert all data to literals
    dataset = convert_to_literals(dataset, metadata)

    #Convert to 2D array 
    joints = dataset.to_numpy()
    #Print array to check
    return joints


def convert_to_literals(data, metadata = True, m = 5):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index > m and metadata == True or metadata == False:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = float(data.iat[i, col_index])

    return data
