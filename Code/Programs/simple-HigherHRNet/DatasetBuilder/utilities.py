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

from ast import literal_eval
import pyrealsense2 as rs

import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


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

joint_connections = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 0], [12, 0], # hips to origin
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [6, 8], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin

colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 

occlusion_boxes = [[140, 0, 190, 42], [190, 0, 236, 80] ]

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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        print("current subdirectory in utility function: ", subdir)
        #Ignore base folder and instance 1 (not in dataset)
        if subdir_iter >= 1:
            for i, file in enumerate(files):
                if i >= len(files) / 2 and ignore_depth:
                    break
                file_name = os.fsdecode(file)
                sub_dir = os.fsdecode(subdir)
                
                #display with openCV original image, overlayed with corresponding joints
                raw_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_COLOR)
                image_data.append(raw_image)
    
    return image_data

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

def get_time_cycles(joint_data, images):
    instances = []
    instance = []
    threshold = 5
    current_instance = 0
    for row_iter, row in enumerate(joint_data):
        current_instance = row[0]

        instance.append(row)
        if len(instance) >= threshold:
            instances.append(copy.deepcopy(instance))
            print("A final length: ", len(instance))
            instance = []
            continue 

        if len(joint_data) > row_iter + 1:
            if joint_data[row_iter + 1][0] != current_instance:
                #Add to previous to make sure no cycle is less than 5
                if len(instance) < threshold:
                    for inst in instance:
                        instances[-1].append(inst)
                #instances.append(copy.deepcopy(instance))
                print("B final length: ", len(instance))
                instance = []
                continue

    print("Time instances made:, lengths: ")
    iter = 0
    cycle_iter = 0
    for cycle in instances:
        print("length: ", len(cycle))
        iter += len(cycle)
        cycle_iter += 1
    print("total number of instances: ", iter, " number of cycles: ", cycle_iter)

    return instances
        


#This will only work with relative data
def get_gait_cycles(joint_data, images):
    instances = []
    instance = []
    threshold = 15
    ahead = 1
    ahead_previous = 1
    changes = 0
    draw_colour = (0,255,0)
    current_instance = 0
    #Firstly, separate all data into arrays of each instance
    for row_iter, row in enumerate(joint_data):
        if len(joint_data) > row_iter + 1 and len(instance) > 1:
            if joint_data[row_iter + 1][0] != current_instance:
                current_instance = row[0]
                instance.append(row)
                print("-A detected overlap, resetting", len(instance))
                instances.append(copy.deepcopy(instance))
                if draw_colour == (0, 255, 0):
                    draw_colour = (0, 0, 255)
                else:
                    draw_colour = (0, 255, 0)

                instance = []
                changes = 0
                continue

        #Check for a new instance
        print("lens: ", len(images), len(joint_data))
        #Take note of what last ahead was
        ahead_previous = ahead
        #Going left as usual
        if row[19][1] > row[18][1] + threshold:
            if ahead_previous == 1:
                changes += 1
                instance.append(row)
                if changes > 2:
                    print("A detected overlap, resetting", len(instance))
                    instances.append(copy.deepcopy(instance))
                    if draw_colour == (0, 255, 0):
                        draw_colour = (0, 0, 255)
                    else:
                        draw_colour = (0, 255, 0)

                    instance = []
                    changes = 0
            else:
                instance.append(row)
            print("A", row[18][1], row[19][1])
            ahead = 0

        #Going right as usual
        elif row[18][1] > row[19][1] + threshold:
            if ahead_previous == 0:
                changes += 1
                instance.append(row)            
                if changes > 2:   
                    print("B detected overlap, resetting", len(instance))             
                    instances.append(copy.deepcopy(instance))
                    if draw_colour == (0, 255, 0):
                        draw_colour = (0, 0, 255)
                    else:
                        draw_colour = (0, 255, 0)
                    instance = []
                    changes = 0
            else:
                instance.append(row)
            ahead = 1

            print("B", row[18][1], row[19][1])
        else:
            instance.append(row)



        print("current row values: ", row[18][1], row[19][1], draw_colour)

        #if ahead == 1:
        #    render_joints(images[row_iter], row, delay=True, use_depth=False, colour=draw_colour)
        #else:
        #   render_joints(images[row_iter], row, delay=True, use_depth=False, colour=draw_colour)

    #Debug drawing, just draw regular blue for left, red for right
    print("gait cycle retreival completed")
    counter = 0
    cycle_count = 0

    blue_cycle = False
    for i, inst in enumerate(instances):
        print("new cycle of length: ", len(inst), cycle_count, i)
        for j, joints_frame in enumerate(inst):            
            if i % 2 == 0:
                render_joints(images[counter], joints_frame, delay=True, use_depth=False, colour=(0,0, 255))
            else:
                render_joints(images[counter], joints_frame, delay=True, use_depth=False, colour=(255,0, 0))
            counter += 1
        #New Cycle incoming
        cycle_count += 1
    
    return instances

def get_stride_lengths(gait_cycles, images, abs_gait_cycles):
    stride_lengths = []
    stride_ratios = []
    image_iter = 0

    print("total rel and abs: ", len(gait_cycles), len(abs_gait_cycles))
    for i, frame in enumerate(gait_cycles):
        max_stride_lengths =[0,0]
        stride_ratio = 0
        #Get max stride length values in this cycle
        for j, joints in enumerate(frame):
            if joints[18][1] > max_stride_lengths[0]:
                print ("this:", joints[18][1] , "is bigger than ", max_stride_lengths[0])
                max_stride_lengths[0] = copy.deepcopy(joints[18][1])
            if joints[19][1] > max_stride_lengths[1]:
                print ("this:", joints[19][1] , "is bigger than ", max_stride_lengths[1])
                max_stride_lengths[1] = copy.deepcopy(joints[19][1])
            print("relative joints: ", joints)
            print("absolute joints: ", abs_gait_cycles[i][j])
            print("JOINTS: ", max_stride_lengths)
            #render_joints(images[image_iter], abs_gait_cycles[i][j], delay=True)
            image_iter += 1
        stride_lengths.append(max_stride_lengths)
        stride_ratio = max_stride_lengths[0]/max_stride_lengths[1]
        print("final max lengs and ratio: ", max_stride_lengths, stride_ratio)
        stride_ratios.append(stride_ratio)

    return stride_lengths, stride_ratios
            

def get_speed(abs_joints, gait_cycles, images):

    #instance_speeds = []
    #current_instance = -1
    #instance_iter = 0
    #for i, joint in enumerate(abs_joints):
    #    if joint[0] == current_instance:
    #        instance_iter += 1
    #        continue
    #    else:
    #        last = joint[i-1]
    #        first = joint[i - instance_iter]
    #        speed = np.sqrt((abs(last[0] - first[0])**2) + (abs(last[1] - first[1])**2) + (abs(last[2] - first[2])**2))
       
    speeds = []
    image_iter = 0
    for i, cycle in enumerate(gait_cycles):

        speed = 0
        #Sometimes first frame can be messy, get second one.
        first = cycle[0]
        #Get last frame (some last ones corrupted, get third from last)
        last = cycle[-1]
        #Get the average speed throughout the frames
        speed = np.sqrt((abs(last[0] - first[0])**2) + (abs(last[1] - first[1])**2) + (abs(last[2] - first[2])**2))
        print("cumulative speed: ", speed)
        speeds.append(speed/len(cycle))
        print("added average speed: ", speed/len(cycle), len(cycle))
        render_joints(images[image_iter], cycle[0])

        for frame in cycle:
            image_iter += 1
    return speeds


def get_time_LofG(gait_cycles, velocity_joints, images):
    frames_off_ground_array = []
    both_not_moving_array = []
    threshold = 0.55
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        frames_off_ground = [0,0]
        frames_not_moving = 0
        for j, joints in enumerate(frame):
            left_velocity = abs(velocity_joints[image_iter][18][0]) + abs(velocity_joints[image_iter][18][1])+ abs(velocity_joints[image_iter][16][0])+ abs(velocity_joints[image_iter][16][1])
            right_velocity = abs(velocity_joints[image_iter][19][0]) + abs(velocity_joints[image_iter][19][1])+ abs(velocity_joints[image_iter][17][0])+ abs(velocity_joints[image_iter][17][1])


            print("velocities: ", left_velocity, right_velocity)
            #Left leg moving, leg is off ground
            if left_velocity > right_velocity and left_velocity >= right_velocity + threshold:
            #if velocity_joints[image_iter][18][0] > 0 + threshold or velocity_joints[image_iter][18][0] < 0 - threshold:
                frames_off_ground[0] += 1
                print("left leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Right leg moving
            elif right_velocity > left_velocity and right_velocity >= left_velocity + threshold:
            #elif velocity_joints[image_iter][19][0] > 0 + threshold or velocity_joints[image_iter][19][0] < 0 - threshold:
                frames_off_ground[1] += 1
                print("right leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Neither leg moving, this is double support
            else:
            #elif velocity_joints[image_iter][18][0] < 0 + threshold and velocity_joints[image_iter][18][0] > 0 - threshold \
            #    and velocity_joints[image_iter][19][0] < 0 + threshold and velocity_joints[image_iter][19][0] > 0 - threshold:
                    print("neither leg off ground")
                    frames_not_moving += 1
                    #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
           
            image_iter += 1

        frames_off_ground_array.append(copy.deepcopy(frames_off_ground))
        both_not_moving_array.append(frames_not_moving)
    
    return frames_off_ground_array, both_not_moving_array
            

def get_feet_height(gait_cycles, images):
    feet_heights = []  
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        total_feet_height = [0,0]
        for j, joints in enumerate(frame):
            #Illustrate feet height by gap between nose and foot to indicate
            #changing height from the ground
            total_feet_height[0] += abs(joints[18][0] - joints[3][0])
            total_feet_height[1] += abs(joints[19][0] - joints[3][0])
            
            #print("feet heights: ", feet_heights, total_feet_height, len(gait_cycles))
            #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(255,0, 0))
            image_iter += 1

        #Take the average height across the gait cycle
        average_feet_height = [0,0]
        average_feet_height[0] = total_feet_height[0] / len(gait_cycles[i])
        average_feet_height[1] = total_feet_height[1] / len(gait_cycles[i])
        feet_heights.append(copy.deepcopy(average_feet_height))

    return feet_heights

def get_cadence(gait_cycles, images):
    cadences = []
    cadence = 0
    threshold = 15
    ahead = 0
    ahead_previous = 0
    image_iter = 0

    for i, frame in enumerate(gait_cycles):
        cadence = 0
        print("CADENCE RESET: ", cadence)
        for j, joints in enumerate(frame):
            #Take note of what last ahead was
            ahead_previous = ahead
            #Going left as usual
            if joints[19][1] > joints[18][1] + threshold:
                if ahead_previous == 1:
                    cadence += 1
                    print("A detected overlap, cadence", cadence)
                print("A", joints[18][1], joints[19][1])
                ahead = 0
            #Going right as usual
            elif joints[18][1] > joints[19][1] + threshold:
                if ahead_previous == 0:   
                    cadence += 1        
                    print("B detected overlap, cadence", cadence)                
                ahead = 1
                print("B", joints[18][1], joints[19][1])

            if ahead == 1:
                render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            else:
                render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(255,0, 0))

            image_iter += 1
        cadences.append(cadence)
    
    return cadences



def create_hcf_dataset(jointfile, rel_jointfile, abs_veljointfile, folder, save = True):
    abs_joint_data = load(jointfile)
    rel_joint_data = load(rel_jointfile)
    abs_veljoint_data = load(abs_veljointfile)
    images = load_images(folder, ignore_depth=False)

    print("images and joints loaded, getting gait cycles...")
    #Gait cycle has instances of rows: each instance contains an array of rows
    #denoting all of the frames in their respective gait cycle, appended with their
    #metadata at the start
    #gait_cycles = get_gait_cycles(abs_joint_data, images)
    gait_cycles = get_time_cycles(abs_joint_data, images)
    #Replicate this pattern for the relative gait cycle (if youn run it directly, 
    #relative gait will return very marginally different gait cycles.)
    rel_gait_cycles = []
    joint_counter = 0
    for cycle in gait_cycles:
        new_cycle = []
        for frame in cycle:
            new_cycle.append(rel_joint_data[joint_counter])
            joint_counter += 1
        rel_gait_cycles.append(copy.deepcopy(new_cycle))

    print("number of total gait cycles: ", len(gait_cycles))
    return
    #Then for every gait cycle create a new instance with the following features: 
    #Cadences returns a scalar for every gait cycle returning the number of steps 
    #cadences = get_cadence(gait_cycles, images)

    #Height of feet above the ground
    #returns feet heights using distance between each foot joint and head in relative
    #terms using absolute data. returned as array of 2 element lists for each foot
    feet_heights = get_feet_height(gait_cycles, images)


    #Time leg off of ground + time both feet not moving
    #if foot velocity > 0 + threshold and math.dist(foot, head) > threshold then foot is in motion, add to vector
    #once this no longer holds, change colour and show
    #implement for left and right leg
    times_LOG, times_not_moving = get_time_LofG(gait_cycles, abs_veljoint_data, images)

    #Speed
    #get absolute values of the head at first frame vs last, divide number of frames by that distance
    speeds = get_speed(gait_cycles, images)

    #Stride length + variance
    #get max and min from left leg relative to head
    #get corresponding values in absolute values for leg
    #math.dist to get the distance between these two values
    #repeat for both legs
    stride_lengths, stride_ratios = get_stride_lengths(rel_gait_cycles, images, gait_cycles)

    #Combine all hand crafted features into one concrete dataset, save and return it.
    gait_cycles_dataset = []
    for i, cycle in enumerate(gait_cycles):
        #Add metadata
        hcf_cycle = [cycle[0][0], cycle[0][1], cycle[0][2]]
        #hcf_cycle.append(cadences[i])
        hcf_cycle.append(feet_heights[i][0])
        hcf_cycle.append(feet_heights[i][1])
        hcf_cycle.append(times_LOG[i][0])
        hcf_cycle.append(times_LOG[i][1])
        hcf_cycle.append(times_not_moving[i])
        hcf_cycle.append(speeds[i])
        hcf_cycle.append(stride_lengths[i][0])
        hcf_cycle.append(stride_lengths[i][1])
        hcf_cycle.append(stride_ratios[i])
        gait_cycles_dataset.append(copy.deepcopy(hcf_cycle))

    if save:
        with open("./EDA/Finished_Data/hcf_dataset_pixels.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(gait_cycles_dataset)

    return gait_cycles_dataset



def blacken_frame(frame):
    dimensions = (len(frame), len(frame[0]))
    blank_frame = np.zeros((dimensions[0],dimensions[1], 3), dtype= np.uint8)
    return blank_frame

def filter_coords(joints, index, metadata = 3):
    coords = []
    for i, j in enumerate(joints):
        if i >= metadata:
            coords.append(j[index])
    
    return coords

def plot3D_joints(joints):
    # generate data
    x = filter_coords(joints, 0)
    y = filter_coords(joints, 1)
    z = filter_coords(joints, 2)

    # axes instance
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    #424 by 240
    ax.set_xlim([0, 240])
    ax.set_ylim([0, 424])
    ax.set_zlim3d([0, 255])
    plt.gca().invert_yaxis()
    plt.gca().invert_zaxis()
    #z forward -90, 180
    #on the corner -175, 120
    ax.view_init(-175, 120)

    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

    # plot
    sc = ax.scatter(x, y, z, s=40, c=x, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # legend
    #-90, 180, 0 angle, azimuth and roll
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    #plt.draw()#block=True)
    plt.show(block=True)
    #plt.waitforbuttonpress(0) # this will wait for indefinite time
    #plt.close(fig)
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight')

def render_joints(image, joints, delay = False, use_depth = True, metadata = 3, colour = (255, 0, 0)):
    tmp_image = copy.deepcopy(image)
    tmp_image = draw_joints_on_frame(tmp_image, joints, use_depth_as_colour=use_depth, metadata = metadata, colour=colour)
    cv2.imshow('Joint Utilities Image',tmp_image)

    cv2.setMouseCallback('Joint Utilities Image', click_event, tmp_image)
    if delay:
        cv2.waitKey(0) & 0xff

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        quit()

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False, metadata = 3, colour = (0, 0, 255)):

    tmp_frame = copy.deepcopy(frame)
    tmp_joints = copy.deepcopy(joints)

    for joint_pair in joint_connections:
            #Draw links between joints
            tmp_a = tmp_joints[joint_pair[1] + metadata]
            tmp_b = tmp_joints[joint_pair[0] + metadata]
            start = [int(float(tmp_a[1])), int(float(tmp_a[0]))]
            end = [int(float(tmp_b[1])), int(float(tmp_b[0]))]

            cv2.line(tmp_frame, start, end, color = (0,255,0), thickness = 2) 


    for i, joint in enumerate(tmp_joints):

        if isinstance(joint, int):
            continue
        #0 is X, Y is 1, 2 is confidence.

        #Clamp joints within frame size
        
        if joint[0] >= 240:
            joint[0] = 239
        if joint[1] >= 424:
            joint[1] = 423
        
        if use_depth_as_colour == False:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=colour, thickness=4)
        else:
            #if i < len(tmp_joints) - 1:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(150, 100, joint[2]), thickness=4)
            #else:
            #    tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(255, 0, 0), thickness=10)
       # break

    return tmp_frame

def save(joints):
    # open the file in the write mode
    with open('image_data.csv', 'w', newline='') as outfile:
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
