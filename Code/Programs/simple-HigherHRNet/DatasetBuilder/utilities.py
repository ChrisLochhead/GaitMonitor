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
15: left foot 
16: right foot

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

#def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
#    """Compute the time-derivative of a Lorentz system."""
#    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint as ODEint

def chart_single_joint(joints):
    x = []
    y = []
    z = []
    for j in joints:
        x.append(j[0])
        y.append(j[1])
        z.append(j[2])
    

    #x =  np.linspace(0, 20, 1000)
    #y, z = 10.*np.cos(x), 10.*np.sin(x) # something simple

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot(x, y, z)

    # now Lorentz
    times = np.linspace(0, 4, 1000) 

    start_pts = 30. - 15.*np.random.random((20,3))  # 20 random xyz starting values

    trajectories = []
    for start_pt in start_pts:
        print("incomplete")
        #trajectory = ODEint(lorentz_deriv, start_pt, times)
        #trajectories.append(trajectory)

    ax = fig.add_subplot(1,2,2,projection='3d')
    for trajectory in trajectories:
        x, y, z = trajectory.T  # transpose and unpack 
        # x, y, z = zip(*trajectory)  # this also works!
        ax.plot(x, y, z)

    plt.show()

#This will only work with relative data
def get_gait_cycles(joint_data):
    instances = []
    instance = []
    current_instance = -1

    #Firstly, separate all data into arrays of each instance
    for row in joint_data:
        if current_instance == -1:
            current_instance = row[0]
        
        #Reset and add the instance once you get to a new instance
        if current_instance != -1 and current_instance != row[0]:
            instances.append(copy.deepcopy(instance))
            instance = [row[0], row[2]]
            current_instance = row[0]
        
        #Keep metadata for each cycle, ignore number in cycle
        print("row: ", row, "\nrow after cut: ", row[3:])
        instance.append([row[3:]])


    #Instances contain every set of joints for each instance in one array, topped
    #with the class and instance metadata
    right_feet = []
    for instance in instances:
        for joints in instance:
            #3D coords of foot instance
            right_foot = joints[18]
            #Get all right foot values into an array
            right_feet.append(right_foot)

        #Chart them as a line chart and view cut-off as percentage for threshold
        chart_single_joint(right_feet)
        break
        #

def create_hcf_dataset(abs_joint_data, rel_joint_data):
    print("incomplete")
    #For every joint instance, first extract each gait cycle

    #Then for every gait cycle create a new instance with the following features: 

    #This will require relative dataset for positions and absolute
    #calculate cadence
        #get left foot, 
        #split into sequences of decreasing and increasing
            #if i > j add to decreasing vector (within threshold), if i < j add to increasing vector (within threshold)
            #count the number of these sequentially
            #whenever it changes direction, iterate cadence counter
            #draw image and change colour of foot when increasing or decreasing
        #double number of these instances, these are the number of steps

    #Height of feet above the ground
    #get math.dist between head and left + right foot for a total of 2 values

    #Time leg off of ground
    #if foot velocity > 0 + threshold and math.dist(foot, head) > threshold then foot is in motion, add to vector
    #once this no longer holds, change colour and show
    #implement for left and right leg

    #Speed
    #get absolute values of the head at first frame vs last, divide number of frames by that distance

    #Stride length
    #get max and min from left leg relative to head
    #get corresponding values in absolute values for leg
    #math.dist to get the distance between these two values
    #repeat for both legs

    #Stride length variance
    #Get ratio of left to right leg stride length, clamp between 0-1 or -0.5 to 0.5

    #Time both feet not moving
    # if foot velocity > 0 + threshold and math.dist(foot, head) > threshold then foot is in motion, add to vector
    #Add every frame that this isn't the case 


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
    ax.view_init(-90, 180)

    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

    # plot
    sc = ax.scatter(x, y, z, s=40, c=x, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # legend
    #-90, 180, 0 angle, azimuth and roll
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show(block=True)
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight')

def render_joints(image, joints, delay = False, use_depth = True, metadata = 3):
    tmp_image = copy.deepcopy(image)
    tmp_image = draw_joints_on_frame(tmp_image, joints, use_depth_as_colour=use_depth, metadata = metadata)
    cv2.imshow('Joint Utilities Image',tmp_image)

    cv2.setMouseCallback('Joint Utilities Image', click_event, tmp_image)
    if delay:
        cv2.waitKey(0) & 0xff

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, ' ', y)
    if event == cv2.EVENT_RBUTTONDOWN:
        quit()

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False, metadata = 3):

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
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(0, 0, 255), thickness=4)
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
