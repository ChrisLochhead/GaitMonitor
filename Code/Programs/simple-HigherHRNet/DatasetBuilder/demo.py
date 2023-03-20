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

from DatasetBuilder.utilities import *

def run_video():
    print("initialising model")
    model = SimpleHigherHRNet(32, 17, ".././weights/pose_higher_hrnet_w32_512.pth")
    cap = cv2.VideoCapture(0)
    width, height = 200, 200
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # get the final frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    #out = cv2.VideoWriter('output.avi', -1, 20.0, (512,512))
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame, joints = get_joints_from_frame(model, frame)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_joints_from_frame(model, frame, anonymous = True):
    joints = model.predict(frame)

    if anonymous:
        frame = blacken_frame(frame)

    for person in joints:

        for joint_pair in joint_connections:
            #Draw links between joints
            tmp_a = person[joint_pair[1]]
            tmp_b = person[joint_pair[0]]
            start = [int(float(tmp_a[1])), int(float(tmp_a[0]))]
            end = [int(float(tmp_b[1])), int(float(tmp_b[0]))]

            cv2.line(frame, start, end, color = (0,255,0), thickness = 2) 

            #Draw joints themselves
            for joint in person:
                #0 is X, Y is 1, 2 is confidence.
                frame = cv2.circle(frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(0, 0, 255), thickness=4)

    return frame, joints

def load_and_overlay_joints(directory = "./Images"):
    joints = load("./EDA/gait_dataset_pixels.csv")
    subdir_iter = 1
    joint_iter = 0
    for i, (subdir, dirs, files) in enumerate(os.walk(directory)):
        
        for j, file in enumerate(files):
            #Ignore depth images which are second half
            if j >= len(files)/2:
                break

            file_name = os.fsdecode(file)
            sub_dir = os.fsdecode(subdir)
        
            #display with openCV original image, overlayed with corresponding joints
            raw_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_COLOR)
            #i - 2 because iter starts at 1, and the first empty subdir also counts as 1.
            render_joints(raw_image, joints[joint_iter], delay = True, use_depth = True)
            plot3D_joints(joints[joint_iter])
            joint_iter += 1
        subdir_iter += 1
        #Debug
        if subdir_iter >= 4:
            break