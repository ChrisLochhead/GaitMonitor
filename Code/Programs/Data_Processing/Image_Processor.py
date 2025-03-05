'''
This file contains the processing functions for using HigherHRNet to extract joint positions from raw images.
'''
#imports
import cv2
import os 
import csv
#dependencies
import Programs.Data_Processing.Utilities as Utilities
from Programs.Data_Processing.Render import *
from Programs.Machine_Learning.Simple_HigherHRNet import SimpleHigherHRNet

def get_joints_from_frame(model, frame, anonymous = True):
    '''
    This function extracts the pose from a skeleton in a single frame

    Arguments
    ---------
    model : HigherHRNet Object
        Pose estimation model
    frame : List
        Image to estimate the pose from
    anonymous : bool (optional, default = True)
        denotes whether to remove the frame when drawing
    
    Returns
    -------
    List, List
        Returns the frame and joints corresponding to it

    '''
    joints = model.predict(frame)
    if anonymous:
        frame = Utilities.blacken_frame(frame)

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

def run_image(image_name, single = True, save = False, directory = None, model= None, image_no = 0):
    '''
    This function estimates the pose of a frame and saves it

    Arguments
    ---------
    image_name : str
        image file location
    single : bool
        denotes whether the input is a single image for debugging
    save : bool
        denotes whether to save
    directory : bool
        denotes whether a directory has been included
    model : HigherHRNet model
        model for pose estimation
    image_no : int
        denotes the number in the sequence to save it

    Returns
    -------
    List, List
        Returns the frame and joints corresponding to it

    '''
    if model == None:
        model = SimpleHigherHRNet.SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
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

def run_images(folder_name, out_folder, write_mode = "w+", start_point = 0):
    '''
    This function loops through a series of directories, extracting the poses from the images and saving the resulting joint files.

    Arguments
    ---------
    folder_name : str
        name of the root directory
    out_folder : str
        name of the output folder root
    write_mode : str (optional, default = "w+")
        write mode for continual saving or saving as a new file
    start point : int (optional, default = 0)
        denotes to start from the nth folder
    
    Returns
    -------
    None

    '''
    model = SimpleHigherHRNet.SimpleHigherHRNet(32, 17, "./Code/Programs/Machine_Learning/Simple_HigherHRNet/weights/pose_higher_hrnet_w32_512.pth")
        #Location info for the dataset
    file_iter = 0
    subdir_iter = 0
    #Check between normals, freezes and obstacles
    phase_iter = 1
    #metadata
    data_class = 0
    obstacle = 0
    freeze = 0
    person = -2
    #Format for the joints file
    #Instance Number: Sequence number: Class : Joint positions 1 - 17
    joints_file = []
    joints_file_metres = []
    for directory_iter, (subdir, dirs, files) in enumerate(os.walk(os.fsencode(folder_name))):
        dirs.sort(key=Utilities.numericalSort)
        #Skip any instances already recorded
        if directory_iter < start_point:
            subdir_iter += 1
            continue

        if subdir_iter % 20 == 0 and subdir_iter != 0:
            data_class += 1
            if data_class > 2:
                data_class = 0

        #First 10 of the phase are just normal walks
        if phase_iter <= 10:
            freeze = 0
            obstacle = 0
        #Next 5 are freezes
        if phase_iter > 10 and phase_iter < 15:
            freeze = 1
            obstacle = 0
        #Final 5 are obstacles
        elif phase_iter > 15:
            freeze = 0
            obstacle = 1
        #first_depth = True
        count_in_directory = 0
        for f, file in enumerate(files):

            file_name = os.fsdecode(file)
            if file_name[0] == 'd':
                continue
            sub_dir = os.fsdecode(subdir)
            print("Sub directory: ", sub_dir, " Instance: ", file_iter - count_in_directory)
            print("subdir iter: ", subdir_iter)

            #Not currently used, only for saving images with overlaid joints
            out_directory = "./example_imgs/"
            os.makedirs(out_directory, exist_ok=True)

             #Get joints from initial image
            image, joints = run_image(sub_dir + "/" + file_name, single=False, save = False, 
                                      directory=out_directory + sub_dir, model=model, image_no = file_iter)
            # Get corresponding depth image, this is always same index += half the length of the instance

            #dep_image = cv2.imread(sub_dir + "/" + os.fsdecode(files[int(f + (len(files)/2))]), cv2.IMREAD_ANYDEPTH)
            dep_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dep_image = dep_image.astype(np.uint16)  # Depth images are often 16-bit
            # Create a zeroed depth image with the same shape and type
            dep_image = np.zeros_like(dep_image, dtype=dep_image.dtype)

            print("types: ", type(dep_image), type(image))

            if len(joints) > 0:
                if len(joints[0]) < 1:
                    joints = [ [0,0,0] for _ in range(17) ]
            
            if len(joints) > 0:
                refined_joints, refined_joints_metres = Utilities.get_3D_coords(joints[0], dep_image, meta_data=0)
            else:
                refined_joints = [ [0,0,0] for _ in range(17) ]
                refined_joints_metres = [ [0,0,0] for _ in range(17) ]

            new_entry = [subdir_iter, file_iter, data_class, freeze, obstacle, person]
            new_metres_entry = [subdir_iter, file_iter, data_class, freeze, obstacle, person]
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
        if len(files) > 0:
            phase_iter += 1
            if phase_iter == 21:
                phase_iter = 1
            subdir_iter +=1
        else:
            person += 1

        #Save after every folder
        if os.path.exists( out_folder + "Absolute_Data.csv"):
            write_mode = "a"
        else:
            write_mode = "w"

        #Save to .csv
        print("SAVING", write_mode)
        os.makedirs(out_folder, exist_ok=True)
        with open( out_folder + "Absolute_Data.csv", write_mode, newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file)
            joints_file = []
        with open(out_folder + "Absolute_Data_Metres.csv",write_mode, newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file_metres)
            joints_file_metres = []
        file_iter = 0
