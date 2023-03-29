import cv2
import numpy as np
from SimpleHigherHRNet import SimpleHigherHRNet
import csv
import os 
import csv
import pandas as pd
from ast import literal_eval

from scripts.DatasetBuilder.utilities import *
from scripts.DatasetBuilder.render import *

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

def load_and_overlay_joints(directory = "./Images", joint_file = "./EDA/gait_dataset_pixels.csv", ignore_depth = True, plot_3D = False):
    joints = load(joint_file)
    #print("joints: ", joints)
    subdir_iter = 1
    joint_iter = 0
    for i, (subdir, dirs, files) in enumerate(os.walk(directory)):
        #print("current subdirectory in overlay function: ", subdir)
        dirs.sort(key=numericalSort)
        for j, file in enumerate(files):
            #Ignore depth images which are second half
            if j >= len(files)/2 and ignore_depth:
                continue

            file_name = os.fsdecode(file)
            sub_dir = os.fsdecode(subdir)
        
            #display with openCV original image, overlayed with corresponding joints
            raw_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_COLOR)
            print("image and joints: ", joint_iter, sub_dir + "/" + file_name)
            #i - 2 because iter starts at 1, and the first empty subdir also counts as 1.
            render_joints(raw_image, joints[joint_iter], delay = True, use_depth = True)
            #if plot_3D:
            #    plot3D_joints(joints[joint_iter])
            joint_iter += 1
        subdir_iter += 1
        #Debug
        #if subdir_iter >= 4:
        #    break


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
    

