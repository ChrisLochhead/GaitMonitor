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

from ast import literal_eval
import pyrealsense2 as rs

joint_connections = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
[6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]


def save(joints):
    # open the file in the write mode
    #f = open('image_data.csv', 'w')

    with open('image_data.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        #Save the joints as a CSV
        for j, pt in enumerate(joints):
            for in_joint in pt:
                list = in_joint.flatten().tolist()
                row = [ round(elem, 4) for elem in list ]
                writer.writerow(row)
    # close the file
    #f.close()

def load(file = "image_data.csv"):
    joints = []
    person = []
    with open("image_data.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            person.append(row)
        joints.append(person)
    
    return joints

def run_video():
    print("initialising model")
    model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
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

def blacken_frame(frame):
    dimensions = (len(frame), len(frame[0]))
    blank_frame = np.zeros((dimensions[0],dimensions[1], 3), dtype= np.uint8)
    return blank_frame

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

def get_3D_coords(coords_2d, dep_img, pts3d_net = True, dilate = True, excl_2D = False ):

    pts_3D = []
    orig_dep_img = copy.deepcopy(dep_img)
    kernel = np.ones((6,6), np.uint8)
    if dilate == True:
        dep_img = cv2.dilate(dep_img, kernel, cv2.BORDER_REFLECT)

    dep_img = np.transpose(dep_img)
    for i in range(3, len(coords_2d)):
        
        x = int(coords_2d[i][0]); y = int(coords_2d[i][1])

        #Deal with any cut-off joints from the frame
        if y >= 424:
            y = 423
        if x >= 240:
            x = 239

        if pts3d_net == True:
            result = rs.rs2_deproject_pixel_to_point(loaded_intrinsics, [x, y], dep_img[y, x])
            if excl_2D == False:
                pts_3D.append([-result[0], -result[1], result[2]])
            else:
                pts_3D.append([x, y, result[2]])

        else:
            pts_3D.append([x,y, dep_img[(y,x)]])

    return pts_3D

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

def run_image(image_name, single = True, save = False, directory = None, model= None, image_no = 0):
    if model == None:
        model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)


    #Test loading function
    #joints = load("image_data.csv")
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

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False):

    tmp_frame = copy.deepcopy(frame)
    for joint in joints:
        #0 is X, Y is 1, 2 is confidence.
        if use_depth_as_colour == False:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(0, 0, 255), thickness=4)
        else:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(150, 100, joint[2]), thickness=4)
        #break
    return tmp_frame

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
            refined_joint_set = get_3D_coords(joint_set, depth_image, excl_2D=True)
            refined_joint_image = draw_joints_on_frame(raw_image, refined_joint_set, use_depth_as_colour=True)
            cv2.imshow('image with refined joints (excl 2D)',refined_joint_image)
            cv2.setMouseCallback('image with refined joints (excl 2D)', click_event, refined_joint_image)
            cv2.waitKey(0) & 0xff


            #display again with 3d positions including altered X and Y
            final_joint_set = get_3D_coords(joint_set, depth_image, excl_2D=False)
            final_joint_image = draw_joints_on_frame(raw_image, final_joint_set, use_depth_as_colour=True)
            cv2.imshow('image with refined joints (incl 2D)',final_joint_image)
            cv2.setMouseCallback('image with refined joints (incl 2D)', click_event, final_joint_image)
            cv2.waitKey(0) & 0xff


def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, ' ', y)
    if event == cv2.EVENT_RBUTTONDOWN:
        quit()


#Exclude 2D depth coord calculations to just get pixel data and depth value, otherwise it will return actual distances good for ML but
#not possible to properly display for debugging without re-projecting back to pixel data.
def run_images(folder_name, exclude_2D = False):
    directory = os.fsencode(folder_name)
    model = SimpleHigherHRNet(48, 17, "./weights/pose_higher_hrnet_w48_640.pth")
    file_iter = 0
    subdir_iter = 1
    data_class = 0
    #Format for the joints file
    #Instance Number: Sequence number: Class : Joint positions 1 - 17
    joints_file = []

    for subdir, dirs, files in os.walk(directory):
        if subdir_iter % 5 == 0:
            data_class += 1
            if data_class > 3:
                data_class = 1

        first_depth = True
        count_in_directory = 0

        for file in files:

            file_name = os.fsdecode(file)
            sub_dir = os.fsdecode(subdir)
            print("Sub directory: ", sub_dir, " Instance: ", file_iter - count_in_directory)

            out_directory = "./example_imgs/"
            
            os.makedirs(out_directory, exist_ok=True)

            #Once depth file has been found
            if file_name[0] == 'd':
                if first_depth == True:
                    count_in_directory = file_iter
                    first_depth = False
                    
                #Load depth image
                dep_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_ANYDEPTH)

                #Find corresponding raw image and extract co-ordinates
                joint_iter = 0

                refined_joints = get_3D_coords(joints_file[file_iter - count_in_directory], dep_image, excl_2D=exclude_2D)

                #print("what is joints file: ", joints_file[file_iter - count_in_directory])
                for i, dep_joint in enumerate(refined_joints):
                    if i >= 3:

                        #Make sure this isnt backwards during EDA
                        if int(dep_joint[0]) >= 424:
                            dep_joint[0] = 423
                        if int(dep_joint[1]) >= 240:
                            dep_joint[1] = 239

                        zDepth = dep_image[int(dep_joint[1])][int(dep_joint[0])]
    
                        #Fix this to record into the right place and then you are done
                        dep_joint = [dep_joint[0], dep_joint[1], zDepth]#[int(dep_joint[0]), int(dep_joint[1]), int(zDepth)]
                        joints_file[file_iter - count_in_directory][i] = dep_joint


            else:
                image, joints = run_image(sub_dir + "/" + file_name, single=False, save = True, directory=out_directory + sub_dir, model=model, image_no = file_iter)
                #print("joints returned: ", joints)
                if len(joints) < 1:
                    joints = [[ [0,0,0] for _ in range(17) ]]
                
                new_entry = [subdir_iter, file_iter, data_class]
                # 0 is instance, 1 is num in sequence, 2 is class, 3 is array of joints
                for i, joint in enumerate(joints[0]):
                    #Convert so it saves with comma delimiters within the joint-sets
                    tmp = [joint[0], joint[1], joint[2]]
                    new_entry.append(tmp)

                joints_file.append(new_entry)
            file_iter += 1
        subdir_iter +=1
        file_iter = 0
    #Save to .csv
    print("SAVING")
    if exclude_2D == True:
        with open("gait_dataset_pixels.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file)
    else:
        with open("gait_dataset_metres.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(joints_file)


def main():
    run_images("./Images", exclude_2D=True)
    print("Running images excluding metres x and y completed, running metres now.")
    run_images("./Images", exclude_2D=False)
    #run_depth_sample("./DepthExamples", "depth_examples.csv")
    #run_video()
if __name__ == '__main__':
    #Main menu
    main()