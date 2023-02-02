import cv2
import numpy as np
from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import check_video_rotation, draw_points_and_skeleton
import csv
import os 
import csv
import copy 
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
            print("this is one row: ", pt)
            for in_joint in pt:
                print("this is the next level down:" , in_joint)
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
            print("this is a row: ", row, "\n")
            person.append(row)
        joints.append(person)
    
    print("read in joints: ", joints)
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
                print("trying to break")
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

def get_3D_coords(coords_2d, dep_img, pts3d_net = True, dilate = True ):
    pts_3D = []
    orig_dep_img = copy.deepcopy(dep_img)
    kernel = np.ones((6,6), np.uint8)
    if dilate == True:
        dep_img = cv2.dilate(dep_img, kernel, cv2.BORDER_REFLECT)

    print(coords_2d)
    print("Coord 1: ", coords_2d[3])
    for i in range(3, len(coords_2d)):
        
        x = int(coords_2d[i][0]); y = int(coords_2d[i][1])
        if pts3d_net == True:
            result = rs.rs2_deproject_pixel_to_point(make_intrinsic(), [x, y], dep_img(y, x))
            pts_3D.append([-result[0], -result[1], result[2]])
        else:
            pts_3D.append(x,y, dep_img[(y,x)])

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
    #print("initialising model")
    if model == None:
        model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
        #print("model built")
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    #print("image read")

    #Test loading function
    #joints = load("image_data.csv")
    #apply joints to image for visualisation
    #print("printing joints for debug: \n" , joints)
    image, joints = get_joints_from_frame(model, image, anonymous=True)


    loop = True
    while loop == True:
        if single == True:
            cv2.imshow('Example', image)
            cv2.waitKey(0) & 0xff

        loop = False
    
    if save and directory != None:
        #print("saving to: ", directory + "/" + str(image_no) + ".jpg")
        cv2.imwrite(directory + "/" + str(image_no) + ".jpg", image)

    return image, joints

def run_images(folder_name):
    directory = os.fsencode(folder_name)
    #print("initialising model")
    model = SimpleHigherHRNet(48, 17, "./weights/pose_higher_hrnet_w48_640.pth")
    #print("model built")
    file_iter = 0
    subdir_iter = 0
    data_class = 0
    #Format for the joints file
    #Instance Number: Sequence number: Class : Joint positions 1 - 17
    joints_file = []

    for subdir, dirs, files in os.walk(directory):
        #print("new subdir: ", subdir)
        if subdir_iter % 10 == 0:
            data_class += 1

        first_depth = True
        count_in_directory = 0

        for file in files:

            file_name = os.fsdecode(file)
            sub_dir = os.fsdecode(subdir)
            print("Sub directory: ", sub_dir, " Instance: ", file_iter - count_in_directory)

            out_directory = "./example_imgs/"
            
            os.makedirs(out_directory, exist_ok=True)

            if file_name[0] == 'd':
                if first_depth == True:
                    count_in_directory = file_iter
                    first_depth = False
                    
                #Load depth image
                dep_image = cv2.imread(sub_dir + "/" + file_name, cv2.IMREAD_ANYDEPTH)

                #Find corresponding raw image and extract co-ordinates
                joint_iter = 0

                refined_joints = get_3D_coords(joints_file[file_iter - count_in_directory], dep_image)

                for i, dep_joint in enumerate(refined_joints):
                    if i >= 3:
                        print("DEP JOINTs: ", dep_joint)
                        if all(j == 0 for j in dep_joint) == False:

                            #Make sure this isnt backwards during EDA
                            if int(dep_joint[0]) >= 424:
                                dep_joint[0] = 423
                            if int(dep_joint[1]) >= 240:
                                dep_joint[1] = 239

                            zDepth = dep_image[int(dep_joint[1])][int(dep_joint[0])]
                            print("Recorded coordinates: ", int(dep_joint[0]), int(dep_joint[1]), zDepth)

                            #Fix this to record into the right place and then you are done
                            dep_joint = [dep_joint[0], dep_joint[1], zDepth]#[int(dep_joint[0]), int(dep_joint[1]), int(zDepth)]
                            joints_file[file_iter - count_in_directory][i] = dep_joint
                        else:
                            print("It's a float or an int: ", dep_joint)
                            joints_file[file_iter - count_in_directory][i] = [0,0,0]


            else:
                image, joints = run_image(sub_dir + "/" + file_name, single=False, save = True, directory=out_directory + sub_dir, model=model, image_no = file_iter)
                if len(joints) < 1:
                    joints = [[ [0,0,0] for _ in range(17) ]]
                
                new_entry = [subdir_iter, file_iter, data_class]
                # 0 is instance, 1 is num in sequence, 2 is class, 3 is array of joints
                print("joints: ", joints)
                for i, joint in enumerate(joints[0]):
                    #Convert so it saves with comma delimiters within the joint-sets
                    tmp = [joint[0], joint[1], joint[2]]
                    new_entry.append(tmp)

                print("New Entry: ", new_entry)
                joints_file.append(new_entry)
            file_iter += 1
        subdir_iter +=1
        file_iter = 0
    #Save to .csv
    print("SAVING")
    with open("example_dataset.csv","w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(joints_file)


def main():
    run_images("./Images")
    #run_video()
if __name__ == '__main__':
    #Main menu
    main()