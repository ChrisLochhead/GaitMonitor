import cv2
import numpy as np
from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import check_video_rotation, draw_points_and_skeleton
import csv
import os 



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
    print("type: ", type(frame))
    print("dimensions: ", dimensions)
    blank_frame = np.zeros((dimensions[0],dimensions[1], 3), dtype= np.uint8)
    print("type now: ", type(blank_frame))
    return blank_frame

def get_joints_from_frame(model, frame, anonymous = True):
    joints = model.predict(frame)

    if anonymous:
        frame = blacken_frame(frame)

    for person in joints:
        print("number of joints: ", len(person))

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
    print("initialising model")
    if model == None:
        model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
        print("model built")
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    print("image read")

    #Test loading function
    #joints = load("image_data.csv")
    #apply joints to image for visualisation
    #print("printing joints for debug: \n" , joints)
    image, joints = get_joints_from_frame(model, image, anonymous=True)


    loop = True
    while loop == True:
        cv2.imshow('Example', image)
        if single == True:
            cv2.waitKey(0) & 0xff
        #print("saving: ")
        #save(joints)
        print("shutting program")

        loop = False
    
    if save and directory != None:
        print("saving to: ", directory + "/" + str(image_no) + ".jpg")
        cv2.imwrite(directory + "/" + str(image_no) + ".jpg", image)

def run_images(folder_name):
    directory = os.fsencode(folder_name)
    print("initialising model")
    model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    print("model built")
    iter = 0
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        print("filename: ", file_name)
        out_directory = "./example_imgs/"
        
        os.makedirs(out_directory, exist_ok=True)
        run_image(folder_name + "/" + file_name, single=False, save = True, directory=out_directory, model=model, image_no = iter)
        iter += 1

#migrate this file into main project.
#Add this as option for main functions
#Done with setup :) 
def main():
    run_images("./Images")
    #run_video()
if __name__ == '__main__':
    #Main menu
    main()