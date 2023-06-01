import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import re
import cv2
import csv
import math

from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *
from Programs.Machine_Learning.Model_Based.GCN.Render import *

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

#Add function to outline velocity change by drawing arrow based on velocity of -1 and +1 frame
def load_images(folder, ignore_depth = True):
    image_data = []
    directory = os.fsencode(folder)
    for subdir_iter, (subdir, dirs, files) in enumerate(os.walk(directory)):
        dirs.sort(key=numericalSort)
        print("types: ", type(subdir), type(dirs), type(files))
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


def assess_data(dataset):
    print("Dataset type: ", type(dataset), type(dataset[0]))
    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of nodes: {dataset[0].num_nodes}')
    print(f'Number of classes: {dataset.num_classes}')
    #Print individual node information
    data = dataset[0]
    print(f'x = {data.x.shape}')
    print(data.x)
    print(data.y.shape, type(data.y), data.y)
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')
    #Plot graph data using networkX
    plot_graph(data)


def create_dataloaders(dataset, train = 0.8, val = 0.9, test = 0.9):
        # Create training, validation, and test sets
        print("created dataloader lengths: {} : {} : {}".format(len(dataset),int(len(dataset)*train), int(len(dataset)*val) - int(len(dataset)*train) ))
        train_dataset = dataset[:int(len(dataset)*train)]
        val_dataset   = dataset[int(len(dataset)*train):int(len(dataset)*val)]
        test_dataset  = dataset[int(len(dataset)*test):]

        print(f'Training set   = {len(train_dataset)} graphs')
        print(f'Validation set = {len(val_dataset)} graphs')
        print(f'Test set       = {len(test_dataset)} graphs')

        # Create mini-batches
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True )
        #print("train loader: ", len(train_loader))
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, val_loader, test_loader, test_dataset

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

def split_data_by_viewpoint(joints_file, save = True):
   
    #Initialize 3 viewpoint arrays
    normal_viewpoint = []
    mod_viewpoint = []

    colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
          'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] 
    
    dataset_master = pd.read_csv(joints_file, names=colnames, header=None)

    for index, row in dataset_master.iterrows():
        #The last one is greater than this: this means a new instance is found.
        if row[0] <= 29:
            normal_viewpoint.append(row)
        else:
            mod_viewpoint.append(row)

    norm = pd.DataFrame(normal_viewpoint, columns = colnames)
    mod = pd.DataFrame(mod_viewpoint, columns = colnames)

    #Save and return final joints and view in excel to see if they worked
    if save:
        norm.to_csv("gait_dataset_pixels_norm_view.csv",index=False, header=False)
        mod.to_csv("gait_dataset_pixels_mod_view.csv",index=False, header=False)

    return norm, mod

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
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(150, 100, joint[2]), thickness=4)


    return tmp_frame

def convert_to_literals(data):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index >= 3:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = int(data.iat[i, col_index])

    return data

#This is from Hrh Utilities
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

global_append = None

#Replace this and draw joints on frame with version found in demo for higher hr net
def ground_truth_event(event, x, y, flags, params):
    global added_coord
        # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        row = params[0]
        estimate = params[1]
        # displaying the coordinates
        print(x, y)
        #Keep original depth value as dummy
        if added_coord == False:
            row.append([x, y, estimate[2]])
        else:
            print("replacing selection: ")
            row[-1] = [x, y, estimate[2]]
        added_coord = True

#Replace this and draw joints on frame with version found in demo for higher hr net
def frame_select_event(event, x, y, flags, params):
    global global_append
        # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        row = params[0]
        estimate = params[1]
        good_frame = params[2]
        index = params[3]
        create_grounds = params[4]
        # displaying the coordinates
        print(x, y)
        #Keep original depth value as dummy
        if index == 0:
            if global_append == None:
                good_frame.append([1])
                global_append = True
            else:
                good_frame[-1]= [1]
            print("this is a good frame")
            print("returning appended  = ", global_append)
        if create_grounds == True:
            row.append([x, y, estimate[2]])

    if event == cv2.EVENT_RBUTTONDOWN:
        good_frame = params[2]
        index = params[3]
        if index == 0:
            if global_append == None:
                good_frame.append([0])
                global_append = True
            else:
                good_frame[-1] = [0]
            print("this is a bad frame")
            print("returning appended  = ", global_append)

def select_good_frames(image_path, joints, save = True, create_grounds = False, start_point = 42): #Start point is -1 because first folder is empty
    global global_append
    joint_iter = 0
    ground_truth_joints = []
    is_good_frame = []
    class_no = 0
    joints = load("pixel_data_absolute.csv")
    directory_iterator = -1
    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        #Assign the class
        if iterator + 1 % 5 == 0:
            if class_no < 2:
                class_no += 1
            else:
                class_no = 0
        
        if len(files) > 0 and directory_iterator >= start_point:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                print("in file iterator: directory: ", directory_iterator)
                #Assign the meta data
                ground_truth_row = [iterator, file_iter, class_no]
                img = cv2.imread(os.path.join(subdir,file))
                img = draw_joints_on_frame(img, joints[joint_iter], use_depth_as_colour = False, metadata = 3, colour = (0, 0, 255))
                cv2.imshow('image',img)

                print("section: ", start_point, directory_iterator)
                print("frame: ", file_iter, " of ", len(files))
                print("total frame: ", joint_iter, "of ", len(joints), "corresponding: ", len(is_good_frame))
                while global_append == None:
                    cv2.setMouseCallback('image', frame_select_event, [ground_truth_row, joints[joint_iter], is_good_frame, i, create_grounds])
                    #Only iterate 18 times if actually setting ground truths
                    print("appended: ", global_append)
                    key = cv2.waitKey(0)  
                global_append = None
                print("key: ", key)
                if key == 113:
                    print("trying to quit?")
                    quit()
                elif key == 115:
                    print("SAVING PROGRESS: ")
                    with open('image_deletion_mask.csv', 'a+', newline='') as f:
                        mywriter = csv.writer(f, delimiter = ",")
                        mywriter.writerows(is_good_frame)   
                        is_good_frame = []       

                    break                   
                ground_truth_joints.append(ground_truth_row)
                joint_iter += 1

        #iterate joint iter regardless of start point
        elif len(files)> 0:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                joint_iter += 1

        directory_iterator +=1
        print("saving instance")
        if save:
            with open("ground_truth_dataset_pixels.csv","a+", newline='') as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(ground_truth_joints)
                ground_truth_joints = []

            with open('image_deletion_mask.csv', 'a+', newline='') as f:
                mywriter = csv.writer(f, delimiter = ",")
                mywriter.writerows(is_good_frame) 
                is_good_frame = []         

def communicate_ground_truths(scores):
    means = []
    for i, joint_score in enumerate(scores):
        print("score before: ", joint_score)
        score = joint_score[3:]
        print("score after", score)
        means.append(np.mean(score))
    
    total_mean = np.mean(means)
    total_std  = np.std(means)

    print("Total mean: ", total_mean, " total std: ", total_std)

def evaluate_ground_truths(predictions, ground_truths, images):
    similarity_scores = []
    for i, pred in enumerate(predictions):
        for j, truth in enumerate(ground_truths):
            similarity_score = []
            #If instance and sequence no are identical, they are the same image
            if pred[0] == ground_truths[0] and pred[1] == ground_truths[1]:
                similarity_score = [pred[0], pred[1], pred[2]]
                cv2.imshow('Corresponding image: ',images[i])
                for k, coords in enumerate(pred):
                    #Ignore metadata
                    if k > 2:
                        similarity_score.append(math.dist(pred(k), truth(k)))
        if len(similarity_score) > 0:
            similarity_scores.append(similarity_score)
    
    #Save scores to csv
    with open('similarity_scores.csv', 'a+', newline='') as f:
        mywriter = csv.writer(f, delimiter = ",")
        mywriter.writerows(similarity_scores)   
        similarity_scores = []       

joint_dict = {
0: "nose",
1: "left eye",
2: "right eye",
3: "left ear",
4: "right ear",
5: "left shoulder",
6: "right shoulder",
7: "left elbow",
8: "right elbow",
9: "left hand",
10: "right hand",
11: "left hip",
12: "right hip",
13: "left knee",
14: "right knee", 
15: "left foot", #18
16: "right foot" }

added_coord = False

def create_ground_truths(image_path, joints, save = True, create_grounds = False, start_point = -1, file_start_point = 0): #Start point is -1 because first folder is empty
    global joint_dict
    global added_coord

    joint_iter = 0
    ground_truth_joints = []
    class_no = 0
    joints = load("pixel_data_absolute.csv")
    directory_iterator = -1
    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        #Assign the class
        if iterator + 1 % 5 == 0:
            if class_no < 2:
                class_no += 1
            else:
                class_no = 0
        
        if len(files) > 0 and directory_iterator >= start_point:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                if file_iter < file_start_point:
                    continue
                print("in file iterator: directory: ", directory_iterator)
                #Assign the meta data
                ground_truth_row = [iterator, file_iter, class_no]
                img = cv2.imread(os.path.join(subdir,file))
                #img = draw_joints_on_frame(img, joints[joint_iter], use_depth_as_colour = False, metadata = 3, colour = (0, 0, 255))
                cv2.imshow('image',img)

                #For each of the 18 joints in the frame
                for i in range(0, 17):
                    while added_coord == False:
                        print("add joint: ", joint_dict[i])
                        cv2.setMouseCallback('image', ground_truth_event, [ground_truth_row, joints[joint_iter]])
                        key = cv2.waitKey(0)  
                    added_coord = False

                ground_truth_joints.append(ground_truth_row)
                print("SAVING PROGRESS: ")
                with open('ground_truths.csv', 'a+', newline='') as f:
                    mywriter = csv.writer(f, delimiter = ",")
                    mywriter.writerows(ground_truth_joints)   
                    ground_truth_joints = []       
                
                joint_iter += 1

        #iterate joint iter regardless of start point
        elif len(files)> 0:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                joint_iter += 1

        directory_iterator +=1

def load_mask(mask_path):
    data = []
    with open(mask_path, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    for i, point in enumerate(data):
        data[i] = data[i][0]
        data[i] = int(data[i])
    
    return data

def interpolate_joints(joints_1, joints_2):
    int_joints = [joints_1[0], joints_1[1] + 1, joints_1[2]]
    print("joint: ", joints_1)
    for i, joint in enumerate(joints_1):
        if i > 2:
            int_joints.append([(joint[0]+joints_2[i][0])/2, \
                                        (joint[1]+joints_2[i][1])/2, \
                                        (joint[2]+joints_2[i][2])/2])
    return int_joints


def run_image_deletion_mask(image_file, joint_file, mask_path, interpolate = True):

    proc_images = []
    proc_joints = []
    mask = load_mask(mask_path)
    image_data = load_images(image_file, ignore_depth=False)
    joint_data = load(joint_file)

    print("len images: ", len(image_data))
    for i, image in enumerate(image_data):
        print("value found: ", mask[i])
        if mask[i] == 1:
            proc_images.append(image)
            proc_joints.append(joint_data[i])
        #Interpolate a new frame of joints, check not first or last value
        elif mask[i] == 0 and i > 0 and i < len(image_data) - 1 and interpolate == True:
            #Check the before and after are both positive examples
            if mask[i-1] == 1 and mask[i + 1] == 1:
                #Finally, check both are in the same instance
                if joint_data[i-1][0] == joint_data[i+1][0]:
                    proc_images.append(image)
                    #Interpolate joints
                    proc_joints.append(interpolate_joints(joint_data[i-1], joint_data[i+1]))
                    print("lens: ", len(proc_joints), len(proc_images))
                    #render_joints(image, joint_data[i], delay=True)


    for j, row in enumerate(proc_joints):
        print("saving instance: ", str(float(row[0])))
        if row[1] < 10:
            file_no = str(0) + str(row[1])
        else:
            file_no = str(row[1])

        #Save Images
        directory = "./Manually_Processed_Images/Instance_" + str(float(row[0]))
        print("i is: ", j , len(proc_images), len(proc_joints))
        print("saving: ", directory + "/" + file_no + ".jpg")
        os.makedirs(directory, exist_ok = True)
        cv2.imwrite(directory + "/" + file_no + ".jpg", proc_images[j])

    #Save joints
    with open("MPI_pixels_omit.csv","w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(proc_joints)

    display_images_and_joints("MPI_pixels_omit.csv", "./Manually_Processed_Images/", "image_deletion_mask.csv")

def display_images_and_joints(joint_file, image_file, mask_file = None):
    print("displaying files and joints...")
    joint_data = load(joint_file)
    image_data = load_images(image_file, ignore_depth=False)
    if mask_file:
        mask = load_mask(mask_file)
    print("lens: ", len(joint_data), len(image_data))
    image_iter = 0
    for index, j in enumerate(joint_data):
        if mask_file:
            if mask[index] == 1:
                render_joints(image_data[image_iter], j, delay=True, use_depth=False, colour=(0,255,0))
            else:
                render_joints(image_data[image_iter], j, delay=True, use_depth=False, colour=(255,255,0))
        else:
            render_joints(image_data[image_iter], j, delay=True)
        #plot3D_joints(j, pixel = False)
        image_iter += 1

def run_ground_truths():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")
    #display_images_and_joints("MPI_pixels_omit.csv", "./Manually_Processed_Images/")

    #Split data by viewpoint
    #split_data_by_viewpoint("pixel_data_absolute.csv")
    
    #Create ground truths
    print("creating ground truths")
    create_ground_truths("./Images", 'pixel_data_absolute.csv')

    #Create "perfect" frame dataset
    #select_good_frames("../simple-HigherHRNet/EDA/Finished_Data/Images", 'pixel_data_absolute.csv')

    #Remove images marked for removal
    #images_path = "./Images"
    #run_image_deletion_mask(images_path, "pixel_data_absolute.csv", "image_deletion_mask.csv")