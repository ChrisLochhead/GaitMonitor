import torch
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from torch_geometric.utils import degree
import re
import cv2
import csv

from Dataset_Obj import *
from Graph_Nets import GCN, GIN, GAT, train, accuracy
from Render import *
                    
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
        train_dataset = dataset[:int(len(dataset)*train)]
        val_dataset   = dataset[int(len(dataset)*train):int(len(dataset)*val)]
        test_dataset  = dataset[int(len(dataset)*test):]

        print(f'Training set   = {len(train_dataset)} graphs')
        print(f'Validation set = {len(val_dataset)} graphs')
        print(f'Test set       = {len(test_dataset)} graphs')

        # Create mini-batches
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, val_loader, test_loader

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
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

#Replace this and draw joints on frame with version found in demo for higher hr net
def ground_truth_event(event, x, y, flags, params):
        # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        row = params[0]
        estimate = params[1]
        good_frame = params[2]
        index = params[3]
        create_grounds = params[4]
        appended = params[5]
        # displaying the coordinates
        print(x, y)
        #Keep original depth value as dummy
        if index == 0:
            if appended == False:
                good_frame.append([1])
                params[5] = True
            else:
                good_frame[-1]= [1]
            print("this is a good frame")
            print("returning appended  = ", appended, params[5])
        if create_grounds == True:
            row.append([x, y, estimate[2]])

        return params[5]
    if event == cv2.EVENT_RBUTTONDOWN:
        good_frame = params[2]
        index = params[3]
        appended = params[5]
        if index == 0:
            if appended == False:
                good_frame.append([0])
                params[5] = True
            else:
                good_frame[-1] = [0]
            print("this is a bad frame")

            print("returning appended  = ", appended, params[5])
            return params[5]
        #else:
        #    quit()

def create_ground_truths(image_path, joints, save = True, create_grounds = False, start_point = 3):
    
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

                #Go through 18 points for every image
                
                for i in range (0, 18):
                    print("frame: ", file_iter, " of ", len(files))
                    print("total frame: ", joint_iter, "of ", len(joints), "corresponding: ", len(is_good_frame))
                    appended = False
                    while appended == False:
                        appended = cv2.setMouseCallback('image', ground_truth_event, [ground_truth_row, joints[joint_iter], is_good_frame, i, create_grounds, appended])
                        #Only iterate 18 times if actually setting ground truths
                        key = cv2.waitKey(0)  
                    if key == 113:
                        quit()
                    elif key == 115:
                        print("SAVING PROGRESS: ")
                        with open('image_deletion_mask.csv', 'a+', newline='') as f:
                            mywriter = csv.writer(f, delimiter = ",")
                            mywriter.writerows(is_good_frame)          

                    if create_grounds == False:
                        break

                    if i == 0 and is_good_frame[-1] == False:
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

            with open('image_deletion_mask.csv', 'a+', newline='') as f:
                mywriter = csv.writer(f, delimiter = ",")
                mywriter.writerows(is_good_frame)          

def run_image_deletion_mask(image_data, joint_data, mask):

    proc_images = []
    proc_joints = []
    for i, image in enumerate(image_data):
        if mask[i] == True:
            proc_images.append(image)
            proc_joints.append(joint_data[i])
    
    for i, row in enumerate(proc_joints):
        print("saving instance: ", str(float(row[0])))
        if row[1] < 10:
            file_no = str(0) + str(row[1])
        else:
            file_no = str(row[1])

        directory = "./Manually_Processed_Images/Instance_" + str(float(row[0]))
        print("i is: ", i , len(image_data), len(joint_data))
        os.makedirs(directory, exist_ok = True)
        cv2.imwrite(directory + "/" + file_no + ".jpg", image_data[i])
        
def run_ground_truths():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    #Split data by viewpoint
    split_data_by_viewpoint("pixel_data_absolute.csv")
    
    #Create ground truths
    create_ground_truths("../simple-HigherHRNet/EDA/Finished_Data/Images", 'pixel_data_absolute.csv')

    #Remove images marked for removal
    run_image_deletion_mask("../simple-HigherHRNet/EDA/Finished_Data/Images", "pixel_data_absolute.csv", "image_deletion_mask")

def run_GCN_training():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

   
    #Create dataset
    dataset = JointDataset('./', 'pixel_data_absolute.csv').shuffle()
    assess_data(dataset)

    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    #Define model
    print("Creating model: ")
    gin_model = GIN(dim_h=16, dataset=dataset)
    gcn_model = GIN(dim_h=16, dataset=dataset)
    gat_model = GIN(dim_h=16, dataset=dataset)

    #Train model
    #embeddings, losses, accuracies, outputs, hs = model.train(model, criterion, optimizer, data)
    print("GCN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gcn_model, train_loader, val_loader, test_loader)
    print("GAT MODEL") 
    model, embeddings, losses, accuracies, outputs, hs = train(gat_model, train_loader, val_loader, test_loader)
    print("GIN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gin_model, train_loader, val_loader, test_loader)

    # Train TSNE
    '''
    tsne = TSNE(n_components=2, learning_rate='auto',
            init='pca').fit_transform(embeddings[7].detach())

    # Plot TSNE
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=dataset[7].y)
    plt.show()
    '''

    #Animate results
    print("training complete, animating")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    run_3d_animation(fig, (embeddings, dataset, losses, accuracies, ax, train_loader))

if __name__ == "__main__":
    run_ground_truths()
