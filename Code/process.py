#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Data_Processing.Model_Based.HCF as hcf
import Programs.Machine_Learning.Model_Based.AutoEncoder.GAE as GAE
import Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj as Dataset_Obj
import Programs.Machine_Learning.Model_Based.GCN.Ground_Truths as GT
import Programs.Machine_Learning.Model_Based.GCN.Utilities as Graph_Utils
import Programs.Data_Processing.Model_Based.Render as Render
import torch
import torch.nn as nn
import torch_geometric
import tqdm
import torch_geometric.transforms as T
from Programs.Machine_Learning.Model_Based.GCN.Graph_Nets import GCN, GIN, GAT, train, accuracy, MultiInputGAT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    #Experimental creating hand crafted features
    #create_hcf_dataset("../EDA/Finished_Data/pixel_data_absolute.csv", "../EDA/Finished_Data/pixel_data_relative.csv", \
    #                    "../EDA/Finished_Data/pixel_velocity_absolute.csv", "../EDA/Finished_Data/Images")

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/Office_Dataset/Raw_Images", "./", exclude_2D=False, start_point=0)

    
    #Sort class labels (for 0ffice Images_chris this is 20-0, 20-1, 20-2)
    #abs_joint_data = Creator.assign_class_labels(num_switches=20, num_classes=2, 
    #                                joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data.csv",
    #                                joint_output="./Code/Datasets/Joint_Data/Office_Dataset/1_Absolute_Data(classes applied).csv")
    
    #Display first 2 instances of results 
    #print("\nStage 1: ")
    #render_joints_series("./Code/Datasets/Office_Dataset/Raw_Images", joints=abs_joint_data,
    #                     size = 20, delay=True, use_depth=True)

    #Remove empty frames
    abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/Office_Dataset/1_Absolute_Data(classes applied).csv",
                                                  image_file="./Code/Datasets/Office_Dataset/Raw_Images/",
                                                  joint_output="./Code/Datasets/Joint_Data/Office_Dataset/2_Absolute_Data(empty frames removed).csv",
                                                  image_output="./Code/Datasets/Office_Dataset/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("\nStage 2: ")
    #render_joints_series(image_data, abs_joint_data, size=15)
    #render_joints_series(image_data, abs_joint_data, size=15, plot_3D=True)
    
    #Trim start and end frames where joints get confused by image borders
    abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
                                                        joint_output="./Code/Datasets/Joint_Data/Office_Dataset/3_Absolute_Data(trimmed instances).csv",
                                                        image_output="./Code/Datasets/Office_Dataset/3_Trimmed Instances/", trim = 5)

    abs_norm_data = Creator.normalize_values(abs_joint_data, 
                                             joint_output="./Code/Datasets/Joint_Data/Office_Dataset/3_Absolute_Data(normed).csv")
    print("\nStage 3: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize outliers
    #abs_joint_data, image_data = Utilities.process_data_input("./Code/Datasets/Joint_Data/Office_Dataset/3_Absolute_Data(trimmed instances).csv",
     #                                                          "./Code/Datasets/Office_Dataset/3_Trimmed Instances/", ignore_depth=False)
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/Office_Dataset/4_Absolute_Data(normalized).csv")
    print("\nStage 4:")
    #render_joints_series(image_data, abs_joint_data, size=10)

    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/Office_Dataset/4.5_Absolute_Data(midhip).csv")
     
    print("\nStage 4.5:")
    #render_joints_series(image_data, abs_joint_data, size=10)
    #Change format of pre-scale to list of arrays instead of list of lists
    pre_scale = abs_joint_data

    #Normalize size (use absolute dataset)
    abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data,
                                               joint_output="./Code/Datasets/Joint_Data/Office_Dataset/5_Absolute_Data(scaled).csv")

    print("\nStage 5: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/6_Relative_Data(relative).csv")

    print("\nStage 6:")
    #render_joints_series("None", relative_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Flip joints so all facing one direction
    flipped_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/7_Relative_Data(flipped).csv") 

    print("\nStage 7:")
    #render_joints_series("None", flipped_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(pre_scale, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/8_Velocity_Data(velocity).csv")
    print("\nStage 8:")
    #render_velocity_series(abs_joint_data, velocity_data, image_data, size=20)

    flipped_velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/9_Velocity_Data(flipped).csv")  
    print("\nStage 9: ")
    #Create joint angles data
    joint_bones_data = Creator.create_bone_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/Office_Dataset/10_Bones_Data(integrated).csv")
    
    #render_velocity_series(abs_joint_data, joint_bones_data, image_data, size=20)

    print("\nStage 10:")
    #Create regions data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(abs_joint_data, #CHANGE BACK TO ABS_JOINT_DATA
                                                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/11_2_Region_Data",
                                                                                 images = image_data)
    regions_data = Creator.create_5_regions_dataset(abs_joint_data,
                                                    "./Code/Datasets/Joint_Data/Office_Dataset/12_5_Data_",
                                                      image_data)
    print("\nStage 11:", pre_scale[0])
    #Create HCF dataset
    hcf_data = Creator.create_hcf_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, image_data, 
                               joints_output="./Code/Datasets/Joint_Data/Office_Dataset/13_HCF_Data.csv")

    print("\nStage 12:")
    #Create ground truth comparison set
    print("data going into gait cycle extractor: ", pre_scale[0])
    ground_truths = Creator.create_ground_truth_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, hcf_data, image_data,
                                                        joints_output="./Code/Datasets/Joint_Data/Office_Dataset/14_Ground_Truth_Data.csv")
    
    print("\nStage 13:")
    #Combine datasets (relative, velocity, joint angles, regions)
    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/Office_Dataset/15_Combined_Data.csv")
    
    combined_norm_data = Creator.normalize_values(combined_data, 
                                             joint_output="./Code/Datasets/Joint_Data/Office_Dataset/15.5_Combined_Data(normed).csv")

    print("\nStage 14:")
    #Create regions data of combined data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(combined_data, #CHANGE BACK TO ABS_JOINT_DATA
                                                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region",
                                                                                 images = image_data)
    regions_data = Creator.create_5_regions_dataset(combined_data,
                                                    "./Code/Datasets/Joint_Data/Office_Dataset/17_Combined_Data_5Region",
                                                      image_data)
#########################################################################################################################

def process_autoencoder():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    #Upper and lower region dataset
    lower_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_bottom/', '16_Combined_Data_2Region_bottom.csv',
                                              joint_connections=Render.bottom_joint_connection).shuffle()
    dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/15.5_Combined_Data(normed)/', '15.5_Combined_Data(normed).csv',
                                       joint_connections=Render.joint_connections_m_hip).shuffle()
    upper_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_top/', '16_Combined_Data_2Region_top.csv',
                                             joint_connections=Render.top_joint_connections).shuffle()

    print(dataset[0])
    print(upper_dataset[0])
    print(lower_dataset[0])

    #train_loader, val_loader, test_loader, whole = GT.create_dataloaders(dataset)

    encoded_datasets = {'normal' : [dataset, "15.5_Combined_Data(normed)" ], 
                        'upper' : [upper_dataset, "16_Combined_Data_2Region_top"],
                        'lower' : [lower_dataset, "16_Combined_Data_2Region_bottom"]}
    
    skeleton_sizes = [17, 10, 6]

    #torch.manual_seed(0)
    num_epochs = 100
    #Upper Region Dataset
    
    for index, (key, value) in enumerate(encoded_datasets.items()):
        if index > -1:
            vae = GAE.VariationalAutoencoder(dim_in=value[0].num_node_features, dim_h=128, latent_dims=3)
            print("dim in: ", value[0].num_node_features)
            optim = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5) # originally 5
            vae.to(device)
            vae.eval()
            for epoch in range(num_epochs):
                train_loader, _, test_loader, whole = GT.create_dataloaders(value[0], batch_size=16)
                train_loss = GAE.train_epoch(vae,device,train_loader,optim)
                #Get embedding of entire dataset and save it
                val_loss = GAE.test_epoch(vae,device,whole, './Code/Datasets/Joint_Data/Office_Dataset/' + str(value[1]) + '/encoded/raw/encoded.csv', 
                                        skeleton_size = skeleton_sizes[index])
                print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))


    print("Autoencoding completing: Merging top and bottom datasets...")

    encoded_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/15.5_Combined_Data(normed)/encoded/', 'encoded.csv',
                                                joint_connections=Render.joint_connections_m_hip).shuffle()
    encoded_upper = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_top/encoded/', 'encoded.csv',
                                              joint_connections=Render.top_joint_connections).shuffle()
    encoded_lower = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_bottom/encoded/', 'encoded.csv',
                                              joint_connections=Render.bottom_joint_connection).shuffle()
    
    print("all encoded datasets loaded sucessfully. ")
    
    #Combine
    #Read both in as csv, append together, resacve, load result and joint dataset
    upper_array  = Utilities.load('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_top/encoded/raw/encoded.csv', metadata=True, 
                                  colnames = Utilities.colnames_top)
    lower_array = Utilities.load('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_bottom/encoded/raw/encoded.csv', metadata=True,
                                 colnames = Utilities.colnames_bottom)

    concatenated_regions = []
    for i, row in enumerate(upper_array):
        print("row: ", type(row), len(row), len(lower_array[i]), type(lower_array[i]))
        concat_row = list(row)
        for j, low_arr in enumerate(lower_array[i]):
            if j > 2:
                concat_row.append(low_arr)
        concatenated_regions.append(concat_row)
        print("lens: ", len(row), len(lower_array[i]), len(concatenated_regions[i]))
    
    Utilities.save_dataset(concatenated_regions, './Code/Datasets/Joint_Data/Office_Dataset/18_encoded_concat_2region/raw/encoded_concat_2region.csv')
    print("Concatenation sucessful...")


def run_multi_input_gat():

    datasets = []
    bottom_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_bottom', '16_Combined_Data_2Region_bottom.csv',
                                              joint_connections=Render.bottom_joint_connection, cycles=True).shuffle()
    
    top_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_top', '16_Combined_Data_2Region_top.csv',
                                            joint_connections=Render.top_joint_connections, cycles=True).shuffle()
    
    datasets.append(top_region)
    datasets.append(bottom_region)
    
    print("Creating model: ")
    gat_model = MultiInputGAT(dim_in=top_region.num_node_features, dim_h=16, dim_out=3)
    gat_model = gat_model.to("cuda")

    print("GAT MODEL") 
    #These regions will be the same for both datasets
    multi_input_train_val = []
    multi_input_test = []
    for dataset in datasets:
        multi_input_train_val.append(dataset[:int(len(dataset)*0.9)])
        multi_input_test.append(dataset[int(len(dataset)*0.9):])


    train_score, val_scores, test_scores = GAE.cross_valid(gat_model, multi_input_test, datasets=multi_input_train_val, k_fold=5, batch=16)

    print("final results: ")
    for ind, t in enumerate(test_scores):
        test_scores[ind] = test_scores[ind].cpu()
        test_scores[ind] = float(test_scores[ind])

    for i, score in enumerate(train_score):
        print("score {:.2f}: training: {:.2f}, validation: {:.2f}, test: {:.2f}".format(i, score, val_scores[i], test_scores[i]))

    mean, var = Utilities.mean_var(test_scores)
    print("mean, std and variance: {:.2f}%, {:.2f}% {:.5f}".format(mean, math.sqrt(var), var))


 
def run_gat():

    dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data_2Region_bottom', '16_Combined_Data_2Region_bottom.csv',
                                              joint_connections=Render.bottom_joint_connection, cycles=True)#.shuffle()
    
    #GT.assess_data(encoded_2region)
    print("concatenated dataset loaded sucessfully...")
    
    print("Creating model: ")
    gat_model = GAT(dim_in = dataset.num_node_features, dim_h=16, dim_out=3)
    gat_model = gat_model.to("cuda")

    print("GAT MODEL") 
    #These regions will be the same for both datasets
    train_val_dataset = dataset[:int(len(dataset)*0.9)]
    test_dataset  = dataset[int(len(dataset)*0.9):]


    train_score, val_scores, test_scores = GAE.cross_valid(gat_model, [test_dataset], datasets=[train_val_dataset], k_fold=5, batch=16)

    print("final results: ")
    for ind, t in enumerate(test_scores):
        test_scores[ind] = test_scores[ind].cpu()
        test_scores[ind] = float(test_scores[ind])

    for i, score in enumerate(train_score):
        print("score {:.2f}: training: {:.2f}, validation: {:.2f}, test: {:.2f}".format(i, score, val_scores[i], test_scores[i]))

    mean, var = Utilities.mean_var(test_scores)
    print("mean, std and variance: {:.2f}%, {:.2f}% {:.5f}".format(mean, math.sqrt(var), var))


if __name__ == '__main__':
    #Main menu
    #process_autoencoder()
    run_multi_input_gat()
    #run_gat()
    #main()