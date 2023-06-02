#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Data_Processing.Model_Based.HCF as hcf
import Programs.Machine_Learning.Model_Based.AutoEncoder.GAE as GAE
import Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj as Dataset_Obj
import Programs.Machine_Learning.Model_Based.GCN.Ground_Truths as GT
import Programs.Machine_Learning.Model_Based.GCN.Utilities as Graph_Utils
import torch
import torch.nn as nn
import torch_geometric
import tqdm
import torch_geometric.transforms as T
from Programs.Machine_Learning.Model_Based.GCN.Graph_Nets import GCN, GIN, GAT, train, accuracy
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

   
    #Create dataset
    dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/', '15.5_Combined_Data(normed).csv').shuffle()
    #GT.assess_data(dataset)

    train_loader, val_loader, test_loader, whole = GT.create_dataloaders(dataset)
    
    
    torch.manual_seed(0)
    d = 3
    vae = GAE.VariationalAutoencoder(dim_in=dataset.num_node_features, dim_h=64, latent_dims=d)
    lr = 1e-3 #originally 3
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-7) # originally 5
    print(f'Selected device: {device}')
    vae.to(device)
    vae.eval()
    num_epochs = 10


    for epoch in range(num_epochs):
        train_loss = GAE.train_epoch(vae,device,train_loader,optim)
        #Get embedding of entire dataset and save it
        val_loss = GAE.test_epoch(vae,device,whole, './Code/Datasets/Joint_Data/Office_Dataset/raw/encoded_data.csv')
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

    encoded_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/Office_Dataset/', 'encoded_data.csv').shuffle()
    train_loader, val_loader, test_loader, _ = GT.create_dataloaders(encoded_dataset, batch_size=64)

    #Define model GCNs
    print("Creating model: ")
    #gcn_model = GCN(dim_in = dataset.num_node_features, dim_h=16, dim_out=3)
    #gcn_model = gcn_model.to("cuda")

    gat_model = GAT(dim_in = encoded_dataset.num_node_features, dim_h=16, dim_out=3)
    gat_model = gat_model.to("cuda")
    #gin_model = GIN(dim_h=16, dataset=dataset)
    #Train model
    #embeddings, losses, accuracies, outputs, hs = model.train(model, criterion, optimizer, data)
    #print("GCN MODEL")
    #model, embeddings, losses, accuracies, outputs, hs = train(gcn_model, train_loader, val_loader, test_loader)
    print("GAT MODEL") 
    model, embeddings, losses, accuracies, outputs, hs = train(gat_model, train_loader, val_loader, test_loader)
    #print("GIN MODEL")
    #model, embeddings, losses, accuracies, outputs, hs = train(gin_model, train_loader, val_loader, test_loader)

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


if __name__ == '__main__':
    #Main menu
    process_autoencoder()
    #main()