#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
from Programs.Data_Processing.Model_Based.HCF import create_hcf_dataset
from Programs.Data_Processing.Model_Based.Render import *
from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
from Programs.Data_Processing.Model_Based.Dataset_Creator import process_empty_frames, assign_class_labels
def main():
    #Process data from EDA into perfect form
    #correct_joints_data("./Images", "./EDA/gait_dataset_pixels.csv", save=True, pixels=True)
    #Test:
    #print("Correction processing sucessfully completed, testing resulting images and joints...")
    #load_and_overlay_joints(directory="./EDA/Finished_Data/Images/", joint_file="./EDA/Finished_Data/pixel_data_absolute.csv", ignore_depth=False, plot_3D=True)


    #Experimental creating hand crafted features
    #create_hcf_dataset("../EDA/Finished_Data/pixel_data_absolute.csv", "../EDA/Finished_Data/pixel_data_relative.csv", \
    #                    "../EDA/Finished_Data/pixel_velocity_absolute.csv", "../EDA/Finished_Data/Images")


    #Draw calculated velocities
    #run_velocity_debugger("./EDA/Finished_Data/Images/", "./EDA/Finished_Data/pixel_data_relative.csv", save= True, debug=False)

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/Office Images_Chris", "./Code/Datasets/Office_Dataset/", exclude_2D=False, start_point=0)

    
    #Sort class labels (for 0ffice Images_chris this is 20-0, 20-1, 20-2)
    joint_data = assign_class_labels(num_switches=20, num_classes=2, 
                                    joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data.csv",
                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(1_classes applied).csv")
    #Remove empty frames
    joint_data, image_data = process_empty_frames(joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(1_classes applied).csv",
                                                  image_file="./Code/Datasets/Office Images_Chris/Raw Images/",
                                                  joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(2_empty frames removed).csv",
                                                  image_output="./Code/Datasets/Office Images_Chris/2_Empty Frames Removed/")


    #Normalize outliers
        #Function implemented in correct_joints_data()
        #Stick in dataset_creator

    #Normalize size (use absolute dataset)
        #joint_data = load("../EDA/Finished_Data/MPI_pixels_omit.csv")
        #image_data = load_images("../EDA/Finished_Data/Images/")
        #normalize_joint_scales(joint_data, image_data)
        #Stick in dataset_creator

    #Create relative dataset
        #Stick in dataset_creator

    #Create velocity dataset
        #Stick in dataset_creator

    #Create joint angles data
        #Stick in dataset_creator

    #Create regions data
        #Stick in dataset_creator

    #Create HCF dataset
        #Create ground truth
        #Stick in dataset_creator

    #Combine datasets (relative, velocity, joint angles, regions)
        #Stick in dataset_creator       

#########################################################################################################################



if __name__ == '__main__':
    #Main menu
    main()