#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
from Programs.Data_Processing.Model_Based.HCF import create_hcf_dataset
from Programs.Data_Processing.Model_Based.Render import render_joints_series

#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Data_Processing.Model_Based.Data_Correction as Data_Correction
def main():

    #Experimental creating hand crafted features
    #create_hcf_dataset("../EDA/Finished_Data/pixel_data_absolute.csv", "../EDA/Finished_Data/pixel_data_relative.csv", \
    #                    "../EDA/Finished_Data/pixel_velocity_absolute.csv", "../EDA/Finished_Data/Images")

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/Office Images_Chris", "./Code/Datasets/Office_Dataset/", exclude_2D=False, start_point=0)

    
    #Sort class labels (for 0ffice Images_chris this is 20-0, 20-1, 20-2)
    '''abs_joint_data = assign_class_labels(num_switches=20, num_classes=2, 
                                    joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data.csv",
                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(1_classes applied).csv")
    
    #Display first 2 instances of results 
    print("Stage 1 checks: ")
    render_joints_series("./Code/Datasets/Office Images_Chris/Raw Images", joints=joint_data,
                         size = 100, delay=True, use_depth=True)
'''
    #Remove empty frames
    abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(1_classes applied).csv",
                                                  image_file="./Code/Datasets/Office Images_Chris/Raw Images/",
                                                  joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(2_empty frames removed).csv",
                                                  image_output="./Code/Datasets/Office Images_Chris/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("Stage 2 checks: ")
    #render_joints_series(image_data, abs_joint_data    , size=20)

    #Trim start and end frames where joints get confused by image borders
    abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
                                                        joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(3_trimmed instances).csv",
                                                        image_output="./Code/Datasets/Office Images_Chris/3_Trimmed Instances/", trim = 5)

    print("Stage 3 checks: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize outliers
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(4_normalized).csv")
    print("Stage 4 checks:")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize size (use absolute dataset)
    scaled_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data,
                                               joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data(5_scaled).csv")

    print("Stage 5 checks: ")
    #render_joints_series(image_data, scaled_joint_data, size=10)

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(scaled_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Relative_Data(6_relative).csv")
    print("Stage 6 checks:")
    #render_joints_series("None", relative_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Flip joints so all facing one direction
    print("lens before: ", len(relative_joint_data))
    flipped_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/Relative_Data(7_flipped).csv")     
    print("len after", len(flipped_joint_data))
    print("Stage 7 checks:")
    #render_joints_series("None", flipped_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/Velocity_Data(8_velocity).csv")

    print("Stage 8 checks")
    render_velocity_series(abs_joint_data, velocity_data, image_data, size=20)

    flipped_velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/Velocity_Data(9_flipped).csv")  
    
    #Create joint angles data
        #Stick in dataset_creator

    #Create regions data
        #Stick in dataset_creator

    #Create HCF dataset
        #Create ground truth
        #Stick in dataset_creator

    #Create ground truth comparison set
        #First generate ground truth average of certain HCF
        #Then for every instance calculate average difference

    #Combine datasets (relative, velocity, joint angles, regions)
        #Stick in dataset_creator       

#########################################################################################################################



if __name__ == '__main__':
    #Main menu
    main()