from Utilities import create_flipped_joint_dataset, load, load_images
import Dataset_Creator
def process_dataset():
    #load in raw images
    #Process into absolute joint dataset
    #Produce relative joint dataset from them

    #Process relative joint dataset with normalizing functions etc, need to find all transforms
    #normalize

    #Produce velocity dataset
    #Produce regions dataset (2 and 5)
    #Create joint angles

    #look for other processes needed by paper 1 implementation plan


    pass

def test_flip():

    abs_data = load('../../../Datasets/Joint_Data/MPI_pixels_omit.csv')
    rel_data = load('../../../Datasets/Joint_Data/MPI/Relative_Data.csv')
    images = load_images('../../../Datasets/Cleaned Home Images_Chris')
    #Dataset_Creator.create_relative_dataset(abs_data)
    create_flipped_joint_dataset(rel_data, abs_data, images, save = False)

if __name__ == '__main__':
    #Main menu
    print("processing dataset")
    test_flip()
