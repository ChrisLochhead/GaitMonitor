import torch
import matplotlib.pyplot as plt
import torch_geometric


#This is paper 2: 
    #Title: Evaluating and generating graph data for gait assessment using GANs.

#This method must:
    #Implement GAN's for graph data instead of image data
    #Be able to evaluate the quality of graphs in a weakly-supervised manner
    #Assert which parts of the graph contribute mostly to the abnormality in order to detect outliers
    #Generate effective new data to solve the sparse data problem in the domain

#Plan for implementation:
    #Get Score frames into dataset 

    #Implement standard GAN
    #Modify this GAN to take graph data instead of image data
    #Visualize the results this spits out firstly of new good/bad examples
    #Process "Good example" ground truth for a single instance and pass it through autoencoders to chart it visually in 3D
    #Plot new examples of good and bad in the same way and see if visualisation works nicely. 

#Experiments:

    #Test regular dataset using best implementation in paper 1 (baseline)
    #Test with GAN images added to the dataset (various percentage size increases)
    #Evaluate gait quality predictions from GAN vs classification with GCN and compare how accurate it is/
    

if __name__ == "__main__":
    pass
