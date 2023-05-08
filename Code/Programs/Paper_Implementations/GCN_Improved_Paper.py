import torch
import matplotlib.pyplot as plt
import torch_geometric


#This is paper 1: 
    #Using Graph Convolutional Networks and Hand-crafted Features to Assess Health via Gait

#This method must:
    #Be self-supervised i.e. it can label the data itself
    #Detect and identify points of anomaly

#Plan for implementation:
    #Extract normal gait features in data
    #split gait into 5 regions (head + torso, left arm, right arm, left leg, right leg)
    #Pass these regions data into a variational autoencoder using a 2s methodology (passing both velocity and positional information in 3D)
    #Imbue with temportal info by adding the previous and next datapoint too
    #This will provide me very good hand crafted features that aren't domain specfic

    #After this, split gait into 2 regions (top and bottom) and pass these into autoencoders for a total of 4 encoders and 4 resulting vectors

    #Now I have a heirarchical pyramid of information at different levels of complexity. 

#Experiments:

    #Test the accuracy doing the following:

        #The baseline normal data (single stream) passed through a GAT  Result:
        #The baseline normal data (dual stream) passed through a GAT    Result: 

        #(Everything following this is dual stream)

        #The baseline, appended with the top and bottom info as passed through an autoencoder and then passed through an SVM    Result:
        #The baseline, appended with the 5 region implementation                                                                Result:
        #The baseline, appended with both the 5 region and the 2 region implementation                                          Result:
        #The baseline, appended with the 5 region, the 2 region and the hand-crafted domain specific features for gait          Result:

if __name__ == "__main__":
    pass
