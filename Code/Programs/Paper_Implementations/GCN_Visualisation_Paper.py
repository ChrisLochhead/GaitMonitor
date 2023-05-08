import torch
import matplotlib.pyplot as plt
import torch_geometric


#This is paper 3: 
    #Title: An investigation into explainable AI techniques for effective communication of medical data.

    #Defence: Use findings of research in developing the UI explaining findings of main system to turn into a paper about design informatics.    
    #Evaluate or justify using opinions of people in PPI groups.

#This method must:
    #Produce a sheet of visual demonstrations of Analysed gait that is legible to a non-domain expert.


#Plan for implementation:
    #Construct the various charts for degradation and produce in matplot or seaborn
        #Draw gait skeleton, colour coded bones and joints to indicate biggest section contributions to abnormality
        #Draw comparison (if applicable) to previous assessment to indicate positive/negative changes.
        #3D chart visualisation of health summary vs previous data points (at least one being doctor-observed ground truth taken at start of study)
        #Heatmap of points in gait cycle where gait is particularly abnormal
        #Timemap of examples with time where gait is more abnormal
        #Chart and extract hand crafted features onto a table, colour coded with positive or negative changes (decreased step length, etc)
        #Chart knee-joint angle changes 
        #Include some kind of lay-summary at the bottom in text indicating all of the involved findings. 

    #Print onto some kind of singular sheet and save

#Experiments:

    #Present demo's to spoken to doctors, physios and also lay-people.
    #Transcribe opinions on results
    #Link to initial feedback
    

if __name__ == "__main__":
    pass
