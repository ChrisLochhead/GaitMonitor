import sys, os

inletPath = os.path.dirname(__file__)
print("Initialising Packages")
sys.path.append(inletPath + "/Programs/Data_Collection/human_detector/JetsonYolo_Main")
sys.path.append(inletPath + "/Programs/Data_Collection/human_detector/JetsonYolo_Main/elements")
sys.path.append(inletPath + "/Programs/Data_Collection/human_detector/JetsonYolo_Main/models")
sys.path.append(inletPath + "/Programs/Data_Collection/human_detector/JetsonYolo_Main/utils")
sys.path.append(inletPath + "/Programs/Data_Collection/image_capture")
sys.path.append(inletPath + "/Programs/Data_Collection/image_processing")
sys.path.append(inletPath + "/Programs/Data_Collection/mask_network/Mask_RCNN/samples")
sys.path.append(inletPath + "/Programs/Data_Collection/Resnet")

#Needs newer version of tensorboard than PhD project environment can handle
sys.path.append(inletPath + "/Programs/Data_Analysis/DeepPrivacy")