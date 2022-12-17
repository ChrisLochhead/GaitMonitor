#Standard packages
import os, sys
import cv2
from pynput.keyboard import Key, Listener, KeyCode
import numpy as np
import matplotlib.pyplot as MPL

#Local files
import init_directories
import capture
import ImageProcessor
#from Utilities import remove_block_images, remove_background_images, generate_labels, unravel_FFGEI, create_HOGFFGEI, generate_instance_lengths, extract_ttest_metrics, create_contingency_table
#import GEI
#import LocalResnet
#import Experiment_Functions
#import File_Decimation
#import Ensemble

#Torch
import torch
from torchvision.transforms import ToTensor, Lambda

#Scipy for T-test
from scipy import stats

#MaskCNN only works on the PC version of this app, the Jetson Nano doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import maskcnn

#0 is main menu, 1 is second menu, 2 is verbosity selection menu
current_menu = 0
selected_function = None 

def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)
   
def run_camera(path="./Images/Instances/", v=0):
    #try:
    camera = capture.Camera()

    camera.run(path="./Images/CameraTest/", verbose=v)
    #except:
    #    main("No camera detected, returning to main menu")

def get_silhouettes(v =1):
    ImageProcessor.get_silhouettes('./Images/Instances', verbose=v)
    
def on_press(key):
    global current_menu
    global selected_function

    if hasattr(key, 'char'):
        if key.char == '1':
            if current_menu == 0:
                selected_function = run_camera
                verbosity_selection(max_verbose = 2)
            #elif current_menu == 1:
            #    extended_menu(3, page_3)
            elif current_menu == 2:
                selected_function(v=0)
                #main()
        if key.char == '2':
            if current_menu == 2:
                selected_function(v=1)
                #main()
        if key.char == '3':
            if current_menu == 2:
                main()
        if key.char == '9':
            if current_menu == 1:
                main()
            else:
                return False

#Verbosity selection for camera and image processing functions
def verbosity_selection(max_verbose = 1):
    clear_console()
    global current_menu
    current_menu = 2
    print("current menu", current_menu)
    print("Choose Verbosity")
    print("Select one of the following options:\n")
    for i in range(0, max_verbose):
        print(str(i+1) + ". " + str(i))
    print(str(max_verbose + 1) + ". Back")

def extended_menu(index, content):
    global current_menu
    current_menu = index
    clear_console()
    print("More")
    print(content)


page_0 = """Welcome

Select one of the following options:
         
REGULAR FUNCTIONS
         
1. Activate Camera
9. Quit"""


def main(error_message = None, init = False):
    global current_menu
    current_menu = 0
    if error_message:
        print(error_message)

    print(page_0)

    if init == True:
        with Listener(on_press=on_press) as listener:
            try:
                listener.join()
            except:
                print("program ended, listener closing")

if __name__ == '__main__':
    #Main menu
    main(init = True)
