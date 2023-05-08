#Standard packages
import os, sys
import cv2
from pynput.keyboard import Key, Listener, KeyCode
import numpy as np
import matplotlib.pyplot as MPL
import datetime
import torch

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
import re

#Torch
import torch
from torchvision.transforms import ToTensor, Lambda

#Scipy for T-test
from scipy import stats

#MaskCNN only works on the PC version of this app, the Jetson Nano doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import maskcnn

#Colour tags for console
class c_colours:
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'

#0 is main menu, 1 is second menu, 2 is verbosity selection menu
current_menu = 0
selected_function = None 
capture_paused = False
restart_time = None

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def reorder_folders(path):
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        for i, dir in enumerate(dirs):
            print("Dir: ",i,  dir)
            print("renaming: ", os.path.join(subdir, dir))
            os.rename(os.path.join(subdir, dir), os.path.join(subdir, str("Instance_" + str(i) + ".0")))
        break

def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)
   
def run_camera(path="./Images/Instances/", v=0):
    #try:
    camera = capture.Camera()

    out_condition = camera.run(path="./Images/", verbose=v)

    return out_condition
    #except:
    #    main("No camera detected, returning to main menu")

def get_silhouettes(v =1):
    ImageProcessor.get_silhouettes('./Images/Instances', verbose=v)
    
def reset_capture_timer():
    print("empty")
    now = datetime.datetime.now()
    later = now + datetime.timedelta(seconds=5)
    return later
    print("now: ", now)
    print("later: ", later)
    #morning_limit = now.replace(hour=8, minute=0, second=0, microsecond=0)
    #evening_limit = now.replace(hour=22, minute=37, second=0, microsecond=0)

def on_press(key):
    global current_menu
    global selected_function
    global capture_paused
    global restart_time

    if hasattr(key, 'char'):
        if key.char == '1':
            if current_menu == 0:
                selected_function = 0
                verbosity_selection(max_verbose = 2)

            elif current_menu == 2:
                if selected_function == 0:
                    capture_paused = run_camera(v=0)

                if capture_paused:
                    restart_time = reset_capture_timer()
                main()

        if key.char == '2':
            if current_menu == 2:
                if selected_function == 0:
                    capture_paused = run_camera(v=1)
                    if capture_paused:
                        restart_time = reset_capture_timer()
                main()
            elif current_menu == 0:
                print("reordering folders")
                reorder_folders("./Images")
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
    print(c_colours.BLUE + "current menu", current_menu)
    print(c_colours.BLUE + "Choose Verbosity")
    print(c_colours.BLUE + "Select one of the following options:\n")
    for i in range(0, max_verbose):
        print(str(i+1) + ". " + str(i))
    print(str(max_verbose + 1) + ". Back")

def extended_menu(index, content):
    global current_menu
    current_menu = index
    clear_console()
    print("More")
    print(content)


page_0 = c_colours.BLUE + """Welcome

Select one of the following options:
         
REGULAR FUNCTIONS
         
1. Activate Camera
2. Reorder Folders
9. Quit"""


def main(error_message = None, init = False, repeat_loop = True):
    global current_menu
    global restart_time
    global capture_paused
    main_loop = True

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

    #Loop to check if the capture is delayed
    if repeat_loop == True:
        while main_loop == True:

            #Exit from the loop if this isnt a capture delay
            if capture_paused == False:
                main_loop = False
                main(repeat_loop=False)

            if restart_time != None:
                #Re-run capture once delay is completed.
                if restart_time < datetime.datetime.now():
                    restart_time = None
                    main_loop = False
                    out_condition = run_camera(v=1)
                    capture_paused = out_condition

                    #Reset delay
                    if capture_paused:
                        restart_time = reset_capture_timer()
                    main()



if __name__ == '__main__':
    #Main menu
    print("version: ", torch.__version__)
    main(init = True)
