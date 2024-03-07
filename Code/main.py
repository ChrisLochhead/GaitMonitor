'''
This is the file controlling the recording system, complete with a minimal GUI
'''
#imports
import os
from pynput.keyboard import Listener
import datetime
import torch
#dependencies
from Programs.Data_Processing.Utilities import numericalSort

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

def reorder_folders(path):
    '''
    re-orders manually converted folders to make sure they all go from 0-59

    Arguments
    ---------
    path: str
        root folder for recorded instances
        
    Returns
    -------
    None
    '''
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        for i, dir in enumerate(dirs):
            os.rename(os.path.join(subdir, dir), os.path.join(subdir, str("Instance_" + str(i) + ".0")))
        

#simple clear console command
def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)
     
def reset_capture_timer():
    '''
    Controls the automatic recording timer

    Returns
    -------
    bool
        Indicates whether it's too late to be recording
    '''
    now = datetime.datetime.now()
    later = now + datetime.timedelta(seconds=5)
    return later

def on_press(key):
    '''
    Controls on press argument, essentially controlling the GUI keystrokes

    Arguments
    ---------
    key: Object
        the pressed key
        
    Returns
    -------
    bool
        Only returns false when ending the program
    '''
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
                    pass
                    #capture_paused = run_camera(v=0)

                if capture_paused:
                    restart_time = reset_capture_timer()
                main()

        if key.char == '2':
            if current_menu == 2:
                if selected_function == 0:
                    pass
                    #capture_paused = run_camera(v=1)
                    if capture_paused:
                        restart_time = reset_capture_timer()
                main()
            elif current_menu == 0:
                print("reordering folders")
                reorder_folders("./Code/Datasets/Individuals/Full_Dataset/Ahmed")
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
    '''
    Verbosity selection menu

    Arguments
    ---------
    max_verbose: int (optional, default = 1)
        
    Returns
    -------
    None
    '''
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
    '''
    Opens additional menus

    Arguments
    ---------
    index: int
        indicates which submenu to open
    content: str
        prints corresponding content
        
    Returns
    -------
    None
    '''
    global current_menu
    current_menu = index
    clear_console()
    print(content)


page_0 = c_colours.BLUE + """Welcome

Select one of the following options:
         
REGULAR FUNCTIONS
         
1. Activate Camera
2. Reorder Folders
9. Quit"""


def main(error_message = None, init = False, repeat_loop = True):
    '''
    main loop for the GUI menu

    Arguments
    ---------
    error_message: str (optional, default = None)
        string denoting any potential errors
    init: bool (optional, default = False)
        denotes whether this is the initial call
    repeat_loop: bool (optional, default = True)
        controls repeating
        
    Returns
    -------
    None
    '''
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
                    #out_condition = run_camera(v=1)
                    #capture_paused = out_condition

                    #Reset delay
                    if capture_paused:
                        restart_time = reset_capture_timer()
                    main()



if __name__ == '__main__':
    main(init = True)
