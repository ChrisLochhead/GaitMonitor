'''
This file contains all of the functions for rendering joints and images and other effects.
'''
#imports
import cv2
import copy
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Arc
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

#dependencies
import Programs.Data_Processing.Utilities as Utilities

'''
The various different joint graph connection configurations
'''
                    #Bottom dataset
joint_connections = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 0], [12, 0], # hips to origin #total of 7 including mid-hip

                    #Top dataset
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [8, 6], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin = total of 11

joint_connections_no_head_m_hip = [[10, 8], [8, 6], # left foot to hip 
                     [11, 9], [9, 7], # right foot to hip
                     [6, 12], [7, 12], # hips to origin #total of 7 including mid-hip

                    #Top dataset
                     [4, 2], [2, 0], # left hand to shoulder
                     [5, 3], [3, 1], #right hand to shoulder
                     [0, 12], [1, 12]] # Shoulders to origin (origin is midhip)

joint_connections_m_hip = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 17], [12, 17], [17,0],# hips to origin #total of 7 including mid-hip

                    #Top dataset
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [8, 6], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin = total of 11

joint_connections_n_head = [[11, 9], [9, 7], # left foot to hip 
                     [12, 10], [10, 8], # right foot to hip
                     [8, 13], [9, 13], [13,0],# hips to origin #total of 7 including mid-hip

                    #Top dataset
                     [5, 3], [3, 1], # left hand to shoulder
                     [6, 4], [4, 2], #right hand to shoulder
                     [1, 0], [2, 0]]# Shoulders to origin (head avg)

bottom_joint_connection = [[5, 3], [3, 1], # left foot to hip 
                     [6, 4], [4, 2], # right foot to hip
                     [1, 0], [2, 0]] # hips to origin 

top_joint_connections = [[9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [6, 8], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin = total of 11

head_joint_connections = [[1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin = 5

limb_connections = [[0, 1], [1, 2]] # From the extremity to the base (i.e the foot - hip or hand - shoulder)


def chart_knee_data(gait_cycles, display = False):
    '''
    This generates a plot charting knee angle data over a gait cycle.

    Arguments
    ---------
    gait_cycles: List(List())
        List of joints segmented by gait cycle
    display: bool (optional, default = False)
        indicates whether to display the chart

    Returns
    -------
    List()
        List of co-efficients to describe the chart
    '''
    hcf_coefficients = []
    for i in range(len(gait_cycles[0])):
        l_x = gait_cycles[0][i]
        l_y = [i for i in range(len(l_x))]
        r_x = gait_cycles[1][i]
        r_y = [i for i in range(len(r_x))]

        #Potentially add interpolation code here to give more examples for a smoother chart
        l_x, l_y = Utilities.interpolate_knee_data(l_x, l_y)
        r_x, r_y = Utilities.interpolate_knee_data(r_x, r_y)
        poly = np.polyfit(l_y,l_x,6)
        poly_alt = np.polyfit(r_y, r_x, 6)
        poly_l = np.poly1d(poly)(l_y)
        poly_r = np.poly1d(poly_alt)(r_y)

        if display:
            print("showing original")
            plt.figure()
            plt.plot(l_y,l_x)
            plt.plot(r_y,r_x)
            plt.show()

            print("showing poly", poly)
            plt.figure()
            plt.plot(l_y,poly_l)
            plt.plot(r_y,poly_r)
            plt.show()
        hcf_coefficients.append(np.concatenate((poly, poly_alt)))
    return hcf_coefficients


def filter_coords(joints, index, metadata = 5):
    '''
    Utility function for removing meta data from all the frames and segmenting into streams by x, y and z co-ordinate

    Arguments
    ---------
    joints: List(List())
        All of the joints in the file
    index: int
        Indicates which of x, y or z to process
    metadata: int (optional, default = 5)
        Indicates the amount of metadata to expect per-frame

    Returns
    -------
    List()
        Returns a list of co-ords at the index
    '''
    coords = []
    for i, j in enumerate(joints):
        if i >= metadata:
            coords.append(j[index])
    return coords

def plot3D_joints(joints, pixel = True, metadata = 5, x_rot = 90, y_rot = 180):
    '''
    This renders the joints as a 3D plot.

    Arguments
    ---------
    joints: List()
        The joints to be rendered
    pixel: bool (optional, default = True)
        Indicates whether the data is pixel or cm-based
    metadata: int (optional, default = 5)
        Indicates the amount of metadata to expect per-frame
    x_rot: int (optional, default = 90)
        The x-rotation of the chart
    y_rot: int (optional, default = 180)
        The y-rotation of the chart

    Returns
    -------
    None
    '''
    # generate data
    x = filter_coords(joints, 0, metadata=metadata)
    y = filter_coords(joints, 1, metadata=metadata)
    z = filter_coords(joints, 2, metadata=metadata)

    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    #424 by 240
    if pixel:
        ax.set_xlim([0, 424])
        ax.set_ylim([0, 240])
        ax.set_zlim3d([0, 255])
    else:
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim3d([-200, 200])

    if pixel:
        plt.gca().invert_yaxis()
    plt.gca().invert_zaxis()

    ax.view_init(x_rot, y_rot)
    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    #plot connections
    for connection in joint_connections:
        ax.plot([joints[connection[0] + metadata][0], joints[connection[1] + metadata][0]], \
                [joints[connection[0] + metadata][1], joints[connection[1] + metadata][1]], \
                [joints[connection[0] + metadata][2], joints[connection[1] + metadata][2]], \
                    color = 'g')
    # plot points
    sc = ax.scatter(x, y, z, s=40, c=x, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # legend
    #-90, 180, 0 angle, azimuth and roll
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show(block=True)

def render_joints_series(image_source, joints, size, delay = True, use_depth = True, plot_3D = False, x_rot = 90, y_rot= 180, background = False):
    '''
    This iterates over a series of images/joints and shows them in sequence

    Arguments
    ---------
    image_source: str
        root image folder 
    joints: List(List) or str
        location of or list of joints to be shown
    size: int
        number of instances from the frame to be shown
    delay: bool (optional, default = True)
        indicates whether to make each frame appear by a spacebar click or to run automatically
    use_depth: bool (optional, default = True)
        indicates whether to colour the joints according to their depth (2D version only)
    plot_3D: bool (optional, default = False)
        indicates whether to render as a 3D object instead of 2D.
    x_rot: int (optional, default = 90)
        The x-rotation of the chart
    y_rot: int (optional, default = 180)
        The y-rotation of the chart
    background: bool (optional, default = False)
        indicates whether to draw on a black frame or the real image (2D version only)

    Returns
    -------
    None
    '''
    joints, images = Utilities.process_data_input(joints, image_source)
    for i in range(size):
        if plot_3D:
            plot3D_joints(joints[i], x_rot=x_rot, y_rot=y_rot)
        else:
            if background:
                render_joints(images[0], joints[i], delay, use_depth)
            else:
                render_joints(images[i], joints[i], delay, use_depth)       
            cv2.destroyWindow("Joint Utilities Image")

def render_joints(image, joints, delay = False, use_depth = True, metadata = 6, colour = (0, 255, 0)):
    '''
    This renders a single frame with the joints overlaid.

    Arguments
    ---------
    image: List()
        current image
    joints: List(List)
        list of joints to be shown
    delay: bool (optional, default = True)
        indicates whether to make each frame appear by a spacebar click or to run automatically
    use_depth: bool (optional, default = True)
        indicates whether to colour the joints according to their depth (2D version only)
    metadata: int (optional, default = 6)
        indicates the expected amount of metadata per-frame
    colour: tuple (optional, default = (0, 255, 0))
        indicates the colour of the joints

    Returns
    -------
    None
    '''
    tmp_image = copy.deepcopy(image)
    tmp_image = draw_joints_on_frame(tmp_image, joints, use_depth_as_colour=use_depth, metadata = metadata, colour=colour)
    cv2.imshow('Joint Utilities Image',tmp_image)
    cv2.setMouseCallback('Joint Utilities Image', click_event, tmp_image)
    if delay:
        cv2.waitKey(0) & 0xff

def render_velocity_series(joint_data, velocity_data, image_data, size):
    '''
    Simple looping function to render a series of velocity images

    Arguments
    ---------
    joint_data: List(List())
        joint data to be rendered
    velocity_data: List(List())
        corresponding velocity vectors
    image_data: List(List())
        corresponding source images
    size: int
        Number of frames in the sequence to render

    Returns
    -------
    None
    '''
    for i in range(size):
        render_velocities(joint_data[i], velocity_data[i], image_data[i])

def render_velocities(joint_data, velocity_data, image_data, delay = True, metadata = 6):
    '''
    Render an image with joints and velocity vectors

    Arguments
    ---------
    joint_data: List(List())
        joint data to be rendered
    velocity_data: List(List())
        corresponding velocity vectors
    image_data: List(List())
        corresponding source images
    delay: bool (optional, default = True)
        indicates whether to use keyboard control for the display
    metadata: int (optional, default = 6)
        amount of metadata to expect per-frame

    Returns
    -------
    None
    '''
    for i, coord in enumerate(joint_data):
        if i >= metadata:
            image_direction = [int((velocity_data[i][1] * 40) + coord[1]),
                                 int((velocity_data[i][0] * 40) + coord[0])]

            image = cv2.arrowedLine(image_data, [int(coord[1]), int(coord[0])] , image_direction,
                                            (0,255,0), 1) 
    cv2.imshow('Joint Utilities Image',image)
    cv2.setMouseCallback('Joint Utilities Image', click_event, image)
    if delay:
        cv2.waitKey(0) & 0xff

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        quit()

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False, metadata = 6, colour = (0, 150, 200), check_leg = False, aux_joints = None):
    '''
    Function to overlay the joints onto the corresponding frame

    Arguments
    ---------
    frame: List()
        source frame
    joints: List()
        single graph skeleton to be drawn on the frame
    use_depth_as_colour: bool (optional, default = False)
        indicates how to colour the joints
    metadata: int (optional, default = 6)
        amount of metadata to expect per-frame
    colour: tuple (optional, default = (0, 255, 0))
        indicates the colour of the joints
    check_leg: bool (optional, default = False)
        indicates whether to colour the left leg to track which one the model thinks is the left
    aux_joints: List() (optional, default = None)
        contains any additional joints for debugging 
        
    Returns
    -------
    List()
        Returns the frame with the joints drawn on
    '''
    tmp_frame = copy.deepcopy(frame)
    tmp_joints = copy.deepcopy(joints)
    connections = joint_connections

    #Top region
    if len(tmp_joints) == 17:
        connections = top_joint_connections 
    #Bottom region
    elif len(tmp_joints) == 13:
        connections = bottom_joint_connection
    #Head region
    elif len(tmp_joints) == 11:
        connections = head_joint_connections
    #limbs
    elif len(tmp_joints) == 9:
        connections = limb_connections
    #Otherwise its the normal dataset with a mid-hip appended
    elif len(tmp_joints) == 24:
        connections = joint_connections_m_hip

    for joint_pair in connections:
            #Draw links between joints
            tmp_a = tmp_joints[joint_pair[1] + metadata]
            tmp_b = tmp_joints[joint_pair[0] + metadata]
            start = [int(float(tmp_a[1])), int(float(tmp_a[0]))]
            end = [int(float(tmp_b[1])), int(float(tmp_b[0]))]

            cv2.line(tmp_frame, start, end, color = (255,255,255), thickness = 2) 

    for i, joint in enumerate(tmp_joints):

        if isinstance(joint, int):
            continue
        #0 is X, Y is 1, 2 is confidence.

        #Clamp joints within frame size
        
        if joint[0] >= 240:
            joint[0] = 239
        if joint[1] >= 424:
            joint[1] = 423

        #Check for auxillary joints:
        if aux_joints != None:
            tmp_frame = cv2.circle(tmp_frame, (int(float(aux_joints[i][1])),int(float(aux_joints[i][0]))), radius=1, color=(0,0,255), thickness=10)    
        
        if i == 17 and check_leg == True or i == 19 and check_leg == True or i == 21 and check_leg == True:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(255,255,255), thickness=4)
        elif use_depth_as_colour == False:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(0,255,255), thickness=6)
        else:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(150, 100, joint[2]), thickness=4)
      
    
    return tmp_frame
