import cv2
import copy
import re, seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from scripts.DatasetBuilder.utilities import joint_connections 

def filter_coords(joints, index, metadata = 3):
    coords = []
    for i, j in enumerate(joints):
        if i >= metadata:
            coords.append(j[index])
    
    return coords

def plot3D_joints(joints, pixel = True, metadata = 3):
    # generate data
    x = filter_coords(joints, 0)
    y = filter_coords(joints, 1)
    z = filter_coords(joints, 2)

    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    #424 by 240
    if pixel:
        ax.set_xlim([0, 240])
        ax.set_ylim([0, 424])
        ax.set_zlim3d([0, 255])
    else:
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim3d([-200, 200])

    if pixel:
        plt.gca().invert_yaxis()
    plt.gca().invert_zaxis()
    #z forward -90, 180
    #on the corner -175, 120
    ax.view_init(90, 180)

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
    #plt.draw()#block=True)
    plt.show(block=True)
    #plt.waitforbuttonpress(0) # this will wait for indefinite time
    #plt.close(fig)
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight')

def render_joints(image, joints, delay = False, use_depth = True, metadata = 3, colour = (255, 0, 0)):
    tmp_image = copy.deepcopy(image)
    tmp_image = draw_joints_on_frame(tmp_image, joints, use_depth_as_colour=use_depth, metadata = metadata, colour=colour)
    cv2.imshow('Joint Utilities Image',tmp_image)

    cv2.setMouseCallback('Joint Utilities Image', click_event, tmp_image)
    if delay:
        cv2.waitKey(0) & 0xff

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        quit()

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False, metadata = 3, colour = (0, 0, 255)):

    tmp_frame = copy.deepcopy(frame)
    tmp_joints = copy.deepcopy(joints)

    for joint_pair in joint_connections:
            #Draw links between joints
            tmp_a = tmp_joints[joint_pair[1] + metadata]
            tmp_b = tmp_joints[joint_pair[0] + metadata]
            start = [int(float(tmp_a[1])), int(float(tmp_a[0]))]
            end = [int(float(tmp_b[1])), int(float(tmp_b[0]))]

            cv2.line(tmp_frame, start, end, color = (0,255,0), thickness = 2) 


    for i, joint in enumerate(tmp_joints):

        if isinstance(joint, int):
            continue
        #0 is X, Y is 1, 2 is confidence.

        #Clamp joints within frame size
        
        if joint[0] >= 240:
            joint[0] = 239
        if joint[1] >= 424:
            joint[1] = 423
        
        if use_depth_as_colour == False:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=colour, thickness=4)
        else:
            tmp_frame = cv2.circle(tmp_frame, (int(float(joint[1])),int(float(joint[0]))), radius=1, color=(150, 100, joint[2]), thickness=4)


    return tmp_frame