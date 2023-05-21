import cv2
import copy
import re, seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
#%%capture
from IPython.display import display, HTML
from matplotlib import animation
plt.rcParams["animation.bitrate"] = 3000
plt.rcParams['animation.ffmpeg_path'] = "C:/Users/Chris/Desktop/ffmpeg-5.1.2-full_build/bin/ffmpeg.exe"

import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import math

from Programs.Data_Processing.Model_Based.Dataset_Obj import get_COO_matrix
import Programs.Data_Processing.Model_Based.Utilities as Utilities

                    #Bottom dataset
joint_connections = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 0], [12, 0], # hips to origin #total of 7 including mid-hip

                    #Top dataset
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [6, 8], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0]]# Eyes to origin = total of 11

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


def get_angle_plot(line1, line2, offset = 1, color = None, origin = [0,0], len_x_axis = 1, len_y_axis = 1):

    # Angle between line1 and x-axis
    slope1 = (line1[1][1] - line1[0][1]) / float(line1[1][0] - line1[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle


    # Angle between line2 and x-axis
    slope2 = (line2[1][1] - line2[0][1]) / float(line2[1][0] - line2[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color='r', label = str(angle)+u"\u00b0"), angle

def chart_knee_data(gait_cycle_angles):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)


    l_x = gait_cycle_angles[0]
    l_y = [i for i in range(len(l_x))]

    r_x = gait_cycle_angles[1]
    r_y = [i for i in range(len(r_x))]

    print("length x: ", len(l_x), len(l_y), l_y)

    #Potentially add interpolation code here to give more examples for a smoother chart
    l_x, l_y = Utilities.interpolate_knee_data(l_x, l_y)
    r_x, r_y = Utilities.interpolate_knee_data(r_x, r_y)

    #print("showing original")
    #plt.figure()
    #plt.plot(l_y,l_x)
    #plt.plot(r_y,r_x)
    #plt.show()


    print("showing poly")
    plt.figure()
    poly = np.polyfit(l_y,l_x,8)
    poly_alt = np.polyfit(r_y, r_x, 8)
    poly_l = np.poly1d(poly)(l_y)
    poly_r = np.poly1d(poly_alt)(r_y)
    plt.plot(l_y,poly_l)
    plt.plot(r_y,poly_r)
    plt.show()


def plot_graph(data):
    G = process_data_to_graph(data, get_COO_matrix())
    print("nodes: ", G.nodes(G))
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=800)
    plt.show()

def animate(i, *fargs):
    data = fargs[0]
    outputs = fargs[1]
    losses = fargs[2]
    accuracies = fargs[3] 

    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color="blue",#outputs[i],
                    cmap="hsv",
                    vmin=-2,
                    vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14
                    )
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=20)


def animate_alt(i, *fargs):
    embeddings = fargs[0]
    data = fargs[1]
    losses = fargs[2]
    accuracies = fargs[3]
    ax = fargs[4]
    train_loader = fargs[5]

    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()

    cols = []
    for j, point in enumerate(train_loader):
        for k, em in enumerate(point):
            if k == 2: 
                class_vals = em[1].numpy()
                for val in class_vals:
                    col = "blue"
                    if val == 1:
                        col = "red"
                    elif val == 2:
                        col = "green"
                    cols.append(col)
        
        break

    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=cols, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

def run_3d_animation(fig, fargs):
        plt.axis('off')
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        anim = animation.FuncAnimation(fig, animate_alt, \
                                    np.arange(0, 200, 10), interval=800, repeat=True, fargs=fargs)
        html = HTML(anim.to_html5_video())

        plt.show()
        display(html)
            
def process_data_to_graph(row, coo_matrix):
    G = nx.Graph()

    #Add nodes
    for i, x in enumerate(row.x.numpy()):
        G.add_node(int(i), pos=(-x[1], x[0]))
        #Break to avoid reading edge indices
        #break
    
    #Add edges
    for connection in joint_connections:
        G.add_edge(connection[0], connection[1])

    return G

def filter_coords(joints, index, metadata = 3):
    coords = []
    for i, j in enumerate(joints):
        if i >= metadata:
            coords.append(j[index])
    
    return coords

def plot3D_joints(joints, pixel = True, metadata = 3, x_rot = 90, y_rot = 180, angles = None):
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
    #z forward -90, 180
    #on the corner -175, 120
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

    for a in angles:
        ax.add_patch(a)

    # legend
    #-90, 180, 0 angle, azimuth and roll
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    #plt.draw()#block=True)
    plt.show(block=True)
    #plt.waitforbuttonpress(0) # this will wait for indefinite time
    #plt.close(fig)
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight')

def render_joints_series(image_source, joints, size, delay = True, use_depth = True, plot_3D = False, x_rot = 90, y_rot= 180):
    
    joints, images = Utilities.process_data_input(joints, image_source)
        
    for i in range(size):
        if plot_3D:
            plot3D_joints(joints[i], x_rot=x_rot, y_rot=y_rot)
        else:
            render_joints(images[i], joints[i], delay, use_depth)
            cv2.destroyWindow("Joint Utilities Image")

def render_joints(image, joints, delay = False, use_depth = True, metadata = 3, colour = (255, 0, 0)):
    tmp_image = copy.deepcopy(image)
    tmp_image = draw_joints_on_frame(tmp_image, joints, use_depth_as_colour=use_depth, metadata = metadata, colour=colour)
    cv2.imshow('Joint Utilities Image',tmp_image)

    cv2.setMouseCallback('Joint Utilities Image', click_event, tmp_image)
    if delay:
        cv2.waitKey(0) & 0xff

def render_velocity_series(joint_data, velocity_data, image_data, size):
    for i in range(size):
        render_velocities(joint_data[i], velocity_data[i], image_data[i])

def render_velocities(joint_data, velocity_data, image_data, delay = True, metadata = 3):
    for i, coord in enumerate(joint_data):
        if i >= metadata:
            #x = int((velocity_data[1] * 40) + coord[1])
            #y = int((velocity_data[0] * 40) + coord[0])
            #print("velocity data: ", velocity_data)
            image_direction = [int((velocity_data[i][1] * 40) + coord[1]),
                                 int((velocity_data[i][0] * 40) + coord[0])]

            image = cv2.arrowedLine(image_data, [int(coord[1]), int(coord[0])] , image_direction,
                                            (0,255,0), 4) 

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

def draw_joints_on_frame(frame, joints, use_depth_as_colour = False, metadata = 3, colour = (0, 0, 255)):

    tmp_frame = copy.deepcopy(frame)
    tmp_joints = copy.deepcopy(joints)
    connections = joint_connections
    print("length of joints: ", len(tmp_joints))

    #Top region
    if len(tmp_joints) == 14:
        connections = top_joint_connections 
    #Bottom region
    elif len(tmp_joints) == 10:
        connections = bottom_joint_connection
    #Head region
    elif len(tmp_joints) == 8:
        connections = head_joint_connections
    elif len(tmp_joints) == 6:
        connections = limb_connections

    for joint_pair in connections:
            #Draw links between joints
            print("joint pair: ", joint_pair, len(tmp_joints), len(connections))
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