from glob import glob
from pickle import TRUE
import pyrealsense2 as rs
import os
import time
import cv2
import numpy as np
import JetsonYolo
from PIL import Image
from pynput.keyboard import Key, Listener, KeyCode
import os
import copy
import File_Decimation
import datetime
import csv

break_program = False
pause_program = False 
save_instrinsics = True

#Colour tags for console
class c_colours:
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'

def on_press(key):
    global break_program
    global pause_program
    if hasattr(key, 'char'):
        if key.char == 'q':
            pause_program = False
            break_program = True
        if key.char == 'p':
            pause_program = True
            break_program = True
        
class Camera:

    def __init__(self, depth = True):
        global break_program
        break_program = False

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.file_count = 0
        self.file_limit = 3000
        config = rs.config()
        if depth:
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15) # original 640 by 480

        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        # Start streaming
        print("where the camera starts")
        profile = self.pipeline.start(config)
        self.dec_filter = rs.decimation_filter()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align = rs.align(rs.stream.color)

    def retrieve_image(self):
        global save_instrinsics
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
    
        # Align the depth frame to color frame
        aligned_frames =  self.align.process(frames)
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if save_instrinsics == True:
            processed_depth_frame = self.dec_filter.process(depth_frame)
            depth_profile = processed_depth_frame.get_profile()
            video_profile = depth_profile.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            print("these are the intrinsics: ", intr)

            intrinsics_array = [intr.width, intr.height, intr.ppx, intr.ppy, intr.fx, intr.fy, intr.model, intr.coeffs] 
            #Save intrinsics to a .csv file
            with open("depth_intrinsics.csv","w+", newline='') as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerow(intrinsics_array)
            
            save_instrinsics = False

        # Convert images to numpy arrays
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        return np.hstack((color_img, depth_colormap)), color_img, depth_colormap#, color_img

    def retrieve_color_image(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        # Convert images to numpy arrays
        color_img = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        return color_img

    def run(self, path = "./capture1/", verbose = 1, depth = True):

        i = 0
        human_detected_count = 0
        human_stationary = False
        image_buffer = []

        #Make storage for image captures
        print("Making it in here?")
        os.makedirs(path, exist_ok=True)
        s0 = time.time()
        record_timer = 0.0
        seen_human = False
        seen_human_previous = False
        objs_last_frame = 0
        local_path = ""
        current_image_array = []
        #delay_timer = 0.0
        frames_since_last = 0

        global break_program
        global pause_program

        with Listener(on_press=on_press) as listener:
            while break_program == False:
                
                #Check if time is appropriate for monitoring
                now = datetime.datetime.now()
                morning_limit = now.replace(hour=8, minute=0, second=0, microsecond=0)
                evening_limit = now.replace(hour=22, minute=37, second=0, microsecond=0)
                if now < morning_limit or now > evening_limit:
                    pause_program = True
                    break_program = True

                #Check if data overloaded and if so conduct a purge
                if self.file_count >= self.file_limit:
                    #Purge and upload data
                    #print("purging data: ", self.file_count)
                    File_Decimation.decimate()
                    self.file_count = 0

                #Record if previous frame seen a human
                seen_human_previous = seen_human

                # Wait for a coherent pair of frames: depth and color
                if depth:
                    color_img, col_only, dep_only = self.retrieve_image()
                else:
                    color_img = self.retrieve_color_image()

                # Plot detected objects
                if depth:
                    refined_img = Image.fromarray(col_only)
                else:
                    refined_img = Image.fromarray(color_img)
                    
                refined_img = refined_img.resize((480, 480))


                #Only scan for humans every 3 frames, or 5 seconds after a human was detected.
                if verbose > 0:
                    if i%3 == 0 and record_timer == 0.0 or time.time() - record_timer >= 10.0:
                        
                        print(c_colours.GREEN + "Searching for Humans")
                        #Get humans
                        objs = JetsonYolo.get_objs_from_frame(np.asarray(refined_img), False)
                        seen_human = False

                        #If a human found in the frame, notify of the detection and start the record timer
                        if len(objs) > 0:
                            print(c_colours.CYAN + "human detected: ", objs_last_frame)
                            if objs_last_frame == 0:
                                seen_human = True
                                record_timer = time.time()

                            #Set that a human was found this frame
                            objs_last_frame = 1
                            frames_since_last = 0
                      
                        #give a 20 frame delay before declaring no humans detected to account for temporary blips 
                        if len(objs) == 0:
                            #print("no human detected")
                            frames_since_last += 1
                            #If it's been more than 20 frames, it's not a blip, set human detection to 0
                            if frames_since_last > 5:
                                print( c_colours.RED + "pretending cant see: ", frames_since_last)
                                objs_last_frame = 0
                            else:
                                print("setting objs last frame to 1", frames_since_last)
                                objs_last_frame = 1

                        #Recheck for existence of human after blips
                        if objs_last_frame == 0:
                            if len(objs) > 0:
                                seen_human = True
                                record_timer = time.time()
                                frames_since_last = 0

                        #Debug
                        debug_img, not_used = JetsonYolo.plot_obj_bounds(objs, np.asarray(refined_img))

                refined_img = np.asarray(debug_img)

                i += 1
                #Print FPS
                if i%10 == 0 and verbose > 0:
                    st = time.time()
                    #print('FPS: ' + str(i/(st - s0)))
                #Show images
                if verbose > 0:
                    cv2.imshow('RealSense', refined_img)
                    cv2.waitKey(1)

                if seen_human:
                    if seen_human_previous == False:
                        #Create a new local path so each instance has it's own folder
                        path_created = False
                        n = 0.0
                        while path_created == False:
                            try:
                                os.mkdir(path + "/Instance_" + str(n) + "/")
                                local_path = path + "/Instance_" + str(n) + "/"
                                path_created = True
                            except:
                                n+=1
                        #Save image, add buffer to array to be saved
                        im_name = str(int(time.time() * 1000)) + '.jpg'
                        if depth:
                            cv2.imwrite(local_path + im_name, col_only)
                            depim_name = 'dep-' + im_name
                            cv2.imwrite(local_path + depim_name, dep_only)
                        else:
                            cv2.imwrite(local_path + im_name, refined_img)


                        current_image_array = copy.deepcopy(image_buffer)
                    else:
                        #Save images
                        im_name = str(int(time.time() * 1000)) + '.jpg'
                            
                        #Save depth image
                        if depth:
                            cv2.imwrite(local_path + im_name, col_only)
                            depim_name = 'dep-' + im_name
                            cv2.imwrite(local_path + depim_name, dep_only)
                        else:
                            cv2.imwrite(local_path + im_name, refined_img)
                else:
                    # Add pre-detection buffer to catch the start of any movement incase it is missed by the detector
                    #Only record to the buffer when we cant see a person
                    #Reset timer to check if human is still in frame
                    if time.time() - record_timer >= 5.0:
                        record_timer = 0.0

                    #Save the buffer images to the instance
                    if len(current_image_array) > 0:
                        record_timer = 0.0
                        for image_data in current_image_array:
                            if verbose > 0:
                                print(c_colours.RED + "dumping buffer")
                            if depth:
                                cv2.imwrite(local_path + image_data[2], image_data[0])
                                depim_name = 'dep-' + image_data[2]
                                cv2.imwrite(local_path + depim_name, image_data[1])
                            else:
                                cv2.imwrite(local_path + image_data[1], image_data[0])

                        self.file_count += 1

                        #clearing buffer
                        current_image_array.clear()
                    if depth:
                        image_buffer.append((col_only, dep_only, '0_buffer_image_{:.3f}'.format(time.time()) + '.jpg'))
                    else:
                        image_buffer.append((refined_img, '0_buffer_image_{:.3f}'.format(time.time()) + '.jpg'))
                    #Keep the buffer at 5 frames long
                    if len(image_buffer) > 5:
                         image_buffer.pop(0)

            try:
                cv2.destroyAllWindows()
                return pause_program
            except:
                print("program ended, listener closing")
                return pause_program
            finally:
                return pause_program
  
