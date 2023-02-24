from deep_privacy import cli
import os
import re

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def anonymize_images(input_folder = 'Images\\CameraTest', output_folder = 'Images\\Anonymized'):

    # get current directory
    path = os.getcwd()
    print("Current Directory", path)
    
    # prints parent directory
    parent = os.path.abspath(os.path.join(path, os.pardir))
    grandparent = os.path.abspath(os.path.join(parent, os.pardir))

    print("Current Directory", grandparent + "/" + input_folder)
    output_folder = grandparent + "\\" + output_folder
    input_folder = grandparent + "\\" + input_folder
       
    #Cycle through all images and subdirectories and do it manually
    for (subdir, dirs, files) in os.walk(input_folder):
        dirs.sort(key=numericalSort)
        for file_iter, file in enumerate(sorted(files, key = numericalSort)):
            tmp = [input_folder, subdir, file]
            #output_subfolder = subdir.split('\\')[-1]
            #if os.path.exists(output_folder + "\\" + output_subfolder) == False:
            #    print("Trying to make: ", output_folder + "\\" + output_subfolder)
            #    os.makedirs(output_folder + "\\" + output_subfolder, exist_ok=True)
            #    output_folder = output_folder + "\\" + output_subfolder
            print("processing file: ", os.path.join(*tmp), "outputting to : ", output_folder)
            cli.main(os.path.join(*tmp), output_folder)


    #cli.main(input_folder, output_folder)

if __name__ == "__main__":
    print("this is main")
    anonymize_images()
    #cli.main()