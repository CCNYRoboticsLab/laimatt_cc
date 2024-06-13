import subprocess
import glob
import shutil
import csv
import json
import re
import laspy

def get_center_point_x(las_file_path):
    with laspy.open(las_file_path) as f:
        x_min, x_max = f.header.x_min, f.header.x_max
        center_x = (x_min + x_max) / 2
        return center_x
    
def get_center_point_y(las_file_path):
    with laspy.open(las_file_path) as f:
        y_min, y_max = f.header.y_min, f.header.y_max
        center_y = (y_min + y_max) / 2
        return center_y
    
def get_center_point_z(las_file_path):
    with laspy.open(las_file_path) as f:
        z_min, z_max = f.header.z_min, f.header.z_max
        center_z = (z_min + z_max) / 2
        return center_z

def extract_index(string):
    # Define a regular expression pattern to match the desired substring
    pattern = r'COMPONENT_(.*?)\.las'
    # Use re.search() to find the first occurrence of the pattern in the string
    match = re.search(pattern, string)
    # If a match is found, extract the matched substring
    if match:
        substring = match.group(0)

        return int(substring[10:-4])
    else:
        return -1

csvoutput = open('data.csv', 'w', newline='')
writer = csv.writer(csvoutput)
writer.writerow(['Crack_ID', 'x', 'y', 'z', 'original file'])

command = [
    "CloudCompare",
    "-SILENT",
    "-O",
    "-GLOBAL_SHIFT",
    "AUTO",
    "/home/roboticslab/Developer/laimatt_api/3sections - 170 - 253.las",
    "-C_EXPORT_FMT",
    "LAS",
    "-EXTRACT_CC",
    "5",
    "20",
]

subprocess.run(command, shell=True)

numFiles = 0
destination = 'output'

for filepath in glob.iglob('3sections - 170 - 253_COMPONENT_*'):
    numFiles += 1
    shutil.move(filepath, destination)


list = [0]*numFiles

for filepath in glob.iglob('output/*'):
    index = extract_index(filepath) - 1
    list[index] = filepath

for count, path in enumerate(list, start=1):
    x = get_center_point_x(path)
    y = get_center_point_y(path)
    z = get_center_point_z(path)
    writer.writerow([count, x, y, z, path])
