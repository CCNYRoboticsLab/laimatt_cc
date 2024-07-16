from flask import Flask, send_file
from filter import getName, TypeColor
import pandas as pd
import subprocess
import csv
import laspy
import glob
import mysql.connector
import os
import shutil
import numpy
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

def find_nearest_neighbors(points, source_point):
    # Initialize the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
    
    distances, indices = nbrs.kneighbors(points)
    
    print(points)
    print(indices)
    print(distances)
    
    nearest_path = [source_point]
    current_point = source_point

    # Set to track visited points
    visited = set()
    visited.add(tuple(current_point))

    for x in range(len(points) - 1):
        # Find the nearest neighbor
        distances, indices = nbrs.kneighbors([current_point])
        next_closest = 1
        next_index = indices[0][next_closest]
        print(indices)
        print("next " + str(next_index))
        
        # If the point is already visited, continue searching
        while tuple(points[next_index]) in visited:
            # Get the next nearest neighbor
            next_index = indices[0][(next_closest) % len(points)]
            next_closest += 1
            # print("next " + str(next_index) + " has been visited")

        next_point = points[next_index]
        nearest_path.append(next_point)
        visited.add(tuple(next_point))
        current_point = next_point

    return nearest_path

def csv_to_asc(csv_file, asc_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, header=None)

    # Extract the first three rows (x, y, z)
    np_coords = numpy.round((pd.DataFrame(df.iloc[1:, :3]).to_numpy(dtype=numpy.float64)), 3)
    nearest_neighbors_path = find_nearest_neighbors(np_coords, np_coords[0])
    print(nearest_neighbors_path)

    # Open the ASC file for writing
    with open(asc_file, 'w') as f:
        for index in nearest_neighbors_path:
            # Join x, y, z with a space and write to the ASC file
            f.write(f"{index[0]} {index[1]} {index[2]}\n")


def csvToLas(test_dir, test_index, length):
    path_las = test_dir + "/component_las_" + test_index
    os.makedirs(path_las)
    
    path_poly = test_dir + "/component_poly_" + test_index
    os.makedirs(path_poly)
        
    for x in range(length):
        csv_file = test_dir + "/component_csv/component_" + f"{x:06d}" + ".csv"
        las_file = path_las + "/component_" + f"{x:06d}" + ".las"
        asc_file = path_poly + "/component_" + f"{x:06d}" + ".poly"
        
        # pdal = "/home/roboticslab/Developer/laimatt/laimatt_pdal/.conda/bin/pdal"
        pdal = "pdal"
        command = [
            pdal,
            "translate",
            csv_file,
            las_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        csv_to_asc(csv_file, asc_file)

def lasToCsv(test_dir, min_p, tolerance, max_p, file_name):
    # pdal = "/home/roboticslab/Developer/laimatt/laimatt_pdal/.conda/bin/pdal"
    pdal = "pdal"
    command = [
        pdal,
        "translate",
        file_name,
        test_dir + "/full_segmented.csv",
        "-f",
        "filters.cluster",
        "--filters.cluster.min_points=" + str(min_p),
        "--filters.cluster.tolerance=" + str(tolerance),
        "--filters.cluster.max_points=" + str(max_p) 
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    
    csv_file = test_dir + "/full_segmented.csv"

    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by="ClusterID", ascending=True)
    df_sorted.to_csv(csv_file, index=False)

def create_csvs(test_dir):
    index = 0
    csv_file = test_dir + "/full_segmented.csv"
    folder_path = test_dir + "/component_csv"
    os.makedirs(folder_path)
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        last_col = len(header) - 1
        previous_value = 0

        csvoutput = open(folder_path + '/component_000000.csv', 'w', newline='')
        writer = csv.writer(csvoutput)
        writer.writerow(header)

        for row in reader:
        # Get the value from the specified column
            current_value = int(float(row[last_col]))   

            # If the value is different from the previous value, start a new row
            if current_value != previous_value:
                csvoutput.close()
                csvoutput = open(folder_path + '/component_' + f"{current_value:06d}" + '.csv', 'w', newline='')
                writer = csv.writer(csvoutput)
                writer.writerow(header)
                
                index += 1
                previous_value = current_value
            else:
                # Otherwise, add the row to the current row
                writer.writerow(row)
        csvoutput.close()
    return index

    
def bounding_box_info(las_file_path):
    with laspy.open(las_file_path) as f:
        x_min, x_max = f.header.x_min, f.header.x_max        
        y_min, y_max = f.header.y_min, f.header.y_max
        z_min, z_max = f.header.z_min, f.header.z_max
        
        center_x = round(((x_min + x_max) / 2), 3)
        center_y = round(((y_min + y_max) / 2), 3)
        center_z = round(((z_min + z_max) / 2 ), 3)
        length = round((x_max - x_min), 3)
        width = round((y_max - y_min), 3)
        height = round((z_max - z_min), 3)
        
        return [center_x, center_y, center_z, length, width, height]
    
def populate_db(test_dir, test_index, uid, project_id, task_id, color):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",  # Your MySQL username
        password="",  # Your MySQL password (if any)
        port=80,  # Your MySQL port
        unix_socket="/app/mysql.sock"
    )
    cursor = mydb.cursor()
    cursor.execute("USE sample")
    
    filepaths = sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))
    # next(iter(sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))))
    # filepaths = next(iter(sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))))
    
    for filepath in filepaths[1:]:
        b = bounding_box_info(filepath)
        link = "https://laimatt.boshang.online/download/" + str(project_id) + "/" + task_id + "/" + os.path.basename(filepath)
        
        query = "INSERT INTO patch_crack (center_lat, center_long, center_alt, box_length, box_width, box_height, type, las_link, whole_data_id) " + \
            "VALUES ('%s', '%s', '%s', '%s', '%s', '%s', %s, %s, %s)"
        data = (b[0], b[1], b[2], b[3], b[4], b[5], color, link, uid)
        # print(query, data)
        cursor.execute(query, data)
        mydb.commit()

    mydb.close()    
    
def populate_csv(test_dir, test_index):
    csvoutput = open(test_dir + '/component_data.csv', 'w', newline='')
    writer = csv.writer(csvoutput)
    writer.writerow(['x', 'y', 'z', 'length', 'width', 'height', 'type', 'original file'])
    
    for filepath in sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*')):
        box_info = bounding_box_info(filepath)
        writer.writerow(box_info + ['crack', filepath])
    
    csvoutput.close()
        
def create_components(project_id, task_id, uid, color): 
    min_p = 4
    tolerance = .65
    max_p = 10000
    
    folder_path = 'tasks/task_{}_{}/'.format(project_id, task_id)
    test_path = os.path.join(folder_path, "tests")
    # file_name = os.path.join(folder_path, '{}_filtered_model.las'.format(getName(TypeColor, color)))
    file_name = "3sections - 170 - 253.las"

    
    if not (os.path.exists(test_path)):
        os.makedirs(os.path.join(test_path))

    test_dir = os.path.join(test_path, ('{}_test_{}_{}_{}').format(getName(TypeColor, color), min_p, tolerance, max_p))
    # test_dir = os.path.join(test_path, ("test_" + str(min_p) + "_" + str(tolerance) + "_" + str(max_p)))
    test_index = str(min_p) + "_" + str(tolerance) + "_" + str(max_p)
    if os.path.exists(test_dir):
        print(test_dir + " already exists, remaking", flush=True)
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    lasToCsv(test_dir, min_p, tolerance, max_p, file_name)
    index = create_csvs(test_dir)
    csvToLas(test_dir, test_index, index + 1)

    # populate_db(test_dir, test_index, uid, project_id, task_id, color)
    populate_csv(test_dir, test_index)
    return "success"

@app.route('/components', methods=['GET'])
def components_api():
    project_id = 151
    task_id = "c9c7deff-e46b-4ed5-8316-79ddf9d19352"
    uid = 12
    color = 2
            
    processed_data = create_components(project_id, task_id, uid, color)

    return processed_data

@app.route('/download/<project_id>/<task_id>/<filename>', methods=['GET'])
def download(project_id, task_id, filename):
    # Assuming files are stored in a directory named 'files' under the app root directory
    task = os.path.join(app.root_path, 'task_{}_{}'.format(project_id, task_id))
    
    uploads = os.path.join(task, 'tests/test_10_0.2_10000/component_las_10_0.2_10000')

    # Use send_file function to send the file
    return send_file(os.path.join(uploads, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2001, debug=True)