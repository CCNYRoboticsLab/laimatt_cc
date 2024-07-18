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


class PointCloudPostProcessor:
    
    def __init__(self, training):
        self.training = training

    def lasToCsv(self, test_dir, min_p, tolerance, max_p, file_name):
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

    def create_csvs(self, test_dir):
        csv_file = test_dir + "/full_segmented.csv"
        
        if not self.training:
            folder_path = test_dir + "/component_csv"
            os.makedirs(folder_path)
        
        
        with open(csv_file, 'r') as file:
            
            df = pd.read_csv(csv_file, header=None)
            all_coords = numpy.round((pd.DataFrame(df.iloc[1:, :3]).to_numpy(dtype=numpy.float64)), 3)
            cluster_list = []
            prev_cluster = 0
            current_cluster = 0
            
            reader = csv.reader(file)
            header = next(reader)
            last_col = len(header) - 1
            previous_value = 0

            if not self.training:
                csvoutput = open(folder_path + '/component_000000.csv', 'w', newline='')
                writer = csv.writer(csvoutput)
                writer.writerow(header)
            

            for index, row in enumerate(reader):
            # Get the value from the specified column
                current_value = int(float(row[last_col])) 

                # If the value is different from the previous value, start a new row
                if current_value != previous_value:
                    
                    current_cluster = index
                    cluster_list.append(all_coords[prev_cluster:current_cluster])
                    # print(all_coords[prev_cluster:current_cluster])
                    prev_cluster = index + 1
                    
                    if not self.training:
                        csvoutput.close()
                        csvoutput = open(folder_path + '/component_' + f"{current_value:06d}" + '.csv', 'w', newline='')
                        writer = csv.writer(csvoutput)
                        writer.writerow(header)
                    
                    previous_value = current_value
                else:
                    # Otherwise, add the row to the current row
                    if not self.training: writer.writerow(row)
            
            current_cluster = index
            cluster_list.append(all_coords[prev_cluster:current_cluster])
            
            if not self.training: csvoutput.close()
            
        print(cluster_list)
        return cluster_list
    
    def csvToLas(self, test_dir, test_index, clusters):
        csv_file, las_file, asc_file = None, None, None
        if not self.training:
            path_las = test_dir + "/component_las_" + test_index
            path_poly = test_dir + "/component_poly_" + test_index
            os.makedirs(path_las)
            os.makedirs(path_poly)
        
            for x in range(len(clusters)):
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
                self.csv_to_asc(csv_file, las_file, asc_file, clusters, x)
            
                
            
            # if x == 1: csv_to_asc(csv_file, las_file, asc_file)
        
        line_list = []
        for index in range(len(clusters)):
            line_list.append(self.csv_to_asc(csv_file, las_file, asc_file, clusters, index))
        return line_list
    
    def csv_to_asc(self, csv_file, las_file, asc_file, clusters, index):
        
        if not self.training:
            df = pd.read_csv(csv_file, header=None)
            np_coords = numpy.round((pd.DataFrame(df.iloc[1:, :3]).to_numpy(dtype=numpy.float64)), 3)
        else:
            np_coords = clusters[index]
        # print("points:")
        # print(clusters[index])
        # print("normal points:")
        # print(np_coords)
        # print()
        
        source = self.nearestToCorner(self.box_corners(las_file, clusters[index]), np_coords)
        
        # print(source)
        # print(np_coords[source])
        
        nearest_neighbors_path = self.find_nearest_neighbors(np_coords, np_coords[source])
        print(nearest_neighbors_path)

        if not self.training:
            # Open the ASC file for writing
            with open(asc_file, 'w') as f:
                for index in nearest_neighbors_path:
                    # Join x, y, z with a space and write to the ASC file
                    f.write(f"{index[0]} {index[1]} {index[2]}\n")
                    
        path = numpy.empty
        for index in nearest_neighbors_path:
            print(numpy.array(index))
            numpy.append(path, numpy.array(index))
        return path
                
    def nearestToCorner(self, corners, points):
        ptsWcorners = numpy.concatenate((corners, points), axis = 0)
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(ptsWcorners)
        distances, indices = nbrs.kneighbors(corners)
        boxpoints = numpy.concatenate((indices, distances), axis = 1)
        min_point = int(numpy.squeeze(numpy.where(boxpoints[:8, 3] == min(boxpoints[:8, 3]))[0]))
        
        # print(boxpoints)
        # print(boxpoints[:8, 3])
        # print(boxpoints[min_point][1])
        
        if boxpoints[min_point][1] < 8:
            return int(boxpoints[min_point][0]) - 8
        
        return int(boxpoints[min_point][1]) - 8

    def box_corners(self, las_file, points):
        if not self.training:
            b = self.bounding_box_info(las_file)
        else:
            b = self.bounding_box_info_training(points)
        
        center = [b[0], b[1], b[2]]
        
        # Calculate half dimensions
        half_length = b[3] / 2
        half_width = b[4] / 2
        half_height = b[5] / 2

        # Create corners based on center
        corners = numpy.array([
            [center[0] - half_length, center[1] - half_width, center[2] - half_height],
            [center[0] + half_length, center[1] - half_width, center[2] - half_height],
            [center[0] + half_length, center[1] + half_width, center[2] - half_height],
            [center[0] - half_length, center[1] + half_width, center[2] - half_height],
            [center[0] - half_length, center[1] - half_width, center[2] + half_height],
            [center[0] + half_length, center[1] - half_width, center[2] + half_height],
            [center[0] + half_length, center[1] + half_width, center[2] + half_height],
            [center[0] - half_length, center[1] + half_width, center[2] + half_height],
        ])

        return numpy.round(corners, 3)
    
    def find_nearest_neighbors(self, points, source_point):
        # Initialize the nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
        
        distances, indices = nbrs.kneighbors(points)
        
        # print(points)
        # print(indices)
        # print(distances)
        
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
            # print(indices)
            # print("next " + str(next_index))
            
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
        
    def bounding_box_info(self, las_file_path):
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
            
            print([center_x, center_y, center_z, length, width, height])
            
            return [center_x, center_y, center_z, length, width, height]
    
    def bounding_box_info_training(self, points):
        # Assuming points is a NumPy array where each row represents [x, y, z] coordinates
        x_min, y_min, z_min = numpy.min(points, axis=0)
        x_max, y_max, z_max = numpy.max(points, axis=0) 
        
        center_x = round(((x_min + x_max) / 2), 3)
        center_y = round(((y_min + y_max) / 2), 3)
        center_z = round(((z_min + z_max) / 2), 3)
        length = round((x_max - x_min), 3)
        width = round((y_max - y_min), 3)
        height = round((z_max - z_min), 3)
        
        print([center_x, center_y, center_z, length, width, height])
        
        return [center_x, center_y, center_z, length, width, height]
        
    def populate_db(self, test_dir, test_index, uid, project_id, task_id, color):
        if self.training:
            return
        
        # mydb = mysql.connector.connect(
        #     host="localhost",
        #     user="root",  # Your MySQL username
        #     password="",  # Your MySQL password (if any)
        #     port=80,  # Your MySQL port
        #     unix_socket="/app/mysql.sock"
        # )
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",  # Your MySQL username
            password="",  # Your MySQL password (if any)
            port=3308,  # Your MySQL port
            unix_socket="/opt/lampp/var/mysql/mysql.sock"
        )
        cursor = mydb.cursor()
        cursor.execute("USE sample")
        
        filepaths = sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))
        # next(iter(sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))))
        # filepaths = next(iter(sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*'))))
        
        for filepath in filepaths[1:]:
            b = self.bounding_box_info(filepath)
            link = "https://laimatt.boshang.online/download/" + str(project_id) + "/" + task_id + "/" + os.path.basename(filepath)
            
            query = "INSERT INTO patch_crack (center_lat, center_long, center_alt, box_length, box_width, box_height, type, las_link, whole_data_id) " + \
                "VALUES ('%s', '%s', '%s', '%s', '%s', '%s', %s, %s, %s)"
            data = (b[0], b[1], b[2], b[3], b[4], b[5], color, link, uid)
            # print(query, data)
            cursor.execute(query, data)
            mydb.commit()

        mydb.close()    
        
    def populate_csv(self, test_dir, test_index):
        csvoutput = open(test_dir + '/component_data.csv', 'w', newline='')
        writer = csv.writer(csvoutput)
        writer.writerow(['x', 'y', 'z', 'length', 'width', 'height', 'type', 'original file'])
        
        for filepath in sorted(glob.iglob(test_dir + '/component_las_' + test_index + '/*')):
            box_info = self.bounding_box_info(filepath)
            writer.writerow(box_info + ['crack', filepath])
        
        csvoutput.close()
            
    def create_components(self, project_id, task_id, uid, color, min_p, tolerance, max_p, file_name, test_path): 
        
            
        test_dir = os.path.join(test_path, ('{}_test_{}_{}_{}').format(getName(TypeColor, color), min_p, tolerance, max_p))
        # test_dir = os.path.join(test_path, ("test_" + str(min_p) + "_" + str(tolerance) + "_" + str(max_p)))
        test_index = str(min_p) + "_" + str(tolerance) + "_" + str(max_p)
        if os.path.exists(test_dir):
            print(test_dir + " already exists, remaking", flush=True)
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)


        self.lasToCsv(test_dir, min_p, tolerance, max_p, file_name)
        clusters = self.create_csvs(test_dir)
        result = self.csvToLas(test_dir, test_index, clusters)
        
        print(result)

        # self.populate_db(test_dir, test_index, uid, project_id, task_id, color)
        self.populate_csv(test_dir, test_index)
        return "success"

@app.route('/components', methods=['GET'])
def components_api():
    project_id = 151
    task_id = "c9c7deff-e46b-4ed5-8316-79ddf9d19352"
    uid = 12
    color = 2
    min_p = 4
    tolerance = .65
    max_p = 10000
    
    folder_path = 'tasks/task_{}_{}/'.format(project_id, task_id)
    test_path = os.path.join(folder_path, "training" if ppp.training else "tests")
        
    # file_name = os.path.join(folder_path, '{}_filtered_model.las'.format(getName(TypeColor, color)))
    file_name = "3sections - 170 - 253.las"        
    if not (os.path.exists(test_path)):
        os.makedirs(os.path.join(test_path))
            
    processed_data = ppp.create_components(project_id, task_id, uid, color, min_p, tolerance, max_p, file_name, test_path)

    return processed_data

@app.route('/download/<project_id>/<task_id>/<filename>', methods=['GET'])
def download(project_id, task_id, filename):
    # Assuming files are stored in a directory named 'files' under the app root directory
    task = os.path.join(app.root_path, 'task_{}_{}'.format(project_id, task_id))
    
    uploads = os.path.join(task, 'tests/test_10_0.2_10000/component_las_10_0.2_10000')

    # Use send_file function to send the file
    return send_file(os.path.join(uploads, filename), as_attachment=True)


ppp = PointCloudPostProcessor(True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2001, debug=True)