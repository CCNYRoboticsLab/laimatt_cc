import glob
import shutil
import os
import re



def extract_substring(string):
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
    
list = [0]*15

for filepath in glob.iglob('output/*'):
    index = extract_substring(filepath) - 1
    list[index] = filepath

print(list)

# # Move the file to the destination directory
# shutil.move(source, destination)