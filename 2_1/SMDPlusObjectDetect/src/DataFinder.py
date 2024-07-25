import os
import glob

def find_files_with_only_object_index(directory, object_index):
    # Get list of txt files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    # Initialize a list to store the names of txt files with only the specified object index
    files_with_only_object_index = []

    # Iterate over each txt file
    for txt_file in txt_files:
        # Open the file and read the lines
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Check if all lines start with the specified object index
        if all(line.startswith(str(object_index)) for line in lines):
            # If all lines start with the object index, add the file to the list
            files_with_only_object_index.append(txt_file)

    # Return the list of txt files with only the specified object index
    return files_with_only_object_index

def count_object_index(directory, object_index):
    # Get list of txt files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    # Initialize a counter for the specified object index
    count = 0

    # Iterate over each txt file
    for txt_file in txt_files:
        # Open the file and read the lines
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Count the lines that start with the specified object index
        count += sum(line.startswith(str(object_index)) for line in lines)

    # Return the count of the specified object index
    return count

def find_empty_txt_files(directory):
    # Get list of txt files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    # Initialize a list to store the names of empty txt files
    empty_files = []

    # Iterate over each txt file
    for txt_file in txt_files:
        # Check if the file is empty
        if os.stat(txt_file).st_size == 0:
            # If the file is empty, add it to the list
            empty_files.append(txt_file)

    # Return the list of empty txt files
    return empty_files

# Set the directory containing the txt files
directory = "datasets/SMD_Plus/val"

# Set the object index to find
object_index = 0

# Find the txt files with only the specified object index
files_with_only_object_index = find_files_with_only_object_index(directory, object_index)

# Print the txt files with only the specified object index
# for file in files_with_only_object_index:
#     print(f"Txt file with only object index {object_index}: {file}")



for i in range(7) :
    count = count_object_index(directory, i)
    print(f"The total count of object index {i} is {count}.")


# # Find the empty txt files
# empty_files = find_empty_txt_files(directory)

# # Print the empty txt files
# for empty_file in empty_files:
#     print(f"Empty txt file: {empty_file}")
    