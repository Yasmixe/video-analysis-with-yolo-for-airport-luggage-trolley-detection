import os

folder_path = r'C:\Users\yasmi\Documents\detection\augmented'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Loop through the files
for i, file in enumerate(files):
    # Get the file extension
    extension = os.path.splitext(file)[1]

    # Define the new file name
    new_name = 'data_' + str(i) + extension

    # Build the old and new file paths
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_path, new_path)