import os
import random

# Define your directory path and output files
directory = "data/lsun/classroom"  
train_output_file = "data/lsun/classroom_train.txt"
val_output_file = "data/lsun/classroom_val.txt"

# Define the number of files for training and validation
train_count = 50000
val_count = 5000

# Get a list of all .webp files in the directory
all_files = [f for f in os.listdir(directory) if f.endswith('.webp')]

# Check if there are enough files for selection
if len(all_files) < train_count + val_count:
    print("Not enough files in the directory for the requested counts.")
else:
    # Randomly shuffle the files
    random.shuffle(all_files)
    
    # Split files into training and validation sets
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    
    # Write training files to the train output file
    with open(train_output_file, 'w') as train_file:
        for file in train_files:
            train_file.write(file + '\n')
    
    # Write validation files to the val output file
    with open(val_output_file, 'w') as val_file:
        for file in val_files:
            val_file.write(file + '\n')
    
    print(f"Training and validation file lists saved as '{train_output_file}' and '{val_output_file}'.")
