import numpy as np

# Input and output file paths
input_file = "data/lsun/church_outdoor_val_full.txt"  
output_file = "data/lsun/church_outdoor_val.txt" 
#input_file = "data/lsun/church_outdoor_train_full.txt"  
#output_file = "data/lsun/church_outdoor_train.txt" 
size = 1000
# Load image file names from the input file
with open(input_file, 'r') as f:
    file_names = f.read().splitlines()

# Ensure there are enough files to sample
if len(file_names) < size:
    raise ValueError("The input file contains fewer than 1000 lines.")

# Set random seed and randomly choose 1000 file names
np.random.seed(42)  # Set seed for reproducibility
sampled_files = np.random.choice(file_names, size=size, replace=False)

# Write the sampled file names to the output file
with open(output_file, 'w') as f:
    f.write('\n'.join(sampled_files))

print(f"Randomly selected {size} image file names saved to {output_file}.")
