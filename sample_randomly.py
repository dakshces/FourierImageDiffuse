import os
import shutil
import random

# Define directories
source_dir = 'models/ldm/churches_vanilla_eval_50k/samples/00034386/2024-12-06-15-47-03/img'
target_dir = 'models/ldm/churches_vanilla_eval_50k/samples/00034386/2024-12-06-15-47-03/img_10k'

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# List all files in the source directory
all_files = os.listdir(source_dir)

# Randomly sample 10k files
sample_size = 10000
sampled_files = random.sample(all_files, sample_size)

# Copy sampled files to the target directory
for file in sampled_files:
    src_path = os.path.join(source_dir, file)
    dest_path = os.path.join(target_dir, file)
    shutil.copy(src_path, dest_path)

print(f"Successfully copied {sample_size} files to {target_dir}")
