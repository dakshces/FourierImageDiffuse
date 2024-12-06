import os
from PIL import Image

# Define the input file containing image paths and the output directory
input_file = "data/lsun/church_outdoor_val.txt"
output_dir = "data/lsun/church_val_jpg"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the file and read all image paths
with open(input_file, "r") as file:
    image_paths = file.readlines()

# Loop through all image paths
for path in image_paths:
    path = path.strip()  # Remove any leading/trailing whitespace
    if path.lower().endswith(".webp"):
        try:
            # Get the output path
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
            
            # Open the WEBP image
            with Image.open('data/lsun/churches/'+path) as img:
                # Resize to 256x256 and convert to RGB
                img = img.resize((256, 256)).convert("RGB")
                # Save as JPG
                img.save(output_path, "JPEG")
                print(f"Converted: {path} -> {output_path}")
        except Exception as e:
            print(f"Failed to convert {path}: {e}")
