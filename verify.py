import os
from PIL import Image

# Define your directory
directory = "data/lsun/conference"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    try:
        # Attempt to open the image file
        with Image.open(filepath) as img:
            img.verify()  # Verify the integrity of the image
    except (IOError, SyntaxError) as e:
        # If an error occurs, it's likely corrupted
        print(f"Corrupted image detected: {filename}")
        os.remove(filepath)  # Delete the corrupted file
        print(f"Deleted: {filename}")

print("Check completed.")
