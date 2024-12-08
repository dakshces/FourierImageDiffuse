import os
from PIL import Image

# Define your directory
directory = "data/lsun/bedroom"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Only process .webp files
    if filename.endswith(".webp"):
        filepath = os.path.join(directory, filename)
        try:
            # Attempt to open and verify the image
            with Image.open(filepath) as img:
                img.verify()  # Verify the integrity of the image
        except (IOError, SyntaxError) as e:
            # If an error occurs, it's likely corrupted
            print(f"Corrupted .webp image detected: {filename}")
            os.remove(filepath)  # Delete the corrupted file
            print(f"Deleted: {filename}")

print("Check for .webp files completed.")
