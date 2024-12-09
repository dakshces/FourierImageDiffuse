from PIL import Image
import os
import random

# Path to folder containing the images
#folder_path = "models/ldm/bedroom_freq_50k/samples/00034706/e73/img/"
#folder_path = "models/ldm/bedroom_vanilla_50k/samples/00044555/2024-12-08-20-45-05/img/"
#folder_path = "models/ldm/churches_vanilla_eval_50k/samples/00034386/2024-12-06-15-47-03/img/"
#folder_path = "models/ldm/churches_freq_50k/samples/00029176/2024-12-09-13-49-55/img/"
#folder_path = "models/ldm/classroom_vanilla_50k/samples/00035949/2024-12-08-19-17-43/img/"
#folder_path = "models/ldm/classroom_freq_50k/samples/00035428/2024-12-08-21-23-56/img/"
#folder_path = "models/ldm/conference_vanilla_50k/samples/00036470/2024-12-08-19-51-34/img/"
folder_path = "models/ldm/conference_freq_50k/samples/00034386/2024-12-08-21-39-02/img/"
# Load all images
#bedroom_freq_idxs = ['0188', '0487', '0509', '7952', '7960', '8042', '8215', '8296', '8693']
#classrooom_freq_idxs = ['0035', '9838', '2505', '2517', '9815', '5501', '5489', '5472', '0581']
idxs =  random.sample(range(1000, 10000), 9)

image_files = [folder_path+f'sample_00{idx}.png' for idx in idxs]

# Sort images to ensure consistent ordering
image_files.sort()

# Ensure there are exactly 9 images
if len(image_files) != 9:
    raise ValueError("The folder must contain exactly 9 PNG images.")

# Open the images and ensure they're all the same size
images = [Image.open(img) for img in image_files]
width, height = images[0].size  # Assuming all images are the same size

# Create a blank image for the collage
collage_width = width * 3
collage_height = height * 3
collage = Image.new('RGB', (collage_width, collage_height))

# Paste images into the collage
for idx, img in enumerate(images):
    x_offset = (idx % 3) * width
    y_offset = (idx // 3) * height
    collage.paste(img, (x_offset, y_offset))

# Save the collage
output_path = "collage.png"
collage.save(output_path)
print(f"Collage saved at {output_path}")
