import os
import numpy as np
from PIL import Image
import re

# Set your image folder path
folder_path = '/mnt/HDD4/anlt/selected_topics/Selected_Topics/HW4/output/rainsnow_edge_hf/'
output_npz = 'pred.npz'

# Initialize dictionary to hold image arrays
images_dict = {}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)

        # Load image and convert to RGB
        image = Image.open(file_path).convert('RGB')
        img_array = np.array(image)

        # Rearrange to (3, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add to dictionary
        filename = re.sub(r'rainsnow_edge_aug20_tta|rainsnow_edge_hf', '', filename)  # Remove file extension
        images_dict[filename] = img_array

# Save to .npz file
np.savez(output_npz, **images_dict)

print(f"Saved {len(images_dict)} images to {output_npz}")
