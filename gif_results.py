import imageio
import os
import glob

# Define the folder containing the images
folder_path = 'generated_images_fused_nipple_dataset'  # Replace with your actual folder path

# Use glob to get all image files in the folder (assuming they are .png files)
file_pattern = os.path.join(folder_path, '*.png')
filenames = glob.glob(file_pattern)

images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('result.gif', images)
