import os

# Set paths
haze_folder = "/kaggle/input/o-haze-combined/o_haze/o_haze/O-HAZY/hazy"
gt_folder = "/kaggle/input/o-haze-combined/o_haze/o_haze/O-HAZY/GT"

# Get filenames without extensions
haze_files = {os.path.splitext(f)[0] for f in os.listdir(haze_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))}
gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))}

# Find common images
common_files = haze_files & gt_files  # Intersection of both sets

# Print counts
print(f"Total images in 'haze' folder: {len(haze_files)}")
print(f"Total images in 'gt' folder: {len(gt_files)}")
print(f"Total common images in both folders: {len(common_files)}\n")

# Print all images in haze folder
print("All images in 'haze' folder:")
for file in sorted(haze_files):
    print(file)

print("\nAll images in 'gt' folder:")
for file in sorted(gt_files):
    print(file)

# Print common images
print("\nImages present in both 'haze' and 'gt' folders:")
for file in sorted(common_files):
    print(file)