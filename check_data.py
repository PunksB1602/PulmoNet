import os
import pandas as pd
from PIL import Image

DATA_DIR = "./data"
dataset_name = "CheXpert-v1.0-small"
csv_path = os.path.join(DATA_DIR, dataset_name, "train.csv")

# Quick sanity check script that verifies the CSV and tries opening the first image.

print(f"Checking for CSV at: {csv_path}")

if not os.path.exists(csv_path):
    print("ERROR: train.csv not found!")
    print(f"   Make sure you unzipped the dataset inside '{DATA_DIR}'")
    exit()

print("Found train.csv. Reading first 5 rows...")
df = pd.read_csv(csv_path)

first_path_in_csv = df.iloc[0]['Path']
full_image_path = os.path.join(DATA_DIR, first_path_in_csv)

print(f"\nPath in CSV:      {first_path_in_csv}")
print(f"Full System Path: {full_image_path}")

if os.path.exists(full_image_path):
    try:
        img = Image.open(full_image_path)
        print(f"SUCCESS: Image found and opened! Size: {img.size}")
        print("Your data directory is perfect. You can run train.py now.")
    except Exception as e:
        print(f"ERROR: File exists but could not be opened. {e}")
else:
    print("   ERROR: Image file not found at the calculated path.")
    print("   Double check you didn't create a nested folder like:")
    print("   data/CheXpert-v1.0-small/CheXpert-v1.0-small/...")
    # If paths look wrong, compare `Path` entries in CSV against the filesystem.