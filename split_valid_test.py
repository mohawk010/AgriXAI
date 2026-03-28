import os
import shutil
import random
from pathlib import Path

# Fix the seed for reproducibility specifically for scientific evaluation
random.seed(42)

BASE_DIR = Path(r"D:\plant disease dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)")
VALID_DIR = BASE_DIR / "valid"
TEST_DIR = BASE_DIR / "test"

# If test dir already exists and has files, this might have been run before.
if TEST_DIR.exists() and len(list(TEST_DIR.iterdir())) > 0:
    print(f"Test directory already exists: {TEST_DIR}. Assuming split is already complete.")
    exit(0)

print(f"Creating exact 50/50 split of the existing valid directory...")
TEST_DIR.mkdir(parents=True, exist_ok=True)

total_moved = 0
total_remained = 0

for class_idx, class_dir in enumerate(VALID_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    
    # Ensure corresponding test class dir exists
    test_class_dir = TEST_DIR / class_dir.name
    test_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images in the class
    images = list(class_dir.glob("*.*"))
    # Sort for deterministic shuffling
    images.sort()
    random.shuffle(images)
    
    # Split exactly in half
    split_idx = len(images) // 2
    test_images = images[:split_idx]
    
    # Move files to test directory
    for img_path in test_images:
        dest_path = test_class_dir / img_path.name
        shutil.move(str(img_path), str(dest_path))
        
    total_moved += len(test_images)
    total_remained += (len(images) - len(test_images))

print("Split Complete!")
print(f"New Valid Set: {total_remained} images")
print(f"New Test Set:  {total_moved} images")
