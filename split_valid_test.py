import os
import shutil
import random
from pathlib import Path

# Fix the seed for reproducibility specifically for scientific evaluation
random.seed(42)

ROOT = Path(__file__).parent
BASE_DIR = ROOT / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
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
    
    # Copy files to test directory
    for img_path in test_images:
        dest_path = test_class_dir / img_path.name
        shutil.copy2(str(img_path), str(dest_path))
        
    total_moved += len(test_images)
    total_remained += (len(images) - len(test_images))

    class_name = class_dir.name
    valid_count = len(images) - len(test_images)
    test_count = len(test_images)
    if abs(valid_count - test_count) > 1:
        print(f"  WARNING: {class_name} — imbalanced split: valid={valid_count}, test={test_count}")

print("Split Complete!")
print(f"New Valid Set: {total_remained} images")
print(f"New Test Set:  {total_moved} images")

import json
manifest = {
    "seed": 42,
    "total_valid": total_remained,
    "total_test": total_moved,
    "timestamp": str(Path(__file__).stat().st_mtime),
}
with open(BASE_DIR / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"-> Saved split manifest to {BASE_DIR / 'split_manifest.json'}")
