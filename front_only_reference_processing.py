"""
Front-Only Reference Processing — Safe & Simple

1. Backup existing feature_templates.json
2. Flip images in reference_poses/front ONLY
3. Extract front templates using existing script
4. Merge new front templates into backup, preserving left/right

This never touches:
  - reference_poses/left
  - reference_poses/right  
  - dataset_split/train/left
  - dataset_split/train/right
  - dataset_split/test/left
  - dataset_split/test/right
"""

import cv2
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Paths
REF_FRONT = Path("reference_poses/front")
TEMPLATES = Path("hybrid_classifier/feature_templates.json")
BACKUP = Path("hybrid_classifier/feature_templates_backup_all.json")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct', 'neutral'
]


def step1_backup_templates():
    print("\n[1/4] Backing up existing templates...")
    if not TEMPLATES.exists():
        print("      No existing templates to backup.")
        return None
    shutil.copy2(TEMPLATES, BACKUP)
    with open(TEMPLATES, 'r') as f:
        data = json.load(f)
    front_n = sum(1 for k in data if k.startswith("front_"))
    left_n = sum(1 for k in data if k.startswith("left_"))
    right_n = sum(1 for k in data if k.startswith("right_"))
    print(f"      Saved backup: front={front_n}, left={left_n}, right={right_n}")
    return data


def step2_flip_front_images():
    print("\n[2/4] Flipping front reference images...")
    if not REF_FRONT.exists():
        print(f"      ERROR: {REF_FRONT} not found!")
        return 0

    count = 0
    for class_name in CLASS_NAMES:
        class_dir = REF_FRONT / class_name
        if not class_dir.exists():
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue
        for img_path in tqdm(images, desc=f"      {class_name}", leave=False):
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(img_path), cv2.flip(img, 1))
                count += 1
    print(f"      Flipped {count} images in reference_poses/front")
    return count


def step3_extract_front_templates():
    print("\n[3/4] Extracting front templates...")
    result = subprocess.run(
        [sys.executable, "hybrid_classifier/1_extract_reference_features.py", "--viewpoint", "front"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"      ERROR: {result.stderr}")
        return False
    return True


def step4_merge_templates(existing_data):
    print("\n[4/4] Merging front templates into existing JSON...")

    # Load the front-only templates just generated
    with open(TEMPLATES, 'r') as f:
        front_only = json.load(f)

    if existing_data is None:
        print("      No existing data — front_only is the new templates.")
        return front_only

    # Remove old front templates from existing data
    old_front = [k for k in existing_data if k.startswith("front_")]
    for k in old_front:
        del existing_data[k]

    # Add new front templates
    existing_data.update(front_only)

    # Save merged result
    with open(TEMPLATES, 'w') as f:
        json.dump(existing_data, f, indent=2)

    front_n = sum(1 for k in existing_data if k.startswith("front_"))
    left_n = sum(1 for k in existing_data if k.startswith("left_"))
    right_n = sum(1 for k in existing_data if k.startswith("right_"))
    print(f"      Merged: front={front_n}, left={left_n}, right={right_n}")
    return existing_data


def main():
    print("="*60)
    print("FRONT-ONLY REFERENCE PROCESSING")
    print("="*60)
    print("\nThis ONLY touches:")
    print("  - reference_poses/front/* (flips images)")
    print("  - hybrid_classifier/feature_templates.json (updates front templates)")
    print("\nIt does NOT touch:")
    print("  - reference_poses/left or right")
    print("  - dataset_split/")
    print("  - Any trained models")
    print("\nA backup of templates is saved to feature_templates_backup_all.json")
    print("="*60)

    ans = input("\nProceed? (yes/no): ").strip().lower()
    if ans != "yes":
        print("Cancelled.")
        return

    # Execute
    existing = step1_backup_templates()
    flipped = step2_flip_front_images()
    if flipped == 0:
        print("No images to flip. Check reference_poses/front exists.")
        return

    success = step3_extract_front_templates()
    if not success:
        print("Template extraction failed. Restoring backup...")
        if BACKUP.exists():
            shutil.copy2(BACKUP, TEMPLATES)
        return

    merged = step4_merge_templates(existing)

    print("\n" + "="*60)
    print("DONE — Front reference processing complete")
    print("="*60)
    print("\nNext steps:")
    print("  1. Augment front training images (no flip)")
    print("  2. python hybrid_classifier/2b_generate_node_hybrid_features.py --viewpoint front")
    print("  3. Generate synthetic features")
    print("  4. Train models")
    print("="*60)


if __name__ == "__main__":
    main()
