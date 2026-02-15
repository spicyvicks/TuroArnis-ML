"""
Step 0a: Convert HEIC to JPG

Batch converts all .HEIC / .heic images in a directory (recursive) to .jpg.
Useful if your dataset was captured on iPhone/iPad.

Usage:
1. pip install pillow-heif
2. Update ROOT_DIR
3. Run script
"""

import os
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

# Register HEIC opener
register_heif_opener()

# Configuration
ROOT_DIR = Path("trio")  # Check this directory recursively
DELETE_ORIGINALS = True  # Set to True to delete .heic files after conversion

def convert_heic_to_jpg():
    print(f"\n{'='*60}")
    print(f"Converting HEIC to JPG in: {ROOT_DIR}")
    print(f"{'='*60}")
    
    if not ROOT_DIR.exists():
        print(f"❌ Directory not found: {ROOT_DIR}")
        return

    # Find all HEIC files (case insensitive)
    heic_files = list(ROOT_DIR.rglob("*.heic")) + list(ROOT_DIR.rglob("*.HEIC"))
    
    if not heic_files:
        print("No HEIC files found.")
        return
        
    print(f"Found {len(heic_files)} HEIC files to convert...")
    
    converted_count = 0
    error_count = 0
    
    for heic_path in tqdm(heic_files):
        try:
            # Open HEIC image
            image = Image.open(heic_path)
            
            # Create JPG path (same name, .jpg extension)
            jpg_path = heic_path.with_suffix(".jpg")
            
            # Save as JPG
            image.convert('RGB').save(jpg_path, "JPEG", quality=95)
            
            converted_count += 1
            
            # Optional: Delete original
            if DELETE_ORIGINALS:
                os.remove(heic_path)
                
        except Exception as e:
            print(f"\n❌ Error converting {heic_path.name}: {e}")
            error_count += 1
            
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"✅ Converted: {converted_count}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    convert_heic_to_jpg()
