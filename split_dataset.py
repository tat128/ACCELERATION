import os
import shutil
import random

SOURCE_DIR = 'D:/acceleration/train/train'
OUTPUT_DIR = 'dataset'
CLASSES = ['crack', 'hole', 'normal', 'rust', 'scratch']
SPLIT_RATIO = 0.8  # 80% train, 20% validation

# Valid image extensions
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

total_copied = 0

for cls in CLASSES:
    src_class_dir = os.path.join(SOURCE_DIR, cls)
    
    if not os.path.exists(src_class_dir):
        print(f"⚠️ Warning: Could not find folder '{src_class_dir}'")
        continue
        
    # Get all image files regardless of extension case
    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(VALID_EXTS)]
    
    if not images:
        print(f"⚠️ Warning: No valid images found in '{src_class_dir}'")
        continue
        
    random.shuffle(images)
    
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    train_dest = os.path.join(OUTPUT_DIR, 'train', cls)
    val_dest = os.path.join(OUTPUT_DIR, 'val', cls)
    
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    for img in train_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(train_dest, img))
        
    for img in val_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(val_dest, img))

    print(f"✅ Processed '{cls}': {len(train_imgs)} train images, {len(val_imgs)} val images")
    total_copied += len(images)

print(f"\n🎉 Finished! Total images organized into dataset: {total_copied}")