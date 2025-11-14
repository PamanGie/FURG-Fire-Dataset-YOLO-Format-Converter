import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import yaml

def yolo_to_xyxy(center_x, center_y, width, height, img_width, img_height):
    """
    Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
    YOLO format is normalized (0-1), need to multiply by image dimensions
    """
    # Denormalize
    cx = center_x * img_width
    cy = center_y * img_height
    w = width * img_width
    h = height * img_height
    
    # Convert to top-left and bottom-right
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    
    # Clamp to image boundaries
    x1 = max(0, min(img_width - 1, x1))
    y1 = max(0, min(img_height - 1, y1))
    x2 = max(0, min(img_width - 1, x2))
    y2 = max(0, min(img_height - 1, y2))
    
    return x1, y1, x2, y2

def read_yolo_label(label_path):
    """
    Read YOLO label file and return list of bounding boxes
    Returns: list of (class_id, center_x, center_y, width, height)
    """
    bboxes = []
    if not label_path.exists():
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate YOLO format (should be 0-1)
                    if 0 <= center_x <= 1 and 0 <= center_y <= 1 and \
                       0 <= width <= 1 and 0 <= height <= 1:
                        bboxes.append((class_id, center_x, center_y, width, height))
                    else:
                        print(f"  WARNING: Invalid YOLO coordinates in {label_path.name}")
                        print(f"    Values should be 0-1, got: {center_x}, {center_y}, {width}, {height}")
                except ValueError:
                    print(f"  WARNING: Could not parse line in {label_path.name}: {line}")
    
    return bboxes

def read_dataset_yaml(yaml_path):
    """
    Read dataset.yaml file and return class names dictionary
    Returns: dict mapping class_id to class_name, or None if error
    """
    if not yaml_path.exists():
        return None
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        if yaml_data and 'names' in yaml_data:
            # Convert names to dict if it's a dict or list
            names = yaml_data['names']
            if isinstance(names, dict):
                return names
            elif isinstance(names, list):
                # Convert list to dict (index -> name)
                return {i: name for i, name in enumerate(names)}
        
        return None
    except Exception as e:
        print(f"  WARNING: Could not read dataset.yaml: {e}")
        return None

def get_class_name(class_id, class_names_dict, default="unknown"):
    """
    Get class name from class_id using class_names_dict
    """
    if class_names_dict and class_id in class_names_dict:
        return class_names_dict[class_id]
    return default

def draw_bbox(image, x1, y1, x2, y2, label="fire", color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on image
    """
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                  (x1 + label_size[0], y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def verify_dataset(dataset_dir, output_dir=None, num_samples=10, visualize=True):
    """
    Verify YOLO dataset by checking bounding boxes and optionally visualizing them
    
    Args:
        dataset_dir: Path to dataset/images directory
        output_dir: Path to save verification images (optional)
        num_samples: Number of random samples to check per video
        visualize: Whether to create visualization images
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} not found!")
        return
    
    video_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(video_folders)} video folders")
    print("=" * 60)
    
    total_errors = 0
    total_checked = 0
    total_annotated = 0
    
    # Create output directory for verification images
    if visualize and output_dir:
        verif_dir = Path(output_dir)
        verif_dir.mkdir(parents=True, exist_ok=True)
    
    for video_folder in tqdm(video_folders, desc="Verifying videos", unit="video"):
        video_name = video_folder.name
        images_dir = video_folder / 'images'
        labels_dir = video_folder / 'labels'
        yaml_path = video_folder / 'dataset.yaml'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"  WARNING: Missing images or labels folder in {video_name}")
            continue
        
        # Read class names from dataset.yaml
        class_names_dict = read_dataset_yaml(yaml_path)
        if class_names_dict is None:
            print(f"  WARNING: Could not read dataset.yaml for {video_name}, using default names")
            class_names_dict = {0: "fire"}  # Default fallback
        
        # Get all image files
        image_files = sorted(images_dir.glob('*.jpg'))
        
        if not image_files:
            print(f"  WARNING: No images found in {video_name}")
            continue
        
        # Sample random images
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        errors = 0
        checked = 0
        annotated = 0
        
        for img_path in sample_files:
            # Find corresponding label file
            label_path = labels_dir / img_path.name.replace('.jpg', '.txt')
            
            if not label_path.exists():
                print(f"  WARNING: Label file not found for {img_path.name}")
                errors += 1
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ERROR: Could not read image {img_path.name}")
                errors += 1
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Read YOLO labels
            bboxes = read_yolo_label(label_path)
            checked += 1
            
            if bboxes:
                annotated += 1
            
            # Verify bounding boxes
            for class_id, center_x, center_y, width, height in bboxes:
                # Check if bounding box is within image bounds
                x1, y1, x2, y2 = yolo_to_xyxy(center_x, center_y, width, height, 
                                              img_width, img_height)
                
                # Validate bounding box
                if x1 >= x2 or y1 >= y2:
                    print(f"  ERROR: Invalid bounding box in {label_path.name}")
                    print(f"    Box: ({x1}, {y1}, {x2}, {y2})")
                    errors += 1
                
                # Draw bounding box if visualizing
                if visualize:
                    # Get class name from yaml
                    class_name = get_class_name(class_id, class_names_dict, "unknown")
                    label = f"{class_name} ({class_id})"
                    
                    # Use different colors for different classes (default: green for fire)
                    colors = {
                        0: (0, 255, 0),    # Green for fire
                        1: (255, 0, 0),    # Blue for other classes
                        2: (0, 0, 255),    # Red
                    }
                    color = colors.get(class_id, (0, 255, 0))
                    
                    image = draw_bbox(image, x1, y1, x2, y2, label, color)
            
            # Save verification image
            if visualize and output_dir and bboxes:
                verif_img_path = verif_dir / f"{video_name}_{img_path.name}"
                cv2.imwrite(str(verif_img_path), image)
        
        total_errors += errors
        total_checked += checked
        total_annotated += annotated
        
        print(f"  {video_name}: Checked {checked} images, {annotated} with annotations, {errors} errors")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total videos: {len(video_folders)}")
    print(f"  Total images checked: {total_checked}")
    print(f"  Images with annotations: {total_annotated}")
    print(f"  Errors found: {total_errors}")
    
    if visualize and output_dir:
        print(f"  Verification images saved to: {verif_dir}")
    
    if total_errors == 0:
        print("\n✅ All bounding boxes verified successfully!")
    else:
        print(f"\n⚠️  Found {total_errors} errors. Please check the warnings above.")

def main():
    # Paths
    dataset_dir = Path('dataset/images')
    output_dir = Path('verification_images')
    
    # Verify dataset
    verify_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        num_samples=10,  # Check 10 random images per video
        visualize=True   # Create visualization images
    )

if __name__ == '__main__':
    main()

