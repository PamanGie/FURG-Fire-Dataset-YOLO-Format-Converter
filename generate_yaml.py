from pathlib import Path

def create_dataset_yaml_content(video_folder_path):
    """
    Create dataset.yaml content for a specific video folder
    Returns yaml content as string
    """
    yaml_template = """# FURG Fire Dataset - YOLO Configuration
# Dataset converted from OpenCV XML format to YOLO format
# Original dataset: https://github.com/steffensbola/furg-fire-dataset

# Path to this dataset folder (relative to this yaml file or absolute)
path: .

# Number of classes
nc: 1

# Class names
names:
  0: fire

# Image and label folders (relative to this yaml file)
train: images
val: images
test: images

# Dataset structure:
# [video_name]/
#   ├── dataset.yaml (this file)
#   ├── images/
#   │   ├── [video_name]000.jpg
#   │   ├── [video_name]001.jpg
#   │   └── ...
#   └── labels/
#       ├── [video_name]000.txt
#       ├── [video_name]001.txt
#       └── ...
"""
    return yaml_template

def generate_yaml_for_all_videos(dataset_dir):
    """
    Generate dataset.yaml for each video folder in the dataset
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} not found!")
        return
    
    video_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(video_folders)} video folders")
    print("=" * 60)
    
    generated_count = 0
    
    for video_folder in video_folders:
        video_name = video_folder.name
        yaml_path = video_folder / 'dataset.yaml'
        
        # Check if images and labels folders exist
        images_dir = video_folder / 'images'
        labels_dir = video_folder / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"  SKIP: {video_name} - Missing images or labels folder")
            continue
        
        # Create yaml content
        yaml_content = create_dataset_yaml_content(video_folder)
        
        # Write yaml file
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        generated_count += 1
        print(f"  ✓ Created dataset.yaml for {video_name}")
    
    print("=" * 60)
    print(f"Generated {generated_count} dataset.yaml files")

def main():
    # Paths
    dataset_dir = Path('dataset/images')
    
    # Generate yaml files
    generate_yaml_for_all_videos(dataset_dir)
    
    print("\n✅ All dataset.yaml files generated successfully!")

if __name__ == '__main__':
    main()

