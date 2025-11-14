import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def parse_xml_annotations(xml_path):
    """
    Parse XML anotasi OpenCV format
    Returns: dict dengan key frame_number dan value list of bboxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get frame dimensions
    frame_width = int(root.find('frameWidth').text)
    frame_height = int(root.find('frameHeight').text)
    
    annotations = {}
    
    # Parse all frames
    frames = root.find('frames')
    if frames is not None:
        for frame_elem in frames.findall('_'):
            frame_num = int(frame_elem.find('frameNumber').text)
            ann_elem = frame_elem.find('annotations')
            
            bboxes = []
            if ann_elem is not None:
                for bbox_elem in ann_elem.findall('_'):
                    bbox_text = bbox_elem.text.strip()
                    if bbox_text:  # Check if not empty
                        parts = bbox_text.split()
                        if len(parts) == 4:
                            x, y, w, h = map(int, parts)
                            bboxes.append((x, y, w, h))
            
            annotations[frame_num] = bboxes
    
    return annotations, frame_width, frame_height

def convert_to_yolo(x, y, w, h, img_width, img_height):
    """
    Convert OpenCV format (x, y, w, h) to YOLO format
    OpenCV: x, y = top-left corner, absolute pixels
    YOLO: center_x, center_y = center point, normalized (0-1)
    """
    # Calculate center point
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    
    # Normalize to 0-1
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    # Clamp to [0, 1] range
    center_x_norm = max(0, min(1, center_x_norm))
    center_y_norm = max(0, min(1, center_y_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return center_x_norm, center_y_norm, width_norm, height_norm

def get_video_name(video_path):
    """Get video name without extension, normalized to lowercase"""
    return Path(video_path).stem.lower()

def extract_frames_and_labels(video_path, xml_path, output_base_dir):
    """
    Extract frames from video and convert annotations to YOLO format
    """
    video_name = get_video_name(video_path)
    
    # Create output directories
    video_output_dir = output_base_dir / video_name
    images_dir = video_output_dir / 'images'
    labels_dir = video_output_dir / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {video_name}...")
    
    # Parse XML annotations
    annotations, xml_frame_width, xml_frame_height = parse_xml_annotations(xml_path)
    print(f"  Found {len(annotations)} annotated frames in XML")
    
    # Open video to get total frame count
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open video {video_path}")
        return
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    extracted_count = 0
    
    # Process frames with progress bar
    with tqdm(total=total_frames, desc=f"  {video_name}", unit="frame", leave=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get actual frame dimensions (might differ from XML)
            img_height, img_width = frame.shape[:2]
            
            # Get annotations for this frame
            bboxes = annotations.get(frame_count, [])
            
            # Save frame
            frame_filename = f"{video_name}{frame_count:03d}.jpg"
            frame_path = images_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create YOLO label file
            label_filename = f"{video_name}{frame_count:03d}.txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for x, y, w, h in bboxes:
                    # Convert to YOLO format
                    # Use XML dimensions for conversion if available, otherwise use actual frame dimensions
                    if xml_frame_width > 0 and xml_frame_height > 0:
                        center_x, center_y, width, height = convert_to_yolo(
                            x, y, w, h, xml_frame_width, xml_frame_height
                        )
                    else:
                        center_x, center_y, width, height = convert_to_yolo(
                            x, y, w, h, img_width, img_height
                        )
                    
                    # YOLO format: class_id center_x center_y width height
                    # Class 0 = fire
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            frame_count += 1
            if bboxes:  # Only count frames with annotations
                extracted_count += 1
            
            # Update progress bar
            pbar.update(1)
    
    cap.release()
    print(f"  Done! Extracted {frame_count} frames ({extracted_count} with annotations)")
    print(f"  Saved to: {video_output_dir}")

def main():
    # Paths
    dataset_dir = Path('furg-fire-dataset-master')
    output_dir = Path('dataset/images')
    
    # Find all video files
    video_files = sorted(dataset_dir.glob('*.mp4'))
    
    print(f"Found {len(video_files)} video files")
    print("=" * 60)
    
    # Process each video with progress bar
    for video_path in tqdm(video_files, desc="Processing videos", unit="video"):
        video_name = get_video_name(video_path)
        xml_path = dataset_dir / f"{video_path.stem}.xml"
        
        # Check if XML exists (case-insensitive)
        if not xml_path.exists():
            # Try to find XML with different case
            xml_files = list(dataset_dir.glob(f"{video_path.stem}*.xml"))
            if xml_files:
                xml_path = xml_files[0]
            else:
                print(f"Warning: No XML found for {video_path.name}, skipping...")
                continue
        
        extract_frames_and_labels(video_path, xml_path, output_dir)
        print()
    
    print("=" * 60)
    print("All videos processed!")

if __name__ == '__main__':
    main()

