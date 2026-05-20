import os
import random
import subprocess
import sys

def main():
    images_dir = os.path.join('data', 'images')
    
    if not os.path.exists(images_dir):
        print(f"Error: The directory '{images_dir}' does not exist.")
        return
        
    # Get a list of all image files in the directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"No images found in '{images_dir}'.")
        return
        
    # Pick a random image
    random_image_name = random.choice(images)
    image_path = os.path.join(images_dir, random_image_name)
    
    print(f"Testing on random image: {image_path}")
    print("-" * 50)
    
    # Run the offline detection script with the chosen random image
    # We use the default YOLO and DNN models configured in the script
    subprocess.run([
        sys.executable,
        "darts_score_detection_offline.py",
        "--input", image_path
    ])

if __name__ == "__main__":
    main()
