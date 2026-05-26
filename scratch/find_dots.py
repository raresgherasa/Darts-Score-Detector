import cv2
import numpy as np

def main():
    img = cv2.imread('test_out/darts_1779195243354.jpg')
    if img is None:
        print("Image not found")
        return
        
    # Find red pixels (BGR: 0, 0, 255)
    # We'll look for pixels close to (0, 0, 255)
    red_mask = (img[:, :, 2] > 200) & (img[:, :, 0] < 50) & (img[:, :, 1] < 50)
    red_pts = np.argwhere(red_mask)
    
    # Find blue pixels (BGR: 255, 0, 0)
    blue_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
    blue_pts = np.argwhere(blue_mask)
    
    print(f"Red pixels found: {len(red_pts)}")
    if len(red_pts) > 0:
        # Group and print centroids
        # Since it's a small circle, let's find the mean
        mean_y, mean_x = np.mean(red_pts, axis=0)
        print(f"Red dot center (y, x): ({mean_y:.2f}, {mean_x:.2f})")
        
    print(f"Blue pixels found: {len(blue_pts)}")
    if len(blue_pts) > 0:
        mean_y, mean_x = np.mean(blue_pts, axis=0)
        print(f"Blue dot center (y, x): ({mean_y:.2f}, {mean_x:.2f})")

if __name__ == '__main__':
    main()
