from image_drawer import ImageDrawer
from capture_screenshot import capture_screenshot

import os
import sys
import argparse
import time
import colorsys
import json
from datetime import datetime

import numpy as np
from mss import mss
import cv2

# YOLO import
from ultralytics import YOLO

def draw_box(img, label, x1, y1, x2, y2, color, thickness=1, text_color=(230, 230, 230)):
    color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness=thickness)

    label_box_thickness = max(1, int(thickness / 3.0))
    text_size = cv2.getTextSize(label, 0, fontScale=label_box_thickness / 2, thickness=thickness)[0]

    offset = 3

    label_box_point = (x1 + text_size[0], y1 - text_size[1] - offset)
    img = cv2.rectangle(img, (x1, y1), label_box_point, color_bgr, thickness=-1)
    img = cv2.putText(
        img,
        label,
        (x1, y1 - offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_box_thickness / 2,
        text_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return img

def main():
    parser = argparse.ArgumentParser(description='Detect ingame using YOLO model')
    parser.add_argument('model_path', nargs='?', help='path to YOLO model weights (.pt)', 
                        default=r'C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\best.pt')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, default='champion_positions.json',
                        help='output file for champion positions (default: champion_positions.json)')

    args = parser.parse_args()
    model_path = args.model_path
    score_threshold = args.conf
    output_file = args.output

    # ---- Load YOLO Model ----
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Use GPU if available (CUDA), otherwise CPU
    # YOLO handles device automatically
    print(f"Using device: {model.device}")
    print(f"Confidence threshold: {score_threshold}")
    print(f"Exporting positions to: {output_file}")

    # ---- Image Drawer (for colors/icons if needed) ----
    icons_path = 'league_icons/'
    
    # For single-class detection, we just use one color
    # If you want different colors per champion later, keep ImageDrawer
    champion_color = (0.0, 1.0, 1.0)  # Cyan in RGB

    # Cache minimap crop coordinates
    minimap_crop = None
    
    # Frame skip for performance
    frame_skip = 1  # YOLO is fast, you might not need to skip
    frame_count = 0

    print("Starting YOLO detection... Press 'q' to quit")

    while True:
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        screenshot = capture_screenshot()
        h, w, c = screenshot.shape

        minimap_ratio = 800 / 1080
        minimap_x = int(minimap_ratio * h)
        minimap_size = h - minimap_x

        minimap = screenshot[-minimap_size:, -minimap_size:]

        # Use cached crop if available, otherwise detect borders
        if minimap_crop is None:
            h, w, c = minimap.shape
            left = right = top = bottom = 0

            for x in range(w):
                y = int(h / 2)
                r, g, b = minimap[y][x]
                if r < 10 and g < 10 and b < 10:
                    left = x
                    break

            for x in range(w - 1, 0, -1):
                y = int(h / 2)
                r, g, b = minimap[y][x]
                if r < 10 and g < 10 and b < 10:
                    right = x
                    break

            for y in range(h):
                x = int(w / 2)
                r, g, b = minimap[y][x]
                if r < 10 and g < 10 and b < 10:
                    top = y
                    break

            for y in range(h - 1, 0, -1):
                x = int(w / 2)
                r, g, b = minimap[y][x]
                if r < 10 and g < 10 and b < 10:
                    bottom = y
                    break

            minimap_crop = (top, bottom, left, right)
        else:
            top, bottom, left, right = minimap_crop

        minimap = minimap[top - 1:bottom + 1, left - 1:right + 1]

        h, w, c = minimap.shape
        if h == 0 or w == 0:
            print('Could not detect game')
            minimap_crop = None
            continue

        # Resize to 640x640 (YOLO's native size)
        minimap_resized = cv2.resize(minimap, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)

        orig_minimap = minimap_resized.copy()
        
        # ---- YOLO Inference ----
        results = model.predict(
            minimap_resized,
            conf=score_threshold,
            verbose=False,  # Suppress YOLO's print output
            device=model.device,
        )

        # Create display image
        img_show = orig_minimap.copy()

        # Store champion positions for export
        champion_positions = {
            'timestamp': datetime.now().isoformat(),
            'frame': frame_count,
            'champions': []
        }

        # Process detections 
        for result in results:
            boxes = result.boxes
            
            if len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Calculate center position
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Get class name (should be 'champion' for your model)
                    label_name = model.names[class_id]
                    
                    # Create label with confidence percentage
                    label_text = '{} {:.1%}'.format(label_name, confidence)
                    
                    print(f"{label_name}: {confidence:.1%} at center ({center_x}, {center_y})")

                    # Store position data
                    champion_positions['champions'].append({
                        'label': label_name,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        },
                        'center': {
                            'x': center_x,
                            'y': center_y
                        },
                        'normalized_center': {
                            'x': round(center_x / 640, 3),  # Normalized 0-1
                            'y': round(center_y / 640, 3)
                        }
                    })

                    # Draw box with confidence
                    img_show = draw_box(
                        img_show, 
                        label_text,
                        x1, y1, x2, y2, 
                        champion_color, 
                        thickness=2
                    )

        # Export positions to JSON file
        with open(output_file, 'w') as f:
            json.dump(champion_positions, f, indent=2)

        if len(results[0].boxes) > 0:
            print(f"Total detections: {len(results[0].boxes)}")
            print(f"Exported to: {output_file}\n")

        # Display using OpenCV - BIGGER WINDOW
        display_size = 1024
        orig_display = cv2.resize(orig_minimap, (display_size, display_size))
        show_display = cv2.resize(img_show, (display_size, display_size))
        
        combined = np.hstack([orig_display, show_display])
        cv2.imshow('YOLO Detection - Original | Detected', combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()