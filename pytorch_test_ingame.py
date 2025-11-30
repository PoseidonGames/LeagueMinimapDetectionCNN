from pytorch_model import create_model
from image_drawer import ImageDrawer
from capture_screenshot import capture_screenshot

import os
import sys
import argparse
import time
import colorsys

import torch
import torch_directml

import numpy as np
from mss import mss
import cv2
from torchvision import datasets, models, transforms

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

score_threshold = 0.6

def main():
    parser = argparse.ArgumentParser(description='Detect ingame using model')
    parser.add_argument('model_path', help='path to model weights (.pt)')
    parser.add_argument('num_classes', type=int, help='number of classes (149)', default=149)

    args = parser.parse_args()
    model_path = args.model_path
    num_classes = args.num_classes

    # ---- DirectML device ----
    device = torch_directml.device(1)
    print("Using DirectML device:", device)

    # ---- Model ----
    model = create_model(num_classes, device)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Using full precision (FP32) - DirectML doesn't support FP16 for NMS")

    icons_path = 'league_icons/'
    image_drawer = ImageDrawer(
        icons_path + 'champions',
        icons_path + 'minimap',
        icons_path + 'fog',
        icons_path + 'misc',
        resize=(256, 256),
    )

    champion_to_color = {
        k: colorsys.hsv_to_rgb(v / num_classes, 1.0, 1.0)
        for k, v in image_drawer.champion_to_id.items()
    }

    # Cache minimap crop coordinates
    minimap_crop = None
    
    # Frame skip for performance (process every Nth frame)
    frame_skip = 1  # Set to 2 or 3 to skip frames if still slow
    frame_count = 0

    print("Starting detection... Press 'q' to quit")

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
            minimap_crop = None  # Reset crop detection
            continue

        # Use INTER_NEAREST for faster resizing
        minimap = cv2.resize(minimap, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        orig_minimap = minimap.copy()
        
        # Convert to tensor more efficiently
        minimap_tensor = minimap.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = torch.from_numpy(minimap_tensor).type(torch.float32)

        with torch.no_grad():
            img = img.to(device)
            predictions = model([img])

            for prediction in predictions:
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()

                # Create display image (BGR for OpenCV)
                img_show = orig_minimap.copy()

                for label, box, score in zip(labels, boxes, scores):
                    if score > score_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        label_name = image_drawer.id_to_champion[label]

                        text = '{} {:.2f}'.format(label_name, score)
                        print(text)

                        color = champion_to_color[label_name]
                        img_show = draw_box(img_show, label_name, x1, y1, x2, y2, color, thickness=2)

                print('\n')

                # Display using OpenCV (much faster than matplotlib)
                combined = np.hstack([orig_minimap, img_show])
                cv2.imshow('Detection - Original | Detected', combined)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()