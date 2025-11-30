from ultralytics import YOLO
import cv2
import os

print("🧪 Testing Trained Model\n")

# Load trained model
model = YOLO('runs/train/minimap_detector/weights/best.pt')

# Test on validation images
val_dir = 'yolo_dataset/images/val'
test_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')][:5]

print(f"📸 Testing on {len(test_images)} images...\n")

for img_name in test_images:
    img_path = os.path.join(val_dir, img_name)
    
    # Run prediction
    results = model.predict(
        img_path,
        save=True,
        conf=0.25,
        show_labels=True,
        show_conf=True,
    )
    
    num_detections = len(results[0].boxes)
    print(f"✅ {img_name}: Detected {num_detections} champions")
    
    # Print each detection
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"   - {class_name}: {confidence:.2%}")
    print()

print(f"📁 Results saved to: runs/detect/predict/")
print("🎉 Check the images to see the detections!")
