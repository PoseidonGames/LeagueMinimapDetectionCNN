from ultralytics import YOLO
import torch
import os

print("="*60)
print("🎮 YOLO Training - League Minimap Detection")
print("="*60)

# Check device
device = 'cpu'
print(f"\n💻 Device: {device} (AMD Ryzen 7 7800X3D)")
print("⚠️  Training on CPU - this will take longer but will work!")

# Verify dataset exists
if not os.path.exists('yolo_dataset/data.yaml'):
    print("\n❌ Error: yolo_dataset/data.yaml not found!")
    print("📁 Current directory:", os.getcwd())
    print("📂 Contents:", os.listdir('.'))
    exit(1)

print("\n✅ Dataset found!")

# Show dataset info
print("\n📊 Dataset Configuration:")
with open('yolo_dataset/data.yaml', 'r') as f:
    print(f.read())

# Count images
train_images = len([f for f in os.listdir('yolo_dataset/images/train') if f.endswith('.jpg')])
val_images = len([f for f in os.listdir('yolo_dataset/images/val') if f.endswith('.jpg')])
print(f"\n📸 Training images: {train_images}")
print(f"📸 Validation images: {val_images}")

print("\n" + "="*60)
print("🚀 Starting YOLO Training...")
print("="*60)
print("\n⏱️  Estimated time: 2-4 hours on CPU")
print("💡 Tip: Don't close this window!\n")

# Load YOLOv8 nano model (smallest, fastest)
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device='cpu',
    workers=4,
    project='runs/train',
    name='minimap_detector',
    patience=20,
    save=True,
    plots=True,
    cache=False,
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    verbose=True,
    seed=42,
    amp=False,
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.0,
)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Results saved to: runs/train/minimap_detector/")
print(f"🎯 Best model: runs/train/minimap_detector/weights/best.pt")
print(f"📊 Training curves: runs/train/minimap_detector/results.png")
print(f"🎯 Confusion matrix: runs/train/minimap_detector/confusion_matrix.png")

print("\n🎉 You can now use your trained model!")
print("💡 Next: Run testmodelyolo.py to test it on images")
