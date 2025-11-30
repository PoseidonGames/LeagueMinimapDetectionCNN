import cv2
import numpy as np
import time
import os
from capture_screenshot import capture_screenshot

class ChampionDetector:
    def __init__(self, icons_folder, threshold=0.75):
        self.threshold = threshold
        self.templates = {}
        
        print("Loading champion icons...")
        for filename in os.listdir(icons_folder):
            if filename.lower().endswith('.png'):
                champ_name = filename.replace('.png', '').replace('.PNG', '')
                icon_path = os.path.join(icons_folder, filename)
                icon = cv2.imread(icon_path, cv2.IMREAD_COLOR)
                
                if icon is not None:
                    # Store original size - we'll resize based on minimap
                    self.templates[champ_name] = icon
                    print(f"  ✓ {champ_name} ({icon.shape[1]}x{icon.shape[0]})")
        
        print(f"\n✓ Loaded {len(self.templates)} champions\n")
    
    def extract_minimap(self, screenshot):
        """Extract minimap from bottom-right"""
        h, w = screenshot.shape[:2]
        minimap_size = int(h * 0.15)  # Adjust if needed
        return screenshot[-minimap_size:, -minimap_size:]
    
    def detect_champions(self, minimap, icon_size=None):
        """
        Detect champions with specific icon size
        icon_size: Target size for templates (e.g., 35 for 35x35 pixels)
        """
        if minimap is None:
            return []
        
        # Auto-detect icon size if not provided
        if icon_size is None:
            # Estimate based on minimap size (icons are ~6-8% of minimap)
            icon_size = int(minimap.shape[0] * 0.07)
        
        detections = []
        
        for champ_name, template_orig in self.templates.items():
            # Resize template to match expected icon size
            template = cv2.resize(template_orig, (icon_size, icon_size))
            
            try:
                result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
                
                # Find all matches above threshold
                locations = np.where(result >= self.threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    detections.append({
                        'champion': champ_name,
                        'x': pt[0],
                        'y': pt[1],
                        'size': icon_size,
                        'confidence': float(confidence)
                    })
            except:
                continue
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlaps(detections)
        
        return detections
    
    def _remove_overlaps(self, detections, overlap_threshold=0.5):
        """Remove overlapping detections, keep highest confidence"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        for det in detections:
            overlap = False
            for kept_det in kept:
                # Calculate IoU (Intersection over Union)
                x1 = max(det['x'], kept_det['x'])
                y1 = max(det['y'], kept_det['y'])
                x2 = min(det['x'] + det['size'], kept_det['x'] + kept_det['size'])
                y2 = min(det['y'] + det['size'], kept_det['y'] + kept_det['size'])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = det['size'] ** 2
                    area2 = kept_det['size'] ** 2
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou > overlap_threshold:
                        overlap = True
                        break
            
            if not overlap:
                kept.append(det)
        
        return kept
    
    def visualize(self, minimap, detections):
        """Draw boxes on detections"""
        vis = minimap.copy()
        
        for det in detections:
            cv2.rectangle(vis, 
                         (det['x'], det['y']), 
                         (det['x'] + det['size'], det['y'] + det['size']), 
                         (0, 255, 0), 2)
            
            label = f"{det['champion'][:8]} {det['confidence']:.2f}"
            cv2.putText(vis, label, 
                       (det['x'], det['y'] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return vis


def main():
    icons_path = r"C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\league_icons\champions"
    detector = ChampionDetector(icons_path, threshold=0.70)
    
    print("="*60)
    print("TESTING DIFFERENT ICON SIZES")
    print("="*60)
    print("Press keys to test different sizes:")
    print("  1 = 25px | 2 = 30px | 3 = 35px | 4 = 40px | 5 = 45px")
    print("  q = quit\n")
    
    icon_size = 35  # Start with 35px
    
    try:
        while True:
            start = time.time()
            
            screenshot = capture_screenshot()
            minimap = detector.extract_minimap(screenshot)
            
            if minimap is None:
                print("No minimap detected")
                time.sleep(1)
                continue
            
            champions = detector.detect_champions(minimap, icon_size=icon_size)
            
            print(f"\r[Size: {icon_size}px] Detected: {len(champions)} | FPS: {1/(time.time()-start):.1f} | {[c['champion'] for c in champions][:5]}", end="")
            
            vis = detector.visualize(minimap, champions)
            cv2.imshow('Minimap Detection', vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                icon_size = 25
                print(f"\n\n>>> Changed to {icon_size}px")
            elif key == ord('2'):
                icon_size = 30
                print(f"\n\n>>> Changed to {icon_size}px")
            elif key == ord('3'):
                icon_size = 35
                print(f"\n\n>>> Changed to {icon_size}px")
            elif key == ord('4'):
                icon_size = 40
                print(f"\n\n>>> Changed to {icon_size}px")
            elif key == ord('5'):
                icon_size = 45
                print(f"\n\n>>> Changed to {icon_size}px")
    
    except KeyboardInterrupt:
        pass
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()