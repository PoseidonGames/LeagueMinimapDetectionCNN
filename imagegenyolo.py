"""
Minimap generator for YOLO training with full realism
Includes: champions, towers, wards, minions, fog operations
NOW WITH PROPER MULTIPROCESSING FOR 3-4X SPEEDUP!
"""

import torch
import numpy as np
import cv2
import os
import random
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time


class MinimapGenerator:
    def __init__(self, champion_icons_path, minimap_path, fog_path, misc_path, resize=(640, 640), silent=False):
        """
        Args:
            champion_icons_path: Path to champion circular icons
            minimap_path: Path to minimap backgrounds
            fog_path: Path to fog of war overlays
            misc_path: Path to misc elements (towers, wards, minions, etc.)
            resize: Output size (width, height) - 640x640 is good for YOLO
            silent: Suppress print statements (for worker processes)
        """
        self.champion_icons_path = champion_icons_path
        self.minimap_path = minimap_path
        self.fog_path = fog_path
        self.misc_path = misc_path
        self.resize = resize
        self.silent = silent
        
        self.setup()
    
    def load_image(self, path):
        """Load PNG image with alpha channel"""
        if os.path.isfile(path):
            basename = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path).convert("RGBA")
            img = transforms.ToTensor()(img)
            return (img, basename)
        return None
    
    def setup(self):
        """Load all assets"""
        if not self.silent:
            print("Loading assets...")
        
        # Load champion icons
        self.champion_icons = []
        for filename in os.listdir(self.champion_icons_path):
            path = os.path.join(self.champion_icons_path, filename)
            result = self.load_image(path)
            if result:
                self.champion_icons.append(result)
        self.champion_icons.sort(key=lambda x: x[1])
        
        # Load minimaps
        self.minimaps = []
        for filename in os.listdir(self.minimap_path):
            path = os.path.join(self.minimap_path, filename)
            result = self.load_image(path)
            if result:
                self.minimaps.append(result)
        
        # Load fog overlays
        self.fogs = []
        for filename in os.listdir(self.fog_path):
            path = os.path.join(self.fog_path, filename)
            result = self.load_image(path)
            if result:
                self.fogs.append(result)
        
        # Load misc elements
        self.miscs = []
        for filename in os.listdir(self.misc_path):
            path = os.path.join(self.misc_path, filename)
            result = self.load_image(path)
            if result:
                self.miscs.append(result)
        
        # Calculate minimap size and ratio
        self.minimap_size = tuple(self.minimaps[0][0].shape[1:3])
        self.minimap_ratio = (self.resize[0] / self.minimap_size[0], 
                             self.resize[1] / self.minimap_size[1])
        
        # Create champion ID mappings
        self.id_to_champion = {}
        self.champion_to_id = {}
        for idx, (_, label) in enumerate(self.champion_icons):
            champion_id = idx  # YOLO uses 0-indexed classes
            self.id_to_champion[champion_id] = label
            self.champion_to_id[label] = champion_id
        
        # Parse misc elements
        self._parse_misc_elements()
        
        if not self.silent:
            print(f"✓ Loaded {len(self.champion_icons)} champions")
            print(f"✓ Loaded {len(self.minimaps)} minimap backgrounds")
            print(f"✓ Loaded {len(self.fogs)} fog overlays")
            print(f"✓ Loaded {len(self.towers)} tower types")
            print(f"✓ Loaded {len(self.wards)} ward types")
            print(f"✓ Output size: {self.resize}")
    
    def _parse_misc_elements(self):
        """Parse misc elements into categories"""
        self.ally_outlines = []
        self.enemy_outlines = []
        self.towers = []
        self.wards = []
        self.blue_minion = None
        self.red_minion = None
        
        for img, label in self.miscs:
            # Champion outlines (colored circles)
            if 'ally_circle' in label or 'recalloutline' in label:
                img_np = img.numpy().transpose((2, 1, 0))
                img_np = cv2.resize(img_np, (120, 120), interpolation=cv2.INTER_AREA)
                img_np = img_np.transpose((2, 1, 0))
                self.ally_outlines.append((torch.from_numpy(img_np), label))
            
            elif 'enemy_circle' in label or 'recallhostileoutline' in label:
                img_np = img.numpy().transpose((2, 1, 0))
                img_np = cv2.resize(img_np, (120, 120), interpolation=cv2.INTER_AREA)
                img_np = img_np.transpose((2, 1, 0))
                self.enemy_outlines.append((torch.from_numpy(img_np), label))
            
            # Towers
            elif 'tower' in label or 'inhib' in label or 'nexus' in label:
                self.towers.append((img, label))
            
            # Wards
            elif 'ward' in label or 'jammer' in label:
                self.wards.append((img, label))
            
            # Minions
            elif 'minionmapcircle_ally' == label:
                self.blue_minion = (img, label)
            elif 'minionmapcircle_enemy' == label:
                self.red_minion = (img, label)
    
    def overlay_transparent(self, background, overlay, x, y):
        """Overlay transparent image onto background"""
        bg_h, bg_w = background.shape[0], background.shape[1]
        
        if x >= bg_w or y >= bg_h:
            return background
        
        h, w = overlay.shape[0], overlay.shape[1]
        if h == 0 or w == 0:
            return background
        
        # Clip overlay to fit within background
        if x + w > bg_w:
            w = bg_w - x
            overlay = overlay[:, :w]
        if y + h > bg_h:
            h = bg_h - y
            overlay = overlay[:h]
        
        # Ensure alpha channel exists
        if overlay.shape[2] < 4:
            overlay = np.concatenate([
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
            ], axis=2)
        
        overlay_img = overlay[..., :3]
        alpha = overlay[..., 3:4]
    
        x = max(x, 0)
        y = max(y, 0)
        
        # Alpha blending
        background[y:y+h, x:x+w, :3] = (1.0 - alpha) * background[y:y+h, x:x+w, :3] + alpha * overlay_img
        
        return background
    
    def create_minimap_base(self):
        """Create base minimap with fog"""
        minimap, _ = random.choice(self.minimaps)
        minimap = minimap.numpy().copy()
        
        fog, _ = random.choice(self.fogs)
        fog = fog.numpy()
        
        # Blend fog with minimap
        alpha = 0.9
        minimap = cv2.addWeighted(minimap, alpha, fog, 1 - alpha, 0)
        
        return minimap
    
    def draw_towers(self, minimap, overlay_ops, fog_ops):
        """Draw towers at random positions"""
        h, w = minimap.shape[1:3]
        
        # Random number of towers (3-8)
        num_towers = random.randint(3, 8)
        
        for _ in range(num_towers):
            if not self.towers:
                continue
                
            tower_img, _ = random.choice(self.towers)
            tower_np = tower_img.numpy()
            
            # Random position
            x = random.randint(0, w - 20)
            y = random.randint(0, h - 20)
            
            overlay_ops.append((tower_np, x, y))
            fog_ops.append((tower_np, x, y))
    
    def draw_wards(self, minimap, overlay_ops, fog_ops):
        """Draw wards at random positions"""
        h, w = minimap.shape[1:3]
        
        # Random number of wards (0-10)
        num_wards = random.randint(0, 10)
        
        for _ in range(num_wards):
            if not self.wards:
                continue
                
            ward_img, _ = random.choice(self.wards)
            ward_np = ward_img.numpy()
            
            # Random position
            constraint = 50
            x = random.randint(constraint, w - constraint)
            y = random.randint(constraint, h - constraint)
            
            overlay_ops.append((ward_np, x, y))
            fog_ops.append((ward_np, x, y))
    
    def draw_minions(self, minimap, overlay_ops, fog_ops):
        """Draw minion waves"""
        h, w = minimap.shape[1:3]
        
        if self.blue_minion is None or self.red_minion is None:
            return
        
        blue_minion_img, _ = self.blue_minion
        red_minion_img, _ = self.red_minion
        blue_np = blue_minion_img.numpy()
        red_np = red_minion_img.numpy()
        
        # Spawn 2-4 minion waves
        num_waves = random.randint(2, 4)
        
        for _ in range(num_waves):
            # Random team
            minion_img = blue_np if random.random() < 0.5 else red_np
            
            # Random lane position
            x = random.randint(50, w - 50)
            y = random.randint(50, h - 50)
            
            # Spawn 3-6 minions in a group
            num_minions = random.randint(3, 6)
            for _ in range(num_minions):
                offset_x = random.randint(-15, 15)
                offset_y = random.randint(-15, 15)
                
                overlay_ops.append((minion_img, x + offset_x, y + offset_y))
                fog_ops.append((minion_img, x + offset_x, y + offset_y))
    
    def create_champions(self, num_champions=10):
        """
        Create champion icons with team-colored outlines
        Returns: (ally_champions, enemy_champions)
        """
        # Sample 10 random champions
        selected = random.sample(self.champion_icons, min(num_champions, len(self.champion_icons)))
        
        # Split into teams
        ally_icons = selected[:5]
        enemy_icons = selected[5:10]
        
        ally_champions = []
        enemy_champions = []
        
        # Create ally champions (blue team)
        for icon, label in ally_icons:
            icon_np = icon.numpy().copy()
            
            # Add colored outline
            if self.ally_outlines:
                outline, _ = random.choice(self.ally_outlines)
                outline_np = outline.numpy()
                
                # FIX: Proper axis order - (C, H, W) -> (H, W, C)
                icon_np = icon_np.transpose((1, 2, 0))
                outline_np = outline_np.transpose((1, 2, 0))
                icon_np = self.overlay_transparent(icon_np, outline_np, 0, 0)
                # Keep in (H, W, C) format
            
            ally_champions.append((icon_np, label))
        
        # Create enemy champions (red team)
        for icon, label in enemy_icons:
            icon_np = icon.numpy().copy()
            
            if self.enemy_outlines:
                outline, _ = random.choice(self.enemy_outlines)
                outline_np = outline.numpy()
                
                # FIX: Proper axis order - (C, H, W) -> (H, W, C)
                icon_np = icon_np.transpose((1, 2, 0))
                outline_np = outline_np.transpose((1, 2, 0))
                icon_np = self.overlay_transparent(icon_np, outline_np, 0, 0)
                # Keep in (H, W, C) format
            
            enemy_champions.append((icon_np, label))
        
        return ally_champions, enemy_champions

    def place_champions(self, minimap, champions, overlay_ops, fog_ops):
        """
        Place champions randomly on minimap
        Returns: List of (label, x, y, w, h) in pixel coordinates
        """
        h, w = minimap.shape[1:3]
        champion_data = []
        
        for champ_img, label in champions:
            # Resize champion icon (already in H, W, C format)
            champ_img = cv2.resize(champ_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            c_h, c_w = champ_img.shape[0:2]  # FIX: Correct order
            half_w, half_h = int(c_w / 2), int(c_h / 2)
            
            # Random position (allow partial off-screen)
            x = random.randint(-half_w + 1, w - half_w)
            y = random.randint(-half_h + 1, h - half_h)
            
            # 20% chance to spawn near another champion (teamfights)
            if len(champion_data) > 0 and random.random() < 0.2:
                other_label, other_x, other_y, other_w, other_h = random.choice(champion_data)
                nearness = c_w
                bias_x = random.uniform(-1, 1)
                bias_y = random.uniform(-1, 1)
                x = int(other_x + nearness * bias_x)
                y = int(other_y + nearness * bias_y)
                x = max(-half_w + 1, min(x, w - half_w))
                y = max(-half_h + 1, min(y, h - half_h))
            
            # Clip champion if going off-screen
            if x < 0:
                champ_img = champ_img[:, -x:]
                c_w = champ_img.shape[1]
                x = 0
            if y < 0:
                champ_img = champ_img[-y:, :]
                c_h = champ_img.shape[0]
                y = 0
            if x + c_w > w:
                champ_img = champ_img[:, :w-x]
                c_w = champ_img.shape[1]
            if y + c_h > h:
                champ_img = champ_img[:h-y, :]
                c_h = champ_img.shape[0]
            
            # Add to overlay operations
            overlay_ops.append((champ_img, x, y))
            fog_ops.append((champ_img, x, y))
            
            # Store position data
            champion_data.append((label, x, y, c_w, c_h))
        
        return champion_data

    def perform_fog_operations(self, minimap, fog_ops):
        """Apply fog of war lighting around visible objects"""
        fog_filter = np.zeros(minimap.shape, minimap.dtype)
        fog_filter = fog_filter.transpose((2, 1, 0)).copy()
        
        for img, x, y in fog_ops:
            h, w = img.shape[1:3]
            distance = 40  # Visibility radius
            
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # Draw visibility circle
            fog_filter = cv2.circle(fog_filter, (center_x, center_y), distance, 
                                   (0.0, 0.0, 0.0, 1.0), -1)
        
        fog_filter = fog_filter.transpose((2, 1, 0))
        fog_mask = fog_filter.astype('uint8')
        fog_mask = fog_mask.transpose((2, 1, 0))
        fog_mask[fog_mask[...] == 1] = 255
        fog_mask = cv2.split(fog_mask)[3]
        
        fog_filter = fog_filter.transpose((2, 1, 0))
        minimap_t = minimap.transpose((2, 1, 0))
        
        fog_filter = cv2.bitwise_or(minimap_t, fog_filter, mask=fog_mask)
        fog_filter[fog_filter[..., 3] == 0] = (0.0, 0.0, 0.0, 1.0)
        
        fog_filter = fog_filter.transpose((2, 1, 0))
        
        # Blend fog
        alpha = 0.3
        minimap = cv2.addWeighted(minimap, alpha, fog_filter, 1 - alpha, 0)
        
        return minimap
    
    def perform_overlay_operations(self, minimap, overlay_ops):
        """Apply all overlays to minimap"""
        minimap_hwc = minimap.transpose((1, 2, 0))  # Convert to H, W, C once
        
        for img, x, y in overlay_ops:
            # img is already in H, W, C format
            minimap_hwc = self.overlay_transparent(minimap_hwc, img, x, y)
        
        minimap = minimap_hwc.transpose((2, 0, 1))  # Convert back to C, H, W
        return minimap
    
    def convert_to_yolo_format(self, champion_data, img_width, img_height):
        """
        Convert bounding boxes to YOLO format
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]
        """
        yolo_labels = []
        
        for label, x, y, w, h in champion_data:
            class_id = self.champion_to_id[label]
            
            # Scale to final image size
            scaled_x = x * self.minimap_ratio[0]
            scaled_y = y * self.minimap_ratio[1]
            scaled_w = w * self.minimap_ratio[0]
            scaled_h = h * self.minimap_ratio[1]
            
            # Calculate center coordinates
            x_center = (scaled_x + scaled_w / 2) / img_width
            y_center = (scaled_y + scaled_h / 2) / img_height
            norm_w = scaled_w / img_width
            norm_h = scaled_h / img_height
            
            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            # Validate box
            if norm_w > 0 and norm_h > 0:
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        return yolo_labels
    
    def generate_sample(self, index):
        """
        Generate one training sample
        Returns: (image_array, yolo_labels_list)
        """
        # Create base minimap
        minimap = self.create_minimap_base()
        
        # Track overlay and fog operations
        overlay_ops = []
        fog_ops = []
        
        # REMOVED: towers, wards, minions - focus on champions only!
        
        # Create and place champions
        ally_champs, enemy_champs = self.create_champions()
        all_champions = ally_champs + enemy_champs
        champion_data = self.place_champions(minimap, all_champions, overlay_ops, fog_ops)
        
        # Apply fog operations
        minimap = self.perform_fog_operations(minimap, fog_ops)
        
        # Apply all overlays
        minimap = self.perform_overlay_operations(minimap, overlay_ops)
        
        # Resize to target size
        minimap = minimap.transpose((1, 2, 0))  # FIX: (C, H, W) -> (H, W, C)
        minimap = cv2.cvtColor(minimap, cv2.COLOR_RGBA2RGB)
        minimap = cv2.resize(minimap, self.resize, interpolation=cv2.INTER_AREA)
        
        # Convert from float [0, 1] to uint8 [0, 255]
        minimap = (minimap * 255).astype(np.uint8)
        
        # Convert to YOLO format
        yolo_labels = self.convert_to_yolo_format(champion_data, self.resize[0], self.resize[1])
        
        return minimap, yolo_labels


    def generate_dataset(self, output_dir, num_samples=1000, train_split=0.8, num_workers=None):
        """
        Generate full YOLO dataset with multiprocessing
        
        Args:
            output_dir: Root directory for dataset
            num_samples: Total number of samples to generate
            train_split: Fraction for training (rest goes to validation)
            num_workers: Number of parallel workers (None = auto-detect, 1 = sequential)
        """
        # Create directory structure
        train_img_dir = Path(output_dir) / 'images' / 'train'
        val_img_dir = Path(output_dir) / 'images' / 'val'
        train_label_dir = Path(output_dir) / 'labels' / 'train'
        val_label_dir = Path(output_dir) / 'labels' / 'val'
        
        for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        num_train = int(num_samples * train_split)
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
        
        print(f"\n{'='*60}")
        print(f"DATASET GENERATION")
        print(f"{'='*60}")
        print(f"Total samples: {num_samples:,}")
        print(f"  Train: {num_train:,}")
        print(f"  Val: {num_samples - num_train:,}")
        print(f"Workers: {num_workers} (CPU cores: {cpu_count()})")
        print(f"Mode: {'Parallel' if num_workers > 1 else 'Sequential'}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Use multiprocessing or sequential
        if num_workers > 1:
            self._generate_parallel(num_samples, num_train, num_workers,
                                   train_img_dir, val_img_dir, train_label_dir, val_label_dir)
        else:
            self._generate_sequential(num_samples, num_train,
                                     train_img_dir, val_img_dir, train_label_dir, val_label_dir)
        
        elapsed = time.time() - start_time
        
        # Create data.yaml for YOLO
        yaml_path = Path(output_dir) / 'data.yaml'
        abs_output_dir = Path(output_dir).absolute()
        with open(yaml_path, 'w') as f:
            f.write(f"path: {abs_output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(self.champion_icons)}\n")
            f.write("names:\n")
            for i in range(len(self.champion_icons)):
                f.write(f"  {i}: {self.id_to_champion[i]}\n")
        
        print(f"\n{'='*60}")
        print(f"✓ Dataset generated successfully!")
        print(f"✓ Location: {output_dir}")
        print(f"✓ Config: {yaml_path}")
        print(f"✓ Time: {elapsed:.1f}s ({num_samples/elapsed:.1f} samples/sec)")
        print(f"{'='*60}\n")
    
    def _generate_sequential(self, num_samples, num_train, train_img_dir, val_img_dir, 
                            train_label_dir, val_label_dir):
        """Sequential generation (fallback)"""
        print("Generating samples (sequential mode)...\n")
        
        for i in range(num_samples):
            img, labels = self.generate_sample(i)
            
            if i < num_train:
                img_path = train_img_dir / f"{i:06d}.jpg"
                label_path = train_label_dir / f"{i:06d}.txt"
            else:
                img_path = val_img_dir / f"{i:06d}.jpg"
                label_path = val_label_dir / f"{i:06d}.txt"
            
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            with open(label_path, 'w') as f:
                if labels:  # Only write if not empty
                    f.write('\n'.join(labels))
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
    
    def _generate_parallel(self, num_samples, num_train, num_workers,
                          train_img_dir, val_img_dir, train_label_dir, val_label_dir):
        """Parallel generation using multiprocessing"""
        print("Generating samples (parallel mode)...\n")
        print("⚠️  Note: Each worker loads assets independently (one-time cost)\n")
        
        # Create tasks
        tasks = []
        for i in range(num_samples):
            is_train = i < num_train
            tasks.append((i, is_train))
        
        # Create worker arguments
        worker_args = {
            'champion_icons_path': self.champion_icons_path,
            'minimap_path': self.minimap_path,
            'fog_path': self.fog_path,
            'misc_path': self.misc_path,
            'resize': self.resize,
            'train_img_dir': str(train_img_dir),
            'val_img_dir': str(val_img_dir),
            'train_label_dir': str(train_label_dir),
            'val_label_dir': str(val_label_dir)
        }
        
        # Process in parallel with progress tracking
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
            completed = 0
            for _ in pool.imap_unordered(_worker_generate_sample, tasks, chunksize=10):
                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{num_samples} ({completed/num_samples*100:.1f}%)")


# Global variables for worker processes
_worker_generator = None
_worker_args = None


def _init_worker(args):
    """Initialize worker process with a generator instance"""
    global _worker_generator, _worker_args
    _worker_args = args
    
    # Each worker creates ONE generator instance (reused for all tasks)
    _worker_generator = MinimapGenerator(
        champion_icons_path=args['champion_icons_path'],
        minimap_path=args['minimap_path'],
        fog_path=args['fog_path'],
        misc_path=args['misc_path'],
        resize=args['resize'],
        silent=True  # Suppress prints in workers
    )


def _worker_generate_sample(task):
    """
    Worker function for parallel generation
    Uses the pre-initialized generator from _init_worker
    """
    global _worker_generator, _worker_args
    
    index, is_train = task
    
    # IMPROVED: Better random seed to avoid collisions
    import os
    seed = (index * 997 + os.getpid()) % (2**32)  # Use process ID for uniqueness
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate sample using pre-loaded generator
    img, labels = _worker_generator.generate_sample(index)
    
    # Determine paths
    if is_train:
        img_path = Path(_worker_args['train_img_dir']) / f"{index:06d}.jpg"
        label_path = Path(_worker_args['train_label_dir']) / f"{index:06d}.txt"
    else:
        img_path = Path(_worker_args['val_img_dir']) / f"{index:06d}.jpg"
        label_path = Path(_worker_args['val_label_dir']) / f"{index:06d}.txt"
    
    # Save files
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    with open(label_path, 'w') as f:
        if labels:  # Only write if not empty
            f.write('\n'.join(labels))
    
    return index


if __name__ == "__main__":
    generator = MinimapGenerator(
        champion_icons_path=r'C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\league_icons\champions',
        minimap_path=r'C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\league_icons\minimap',
        fog_path=r'C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\league_icons\fog',
        misc_path=r'C:\Users\beaux\leagueminimapdetection\LeagueMinimapDetectionCNN\league_icons\misc',
        resize=(640, 640)
    )
    
    # For your 8-core CPU, you can try different worker counts:
    # num_workers=7  (default, leaves 1 core free)
    # num_workers=8  (use all cores for max speed)
    # num_workers=6  (more conservative, leaves room for OS)
    
    generator.generate_dataset(
        output_dir='yolo_dataset',
        num_samples=10000,
        train_split=0.8,
        num_workers=7  # Optimal for your 8-core CPU
    )
    