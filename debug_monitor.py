from mss import mss
import numpy as np
import cv2

with mss() as sct:
    print("Monitors list:", sct.monitors)
    for i, m in enumerate(sct.monitors):
        print(f"Monitor {i}: {m}")
        img = np.array(sct.grab(m))[:, :, :3]  # take RGB
        cv2.imwrite(f"monitor_{i}.png", img)
        print(f"Saved monitor_{i}.png")
