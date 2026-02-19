#!/usr/bin/env python3
"""
Standalone P2PNet Webcam Stampede Detection
Runs P2PNet on webcam with real-time overlays: count, density, velocity, risk level.
Displays in OpenCV window using cv2.imshow.
Press 'q' to quit.
"""

import cv2
import numpy as np
from camera_p2pnet import VideoCamera

def main():
    # Initialize P2PNet VideoCamera for webcam
    camera = VideoCamera('')  # '' for webcam
    camera.running = False  # Disable MySQL logging for standalone mode

    print("[INFO] Starting P2PNet webcam stampede detection...")
    print("Press 'q' in the window to quit.")

    while True:
        # Get frame from camera (JPEG bytes)
        frame = camera.get_frame()

        # Decode JPEG to OpenCV image
        img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        # Display
        cv2.imshow('P2PNet Stampede Detection', img)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.__del__()  # Clean up
    cv2.destroyAllWindows()
    print("[INFO] Exited.")

if __name__ == '__main__':
    main()
