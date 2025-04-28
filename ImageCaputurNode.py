#!/usr/bin/env python3

import cv2
import os
import time
from Camera import Camera

class ImageCaptureNode:
    def __init__(self, camera_id=0, save_dir="/home/wenxin/Images/original", capture_interval=2.0):
        # Initialize camera
        self.camera = Camera(i2c_bus=10,sensor_id = 0,
                 capture_width=3840, capture_height=2160, 
                 display_width=3840,display_height=2160,
                 flip_method=0, framerate=30)
        if not self.camera.finish_focus:
            self.camera.autofocus()
        
        self.save_dir = save_dir
        self.capture_interval = capture_interval
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        today = time.strftime("%Y-%m-%d")
        daily_dir = os.path.join(self.save_dir,today)
        os.makedirs(daily_dir,exist_ok=True)

        last_capture_time = time.time()
        print("here")
        while True:
            frame = self.camera.read()
            current_time = time.time()
            if current_time - last_capture_time >= self.capture_interval:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(daily_dir, f"{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"📷 Saved {filename}")
                last_capture_time = current_time

            # Show camera feed (optional)
            cv2.imshow('Camera View', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()


if __name__ == "__main__":
    try:
        capturer = ImageCaptureNode()
        capturer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
