#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import cv2
from Camera_no_autofocus import Camera
from ModelManager import ModelManager
import csv
import time
import os
import numpy as np

class ComputerVisionNode:
    def __init__(self,model_name = "cascade",interval = 2):
        init_start = time.perf_counter()

        rospy.init_node('computer_vision_node', anonymous=True)
        rospy.loginfo("🟢 ComputerVisionNode initialized")

        spray_pub_init_start = time.perf_counter()
        self.spray_pub = rospy.Publisher('/spray_trigger', Bool, queue_size=1)

        # Initialize camera
        camera_init_start = time.perf_counter()
        self.camera = Camera(i2c_bus=10)

        self.save_dir = "/home/wenxin/Images/small/"
        self.interval = interval
        self.last_capture_time = time.perf_counter()

        # Initialize model
        model_init_start = time.perf_counter()
        self.model_name = model_name
        self.modelManager = None
        if self.model_name != "cascade":
            self.modelManager = ModelManager(model_name=model_name)
        self.model = self.load_model()

        # Initialize csv logging
        csv_init_start = time.perf_counter()
        self.csv_init_file = open("/home/wenxin/inference_logs/init.csv",mode = "w",newline="")
        self.csv_init_writer = csv.writer(self.csv_init_file)
        self.csv_init_writer.writerow(["progress name","inference time"])
        self.csv_init_writer.writerow([">> init start",""])

        self.csv_frame_read = open("/home/wenxin/inference_logs/frame_read.csv",mode = "w",newline="")
        self.csv_decision_making = open("/home/wenxin/inference_logs/decision_making.csv",mode = "w",newline="")
        self.csv_run = open("/home/wenxin/inference_logs/run.csv",mode = "w",newline="")

        self.csv_frame_read_writer = csv.writer(self.csv_frame_read)
        self.csv_decision_making_writer = csv.writer(self.csv_decision_making)
        self.csv_run_writer = csv.writer(self.csv_run)

        self.csv_frame_read_writer.writerow(["frame #","inference time"])
        self.csv_decision_making_writer.writerow(["frame #","inference_ time"])
        self.csv_run_writer.writerow(["frame #","inference_ time"])

        print("Computer vision node initialization finished!")
        init_end = time.perf_counter()

        # log init inference time to csv
        infer_progress = ["--sprayer init",
                          "--camera",
                          "--model",
                          "--csv logger"]
        infer_times = [spray_pub_init_start,
                       camera_init_start,
                       model_init_start,
                       csv_init_start,
                       init_end]
        for i in range(len(infer_progress)):
            self.csv_init_writer.writerow([
                infer_progress[i],f"{(infer_times[i+1] - infer_times[i]):.4f}"
            ])

        self.csv_init_writer.writerow([">> init end",""])

    def load_model(self):
        if self.model_name == "cascade":
            return cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            )
        else:
            _,frame = self.camera.cap.read()
            self.modelManager.run_dummy(frame)
            return self.modelManager.model
    
    def run_model(self,frame):
        if self.model_name == "cascade":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return faces
        else:
            return self.modelManager.run_inference(frame)

    def save_frame(self,frame):
        today = time.strftime("%Y-%m-%d")
        daily_dir = os.path.join(self.save_dir,today)
        os.makedirs(daily_dir,exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(daily_dir,f"{timestamp}.png")
        cv2.imwrite(filename,frame)
        rospy.loginfo(f"📷 saved frame: {filename}")

    def infer_and_decide(self,frame):
        output = self.run_model(frame)
        # cv2.imshow("CSI Camera",frame)
        # cv2.waitKey(1)

        if self.model_name == "cascade":
            return len(output) > 0
        
        h,w = output.shape[:2]
        nh,nw = h//2,w//4
        output = output[0:nh//2,nw:3*nw]
        sum = np.sum(output)
        return sum > 0
    
    def run(self):
        rospy.loginfo("🚀 Computer Vision node started.")

        frame_count = 1
        max_frame_count = 50
        while not rospy.is_shutdown():
            frame_read_start = time.perf_counter()
            frame = self.camera.read()
            if frame_read_start- self.last_capture_time >= self.interval:
                self.save_frame(frame)
                self.last_capture_time = frame_read_start

            decision_making_start = time.perf_counter()
            decision = self.infer_and_decide(frame)

            spray_publish = time.perf_counter()
            self.spray_pub.publish(decision)

            rospy.loginfo(
                f"frame read: {(decision_making_start - frame_read_start):.4f}s | "
                f"decision made: {(spray_publish - decision_making_start):.4f}s | "
            )
            rospy.loginfo("Decision made: " + str(decision))

            run_end = time.perf_counter()

            if frame_count <= max_frame_count:
                self.csv_frame_read_writer.writerow([
                    frame_count,f"{(decision_making_start - frame_read_start):.4f}"
                ])
                self.csv_decision_making_writer.writerow([
                    frame_count,f"{(spray_publish - decision_making_start):.4f}"
                ])
                self.csv_run_writer.writerow([
                    frame_count,f"{(run_end - frame_read_start):.4f}"
                ])
                frame_count += 1
        
        self.camera.release()

        
if __name__ == '__main__':
    try:
        node = ComputerVisionNode("cascade")
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"❌ Node crashed: {e}")