#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import Jetson.GPIO as GPIO
import os
import csv
from datetime import datetime
import time

class SprayerNode:
    def __init__(self):
        # ROS setup
        rospy.init_node('sprayer_node', anonymous=True)
        rospy.loginfo("🟢 SprayerNode initialized")

        # GPIO setup
        self.gpio_pin = 7  # Adjust this based on your wiring
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.gpio_pin, GPIO.OUT,initial=GPIO.LOW)
        self.gpio_status = 0
        self.max_frame_count = 50
        self.cnt_frame = 0

        # Optional logging
        self.csv_path = "/home/wenxin/spray_logs/spray_log.csv"  # Change to your path
        self.csv_reception_path = "/home/wenxin/inference_logs/receptive_time.csv"
        self.ensure_log_file()

        # ROS subscriber
        self.subscriber = rospy.Subscriber('/spray_trigger', Bool, self.spray_callback)
        rospy.on_shutdown(self.cleanup)

    def ensure_log_file(self):
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'spray_status'])
        with open(self.csv_reception_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp'])

    def log_status(self, status):
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), status])

    def spray_callback(self, msg):
        if msg.data:
            if not self.gpio_status:
                self.log_status(msg.data)
            GPIO.output(self.gpio_pin, GPIO.HIGH)
            self.gpio_status = 1
            rospy.loginfo("🚿 Nozzle ON")
        else:
            if self.gpio_status:
                self.log_status(msg.data)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            self.gpio_status = 0
            rospy.loginfo("❌ Nozzle OFF")
        if self.cnt_frame < self.max_frame_count:
            with open(self.csv_reception_path,mode = "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([time.perf_counter()])
            self.cnt_frame += 1

    def cleanup(self):
        rospy.loginfo("🧹 Cleaning up GPIO")
        GPIO.output(self.gpio_pin, GPIO.LOW)
        GPIO.cleanup()

    def run(self):
        rospy.loginfo("📡 SprayerNode running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SprayerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
