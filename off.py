#!/usr/bin/env python3
import rospy

import Jetson.GPIO as GPIO
out_pin = 7
GPIO.setmode(GPIO.BOARD)
import time

GPIO.setup(out_pin, GPIO.OUT, initial = GPIO.LOW)
GPIO.output(out_pin, GPIO.LOW)
# for i in range(10):
#     GPIO.output(out_pin, GPIO.HIGH)
#     time.sleep(1)
#     GPIO.output(out_pin, GPIO.LOW)
"""
for i in range(10):
    GPIO.output(out_pin, GPIO.HIGH)
    time.sleep(2.5)
    GPIO.output(out_pin, GPIO.LOW)
"""
