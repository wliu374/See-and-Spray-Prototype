#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
import time

class SprayTriggerPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('spray_trigger_publisher', anonymous=True)

        # Publisher to /spray_trigger
        self.pub = rospy.Publisher('/spray_trigger', Bool, queue_size=1)

        # Wait for publisher to register
        rospy.sleep(0.5)

    def run(self):
	
        # Publish True
        for i in range(5):
            rospy.loginfo("Publishing: True")
            self.pub.publish(Bool(data=True))

        # Wait for 1 second
            rospy.sleep(2.0)

        # Publish False
            rospy.loginfo("Publishing: False")
            self.pub.publish(Bool(data=False))

        # Wait for 1 second
            rospy.sleep(2.0)

if __name__ == '__main__':
    try:
        node = SprayTriggerPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
