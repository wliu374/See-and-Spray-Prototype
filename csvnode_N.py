#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import Float64
#from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int8 #Float64
import pandas as pd

fl = pd.read_csv("/home/wenxin/catkin_ws/src/Onboard-SDK-ROS/dji_sdk/bigN.csv", header=None)
#f2=fl.drop(fl.index[0])
wp_list=[]#MultiArrayDimension()


num_wp =len(fl)
for i in range(num_wp):
    wp=(fl[0][i], fl[1][i]) #float
    wp_list.append(wp)
    #wp_list.append(fl[1][i])

#wpl=[40.84,-96.99]
#stri = str(wp_list)

def talker():
    nump = rospy.Publisher('tnum', Int8, queue_size=10)
    for i in range(num_wp):
        locals()['latp'+str(i)] = rospy.Publisher('tlat'+str(i), Float64, queue_size=10)
        locals()['lonp'+str(i)] = rospy.Publisher('tlon'+str(i), Float64, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        for i in range(num_wp):
        #flt = 46.08 ###
        #"hello world %s" % rospy.get_time()
                #rospy.loginfo(num_wp) #(wp_list)
                #rospy.loginfo(wp_list[i][0])
                #rospy.loginfo(wp_list[i][1])
                nump.publish(num_wp) #(wp_list)        
                locals()['latp'+str(i)].publish(wp_list[i][0])
                locals()['lonp'+str(i)].publish(wp_list[i][1])
    
        #rospy.spin()
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
