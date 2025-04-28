#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import Float64
#from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int8 #Float64
from std_msgs.msg import String

from geometry_msgs.msg import Vector3Stamped 


from sensor_msgs.msg import NavSatFix

import message_filters

import pandas as pd
import csv
import os
import Jetson.GPIO as GPIO
out_pin = 7
GPIO.setmode(GPIO.BOARD)
import time

GPIO.setup(out_pin, GPIO.OUT, initial = GPIO.LOW)


"""
fl = pd.read_csv("/home/shi/catkin_ws/src/Onboard-SDK-ROS-3.8/dji_sdk/litchi.csv", header=None)
#f2=fl.drop(fl.index[0])
wp_list=[]#MultiArrayDimension()


num_wp =len(fl)
for i in range(num_wp):
    wp=(fl[0][i], fl[1][i]) #float
    wp_list.append(wp)
    #wp_list.append(fl[1][i])

#wpl=[40.84,-96.99]
#stri = str(wp_list)

num_wp = 4

def target_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "%f", data.data)

def vel_callback(data):

    rospy.loginfo("altitude: %f", data)
    if (data.vector < 0.01):
        rospy.loginfo("x velo")
        #os.system("rosrun dji_sdk nozzle.py")


def gps_callback(data):

    rospy.loginfo("altitude: %f", data.altitude)
    if ((data.altitude >= 1.9) and (data.altitude <= 2.1)):
        rospy.loginfo("spraying from 2 m")
        os.system("rosrun dji_sdk nozzle.py")


        
    #rospy.loginfo("gps_callback")
    ###ggg=data.data

"""



def v3s_callback(v3data, navdata):
    rospy.loginfo("x vect: %f, y vect: %f, z vect: %f, alt: %f", v3data.vector.x, v3data.vector.y, v3data.vector.z, navdata.altitude)
    if ((navdata.altitude > 0.5) and (navdata.altitude < 342.7)):
        rospy.loginfo("stable, spraying...")
        GPIO.output(out_pin, GPIO.HIGH)
        #os.system("rosrun dji_sdk nozzle.py")
        time.sleep(0.1)    
        GPIO.output(out_pin, GPIO.LOW)
"""
(abs(v3data.vector.x) < 0.01) and (abs(v3data.vector.y) < 0.01) and (abs(v3data.vector.z) < 0.01) and
if ((navdata.altitude > 0.5) and (navdata.altitude < 342.7)):
        rospy.loginfo("Spraying from 1 m")
        os.system("rosrun dji_sdk nozzle.py")
"""    


def gps_listener():
    rospy.init_node('gps_listener', anonymous=True)
    rate = rospy.Rate(0.5)
    
    #rospy.loginfo(rospy.get_caller_id() + "%f", ggg)
    while not rospy.is_shutdown():
        navsub = message_filters.Subscriber('/dji_sdk/gps_position', NavSatFix)
        
        v3sub = message_filters.Subscriber('/dji_sdk/velocity', Vector3Stamped) 

        cb = message_filters.ApproximateTimeSynchronizer([v3sub, navsub], queue_size = 10, slop = 0.1)
        cb.registerCallback(v3s_callback)
        #rate.sleep()
        rospy.spin()
 



"""
##rospy.Subscriber('/dji_sdk/velocity', Vector3.x, vel_callback)

#if (aaa == 2):

   #gps_lats = rospy.Subscriber('/dji_sdk/gps_position/latitude', Float64, gps_callback)
    #gps = rospy.Subscriber('/dji_sdk/gps_position', Float64, gps_callback)
    
    #nums = rospy.Subscriber('tnum', Int8, queue_size=10)
    #while not rospy.is_shutdown():

    with open('/home/shi/catkin_ws/src/Onboard-SDK-ROS-3.8/dji_sdk/gps.csv', mode = 'w') as csvfile:
        fieldnames = ['lon', 'lat']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'lon':gps_lons,'lat':gps_lons})

DID NOT WORK !!
    if (((data.latitude >= 40.832244) and (data.latitude <= 40.832245)) and ((data.longitude >= -96.669803) and (data.longitude <= -96.669804))):
        rospy.loginfo("spraying from point 1")
        os.system("rosrun dji_sdk nozzle.py")

    elif (((data.latitude >= 40.832245) and (data.latitude <= 40.832246)) and ((data.longitude >= -96.669912) and (data.longitude <= -96.669913))):
        rospy.loginfo("spraying from point 2")
        os.system("rosrun dji_sdk nozzle.py")




    if ((gps_lats==40.8320) and (gps_lons==-96.6697)):
        rospy.loginfo("point match")
        GPIO.output(out_pin, GPIO.HIGH)
        time.sleep(2.5)
        GPIO.output(out_pin, GPIO.LOW)

    
#for i in range(num_wp): 

#if(((rospy.Subscriber('tlat'+str(i), Float64, target_callback))==gps_lats) and ((rospy.Subscriber('tlon'+str(i), Float64, target_callback))==gps_lons)):
       

        
    

        rate = rospy.Rate(1) # 10hz
    
        for i in range(num_wp):
        #flt = 46.08 ###
        #"hello world %s" % rospy.get_time()
                #rospy.loginfo(num_wp) #(wp_list)
                #rospy.loginfo(wp_list[i][0])
                #rospy.loginfo(wp_list[i][1])
                nump.publish(num_wp) #(wp_list)        
                locals()['latp'+str(i)].publish(wp_list[i][0])
                locals()['lonp'+str(i)].publish(wp_list[i][1])
    

        rate.sleep()
"""
if __name__ == '__main__':
    try:
        gps_listener()
    except rospy.ROSInterruptException:
        pass
