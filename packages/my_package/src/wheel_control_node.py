#!/usr/bin/env python3

import os
import rospy
import rosbag
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from nav_msgs.msg import Odometry
import time

WHEEL_SPEED = 0.2
DISTANCE = 1.25

TIME_TO_TRAVEL = DISTANCE / WHEEL_SPEED

ROTATION_SPEED = 0.2  # Speed for rotation
ROTATION_SPEED_RIGHT = 0.15
ROTATION_DURATION = 1

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)


    def move(self, vel_left, vel_right, duration, direction):
        message = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        start_time = time.time()
        current_distance = 0
        while time.time() - start_time < duration and not rospy.is_shutdown():
            self._publisher.publish(message)
            time_elapsed = time.time() - start_time
            current_distance = time_elapsed * WHEEL_SPEED
            rospy.loginfo(f"Driving {direction}: traveled {current_distance:.2f}/{DISTANCE:.2f} meters")
            rospy.sleep(0.1)
        self.stop()

    def stop(self):
        stop_message = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_message)
        rospy.sleep(0.1)

    def run(self):
        rospy.sleep(0.1) 
        self.move(WHEEL_SPEED, WHEEL_SPEED, TIME_TO_TRAVEL, "forward")
        rospy.sleep(0.1) 
        self.move(-WHEEL_SPEED, -WHEEL_SPEED, TIME_TO_TRAVEL, "backward")

        rospy.sleep(0.1)
        self.move(ROTATION_SPEED, -ROTATION_SPEED_RIGHT, ROTATION_DURATION, "clockwise")
        rospy.sleep(0.1)
        self.move(-ROTATION_SPEED, ROTATION_SPEED, ROTATION_DURATION, "counterclockwise")
        # self.bag.close()

if __name__ == '__main__':
    node = WheelControlNode(node_name='wheel_control_node')
    node.run()