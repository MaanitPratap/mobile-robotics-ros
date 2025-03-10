#!/usr/bin/env python3

# import required libraries
import os
import rospy
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped

class NavigationControl(DTROS):
    def __init__(self, node_name):
        super(NavigationControl, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        # add your code here
        
        # Get Duckiebot's name
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        
        # publisher for wheel commands
        # NOTE: you can directly publish to wheel chassis using the car_cmd_switch_node topic in this assignment (check documentation)
        # you can also use your exercise 2 code
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        
        # robot params
        self.WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
        self.WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
        self.TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
        self.CURVE_SPEED = 0.5  # Base speed for curved movement

        # define other variables as needed
        # Encoder topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        # Encoder tick tracking
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        # Subscribers to wheel encoders
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Wait for encoders to initialize
        rospy.sleep(2)
    
    def callback_left(self, data):
        """Callback for left encoder ticks."""
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
            self._ticks_left = 0
        else:
            self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        """Callback for right encoder ticks."""
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
            self._ticks_right = 0
        else:
            self._ticks_right = data.data - self._ticks_right_init

    def reset_encoders(self):
        """Reset encoder counters to track new movements."""
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None
        rospy.loginfo("Initializing encoder tracking...")
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                break
            rate.sleep()
        rospy.loginfo("Encoder initialization complete.")
        
    def publish_velocity(self, left_vel, right_vel):
        # add your code here
        """
        Publish wheel velocities to move the Duckiebot.
        :param left_vel: Left wheel velocity.
        :param right_vel: Right wheel velocity.
        """
        cmd = WheelsCmdStamped(vel_left=left_vel, vel_right=right_vel)
        self._publisher.publish(cmd)
        
    def stop(self, duration=0):
        # add your code here
        """
        Stop the Duckiebot for a specified duration.
        :param duration: Duration to stop (in seconds).
        """
        rospy.loginfo(f"Stopping for {duration} seconds...")
        self.publish_velocity(0, 0)
        rospy.sleep(duration)
        
    def move_straight(self, distance):
        # add your code here
        """
        Move the Duckiebot in a straight line for a specified distance.
        :param distance: Distance to move (in meters).
        """
        rospy.loginfo(f"Moving straight for {distance} meters...")
        self.reset_encoders()

        # Compute required encoder ticks for the distance
        ticks_needed = (distance / (2 * math.pi * self.WHEEL_RADIUS)) * self.TICKS_PER_ROTATION

        # Command wheels to move forward
        self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 100 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("Forward movement completed.")
                    break

            self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED)
            rate.sleep()

        # Stop the robot
        self.stop(5)
        
    def turn_right(self):
        # add your code here
        """
        Move the Duckiebot in a curve through 90 degrees to the right.
        """
        rospy.loginfo("Turning 90 degrees to the right...")
        self.reset_encoders()

        # Define curve radius and calculate arc length
        curve_radius = 0.4  # meters
        arc_length = (math.pi / 2) * curve_radius  # Arc length for 90 degrees

        # Compute required encoder ticks for the arc length
        ticks_needed = (arc_length / (2 * math.pi * self.WHEEL_RADIUS)) * self.TICKS_PER_ROTATION

        # Command wheels to move in a curve (right wheel slower)
        self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED * 0.35)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("Right turn completed.")
                    break

            self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED * 0.35)
            rate.sleep()

        # Stop the robot
        self.stop()
        
    def turn_left(self):
        # add your code here
        """
        Move the Duckiebot in a curve through 90 degrees to the left.
        """
        rospy.loginfo("Turning 90 degrees to the left...")
        self.reset_encoders()

        # Define curve radius and calculate arc length
        curve_radius = 0.4  # meters
        arc_length = (math.pi / 2) * curve_radius  # Arc length for 90 degrees

        # Compute required encoder ticks for the arc length
        ticks_needed = (arc_length / (2 * math.pi * self.WHEEL_RADIUS)) * self.TICKS_PER_ROTATION

        # Command wheels to move in a curve (left wheel slower)
        self.publish_velocity(self.CURVE_SPEED * 0.3, self.CURVE_SPEED)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("Left turn completed.")
                    break

            self.publish_velocity(self.CURVE_SPEED * 0.3, self.CURVE_SPEED)
            rate.sleep()

        # Stop the robot
        self.stop()

    # add other functions as needed
    def execute_path(self):
        """
        Execute a predefined path for testing
        """
        # Move forward 10cm
        self.move_straight(0.3)
        
        # Turn right 90 degrees
        self.turn_right()
        
        # Stop for 3 seconds
        self.stop(duration=5)
        
        # Turn left 90 degrees
        self.turn_left()

if __name__ == '__main__':
    node = NavigationControl(node_name='navigation_control_node')
    # Uncomment the following line to test the robot's movement
    node.execute_path()
    rospy.spin()