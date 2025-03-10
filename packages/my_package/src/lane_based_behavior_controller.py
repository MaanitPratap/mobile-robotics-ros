#!/usr/bin/env python3

# import required libraries
import os
import rospy
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA, String
from led_service.srv import SetLEDColor, SetLEDColorRequest, LaneDetect, LaneDetectResponse

class BehaviorController(DTROS):
    def __init__(self, node_name):
        super(BehaviorController, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Get Duckiebot's name
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        
        # Publisher for wheel commands
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        
        # Define parameters
        self.WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
        self.WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
        self.TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
        self.CURVE_SPEED = 0.5  # Base speed for curved movement
        
        # Encoder variables and subscription
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None
        
        # Subscribers to wheel encoders
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        
        # LED service client
        rospy.loginfo("Waiting for LED service...")
        try:
            rospy.wait_for_service('set_led_color', timeout=5.0)
            self.led_service = rospy.ServiceProxy('set_led_color', SetLEDColor)
            rospy.loginfo("Connected to LED service")
        except rospy.ROSException:
            self.led_service = None
            rospy.logwarn("LED service not available, continuing without LED control")
            
        
        # State variables
        self.current_color = None
        self.is_executing = False
        
        # Wait for encoders to initialize
        rospy.sleep(2)
        self.behavior_service = rospy.Service('behavior_service', LaneDetect, self.callback)
        rospy.loginfo("Behavior controller initialized and service ready")
        
        
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
        
    def set_led_pattern(self, color):
        """Set LED pattern using LED service"""
        if self.led_service is None:
            rospy.logwarn("LED service not available, skipping LED control")
            return False
            
        try:
            req = SetLEDColorRequest()
            req.color = color
            response = self.led_service(req)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"LED service call failed: {e}")
            return False
            
    def publish_velocity(self, left_vel, right_vel):
        """Publish wheel velocities."""
        cmd = WheelsCmdStamped(vel_left=left_vel, vel_right=right_vel)
        self._publisher.publish(cmd)
        
    def stop(self, duration=0):
        """Stop the robot for a specified duration."""
        rospy.loginfo(f"Stopping for {duration} seconds...")
        self.publish_velocity(0, 0)
        rospy.sleep(duration)
        
    def move_straight(self, distance):
        """Move in a straight line for the specified distance."""
        rospy.loginfo(f"Moving straight for {distance} meters...")
        self.reset_encoders()

        # Compute required encoder ticks for the distance
        ticks_needed = (distance / (2 * math.pi * self.WHEEL_RADIUS)) * self.TICKS_PER_ROTATION

        # Command wheels to move forward
        self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)
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
        self.stop()
        
    def turn_right(self):
        """Turn 90 degrees to the right."""
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
        """Turn 90 degrees to the left."""
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
        
    def execute_blue_line_behavior(self):
        """
        Behavior for blue line:
        1. Set right side LEDs to blue
        2. Stop for 3-5 seconds
        3. Move in a curve through 90 degrees to the right
        """
        self.is_executing = True
        rospy.loginfo("Executing blue line behavior")
        
        # Set right side LEDs to blue
        self.set_led_pattern("blue")  # Using the enhanced LED service for right signaling
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        # Turn right 90 degrees
        self.turn_right()
        
        # Turn off LEDs
        self.set_led_pattern("off")
        
        self.is_executing = False
        
    def execute_red_line_behavior(self):
        """
        Behavior for red line:
        1. Set LEDs to red
        2. Stop for 3-5 seconds
        3. Move straight for at least 30 cm
        """
        self.is_executing = True
        rospy.loginfo("Executing red line behavior")
        
        # Set LEDs to red
        self.set_led_pattern("red")
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        # Move straight for 50 cm (greater than required 30 cm)
        self.move_straight(0.5)
        
        # Turn off LEDs
        self.set_led_pattern("off")
        
        self.is_executing = False
        
    def execute_green_line_behavior(self):
        """
        Behavior for green line:
        1. Set left side LEDs to green
        2. Stop for 3-5 seconds
        3. Move in a curve through 90 degrees to the left
        """
        self.is_executing = True
        rospy.loginfo("Executing green line behavior")
        
        # Set left side LEDs to green
        self.set_led_pattern("green")  # Using the enhanced LED service for left signaling
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        # Turn left 90 degrees
        self.turn_left()
        
        # Turn off LEDs
        self.set_led_pattern("off")
        
        self.is_executing = False
        
    def callback(self, req):
        """Handle lane detection callback"""
        if self.is_executing:
            rospy.loginfo(f"Already executing a behavior, ignoring {req.cmd}")
            return LaneDetectResponse(True)
            
        detected_color = req.cmd
        rospy.loginfo(f"Detected color: {detected_color}")

        if detected_color == "blue":
            self.execute_blue_line_behavior()
        elif detected_color == "red":
            self.execute_red_line_behavior()
        elif detected_color == "green":
            self.execute_green_line_behavior()
        elif detected_color == "shutdown":
            rospy.loginfo("Received shutdown command")
            self.set_led_pattern("off")
            self.stop()
            rospy.signal_shutdown("Task completed")
        
        return LaneDetectResponse(True)
        
    def on_shutdown(self):
        """Handle shutdown cleanly"""
        self.set_led_pattern("off")
        self.stop()
        rospy.loginfo("Behavior controller shutting down")

if __name__ == '__main__':
    node = BehaviorController(node_name='behavior_controller_node')
    rospy.spin()