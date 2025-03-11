#!/usr/bin/env python3
import os
import rospy
import math
import re
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from std_msgs.msg import String
from led_service.srv import SetLEDColor, SetLEDColorRequest

class BehaviorController(DTROS):
    def __init__(self, node_name):
        super(BehaviorController, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self._vehicle_name = os.environ["VEHICLE_NAME"]


        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        
        self.WHEEL_RADIUS = 0.0318  # meters
        self.TICKS_PER_ROTATION = 135
        self.CURVE_SPEED = 0.7

        
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = 0
        self._ticks_right = 0
        rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # LED service
        self.led_service = None
        try:
            rospy.wait_for_service('set_led_color', timeout=5.0)
            self.led_service = rospy.ServiceProxy('set_led_color', SetLEDColor)
            rospy.loginfo("Connected to LED service")
        except rospy.ROSException:
            rospy.logwarn("LED service not available")

        # Subscribe to lane detections (e.g. "blue_lane:25.00,white_lane:60.00")
        self._lane_sub = rospy.Subscriber("lane_detections", String, self.lane_detection_callback)
        self.detected_distances = {}  # store parsed distances per color (in cm)

        # Track which behaviors have been executed
        self.blue_done = False
        self.red_done = False
        self.green_done = False

        # Main loop timer (~10 Hz)
        self.main_timer = rospy.Timer(rospy.Duration(0.1), self.main_loop)

        rospy.loginfo("BehaviorController initialized.")

  
    def callback_left(self, data):
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
        self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
        self._ticks_right = data.data - self._ticks_right_init

    def reset_encoders(self):
        """Reset encoders to zero so we can measure movement distance."""
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = 0
        self._ticks_right = 0
        rospy.sleep(0.5)  # wait a bit to ensure new ticks come in


    def set_led_pattern(self, color):
        """Use the LED service to set a color or 'off'."""
        if not self.led_service:
            return
        try:
            req = SetLEDColorRequest()
            req.color = color  # "blue", "red", "green", "off", etc.
            self.led_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"LED service call failed: {e}")

   
    def publish_velocity(self, left_vel, right_vel):
        cmd = WheelsCmdStamped()
        cmd.vel_left = left_vel
        cmd.vel_right = right_vel
        self._publisher.publish(cmd)

    def stop(self, duration=0):
        """Stop the robot for a specified duration."""
        self.publish_velocity(0, 0)
        if duration > 0:
            rospy.sleep(duration)

    
    def move_straight(self, distance_m, speed=0.5):
        """Move straight for a specified distance in meters using encoders."""
        rospy.loginfo(f"Moving straight {distance_m:.2f} m")
        self.reset_encoders()
        circumference = 2.0 * math.pi * self.WHEEL_RADIUS
        needed_ticks = (distance_m / circumference) * self.TICKS_PER_ROTATION

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            avg_ticks = (self._ticks_left + self._ticks_right) / 2.0
            if avg_ticks >= needed_ticks:
                break
            self.publish_velocity(speed, speed)
            rate.sleep()

        self.stop()

    def turn_right(self):
        """Simple 90° right turn using encoders and a fixed arc approach."""
        rospy.loginfo("Turning 90° to the right.")
        self.reset_encoders()
        arc_length = 0.4 * (math.pi / 2)  # radius 0.4, 90 deg turn
        needed_ticks = self.distance_to_ticks(arc_length)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            avg_ticks = (self._ticks_left + self._ticks_right) / 2.0
            if avg_ticks >= needed_ticks:
                break
            self.publish_velocity(self.CURVE_SPEED, self.CURVE_SPEED * 0.35)
            rate.sleep()
        self.stop()

    def turn_left(self):
        """Simple 90° left turn using encoders and a fixed arc approach."""
        rospy.loginfo("Turning 90° to the left.")
        self.reset_encoders()
        arc_length = 0.4 * (math.pi / 2)
        needed_ticks = self.distance_to_ticks(arc_length)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            avg_ticks = (self._ticks_left + self._ticks_right) / 2.0
            if avg_ticks >= needed_ticks:
                break
            self.publish_velocity(self.CURVE_SPEED * 0.35, self.CURVE_SPEED)
            rate.sleep()
        self.stop()

    def distance_to_ticks(self, dist_m):
        """Helper to convert meters to encoder ticks."""
        circumference = 2.0 * math.pi * self.WHEEL_RADIUS
        return (dist_m / circumference) * self.TICKS_PER_ROTATION

 
    def lane_detection_callback(self, msg):
        """
        Example message: "blue_lane:25.00,white_lane:60.00"
        Parse distances (in cm) for each color and store in self.detected_distances.
        """
        self.detected_distances.clear()
        data_str = msg.data.strip()
        if not data_str:
            return
        items = data_str.split(',')
        for item in items:
            match = re.match(r'(\w+)_lane:(\d+(\.\d+)?)', item)
            if match:
                color = match.group(1)
                dist_cm = float(match.group(2))
                self.detected_distances[color] = dist_cm

  
    def main_loop(self, event):
        """
        Main logic loop (~10 Hz):
          - Execute special behaviors when blue, red, or green lines are detected within threshold.
          - Yellow and white detections are ignored (treated as no detection).
          - If no special behavior is triggered, keep moving forward slowly.
        """
        executed_something = False

        # BLUE line behavior
        if ('blue' in self.detected_distances and self.detected_distances['blue'] <= 3 and not self.blue_done):
            rospy.loginfo("Blue line detected within threshold. Executing blue behavior.")
            self.blue_done = True
            executed_something = True
            self.stop(duration=4)
            self.set_led_pattern("blue")
            self.turn_right()
            self.set_led_pattern("off")

        # RED line behavior
        elif ('red' in self.detected_distances and self.detected_distances['red'] <= 3 and not self.red_done):
            rospy.loginfo("Red line detected within threshold. Executing red behavior.")
            self.red_done = True
            executed_something = True
            self.stop(duration=4)
            self.move_straight(0.3)

        # GREEN line behavior
        elif ('green' in self.detected_distances and self.detected_distances['green'] <= 3 and not self.green_done):
            rospy.loginfo("Green line detected within threshold. Executing green behavior.")
            self.green_done = True
            executed_something = True
            self.stop(duration=4)
            self.set_led_pattern("green")
            self.turn_left()
            self.set_led_pattern("off")

        # If no red, blue, or green detections, treat yellow/white as no detection and move forward slowly.
        if not executed_something:
            self.publish_velocity(0.1, 0.1)

    def on_shutdown(self):
        """Stop robot and turn off LEDs on shutdown."""
        self.stop()
        self.set_led_pattern("off")
        super().on_shutdown()

if __name__ == '__main__':
    node = BehaviorController(node_name='behavior_controller_node')
    rospy.spin()
