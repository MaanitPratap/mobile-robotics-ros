#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import Float32, BoolStamped
from std_msgs.msg import String

class FollowerLEDIndicator(DTROS):
    def __init__(self, node_name):
        super(FollowerLEDIndicator, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Initialize vehicle name
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        # LED publisher
        self.led_pub = rospy.Publisher(
            f'/{self._vehicle_name}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1
        )
        
        # Subscribe to distance information
        self.distance_sub = rospy.Subscriber(
            f'/{self._vehicle_name}/duckiebot_follower/distance',
            Float32,
            self.distance_callback,
            queue_size=1
        )
        
        # Subscribe to detection information
        self.detection_sub = rospy.Subscriber(
            f'/{self._vehicle_name}/duckiebot_follower/detection',
            BoolStamped,
            self.detection_callback,
            queue_size=1
        )
        
        # LED patterns
        self.colors = {
            'off': {'r': 0.0, 'g': 0.0, 'b': 0.0},
            'red': {'r': 1.0, 'g': 0.0, 'b': 0.0},
            'green': {'r': 0.0, 'g': 1.0, 'b': 0.0},
            'blue': {'r': 0.0, 'g': 0.0, 'b': 1.0},
            'yellow': {'r': 1.0, 'g': 1.0, 'b': 0.0}
        }
        
        # Distance thresholds
        self.min_safe_distance = 0.2
        self.target_distance = 0.4
        
        # Initial state
        self.duckiebot_detected = False
        self.current_distance = float('inf')
        
        # Start LED update timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update_leds)
        
        self.log("LED indicator node initialized")
    
    def detection_callback(self, msg):
        """
        Handle detection status updates
        """
        self.duckiebot_detected = msg.data
    
    def distance_callback(self, msg):
        """
        Handle distance updates
        """
        self.current_distance = msg.data
    
    def create_led_pattern(self, color_name):
        """
        Create LED pattern message with the specified color
        """
        if color_name not in self.colors:
            color_name = 'off'
        
        pattern = LEDPattern()
        pattern.color_list = [color_name] * 5  # All 5 LEDs
        
        # Set RGB values
        color = self.colors[color_name]
        for i in range(5):
            rgb = pattern.rgb_vals.append(color['r'], color['g'], color['b'], 1.0)
        
        # No blinking
        pattern.frequency = 0.0
        pattern.frequency_mask = [0] * 5
        
        # All LEDs on
        pattern.color_mask = [1] * 5
        
        return pattern
    
    def update_leds(self, event):
        """
        Update LEDs based on current state
        """
        if not self.duckiebot_detected:
            # No Duckiebot detected - off
            pattern = self.create_led_pattern('off')
        elif self.current_distance < self.min_safe_distance:
            # Too close - red
            pattern = self.create_led_pattern('red')
        elif abs(self.current_distance - self.target_distance) < 0.1:
            # At target distance - green
            pattern = self.create_led_pattern('green')
        elif self.current_distance > self.target_distance:
            # Too far - blue
            pattern = self.create_led_pattern('blue')
        else:
            # Between min_safe and target - yellow
            pattern = self.create_led_pattern('yellow')
        
        # Publish LED pattern
        self.led_pub.publish(pattern)
    
    def on_shutdown(self):
        """
        Turn off LEDs when shutting down
        """
        pattern = self.create_led_pattern('off')
        self.led_pub.publish(pattern)
        rospy.sleep(0.1)
        self.log("LED indicator node shutting down")

if __name__ == '__main__':
    # Initialize the node
    node = FollowerLEDIndicator(node_name='follower_led_indicator')
    # Keep it spinning
    rospy.spin()