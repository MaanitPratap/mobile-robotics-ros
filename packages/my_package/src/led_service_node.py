#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from my_package.srv import SetLEDColor, SetLEDColorResponse

class LEDServiceNode(DTROS):
    def __init__(self, node_name):
        super(LEDServiceNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        
        # Publisher for LEDPattern
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        
        # Service server
        self.srv = rospy.Service('~set_led_color', SetLEDColor, self.handle_set_color)
        
        # Color mappings for states
        self.colors = {
            'stop': 'red',      # State 1: Stop
            'moving': 'blue',   # State 2: D-pattern
            'return': 'green'   # State 3: Return
        }
        
        rospy.loginfo("LED Service is ready")

    def handle_set_color(self, req):
        color = req.color.lower()
        rospy.loginfo(f"Setting LED color to: {color}")
        
        # Create LED pattern message
        pattern = LEDPattern()
        pattern.header.stamp = rospy.Time.now()
        
        # Set RGB values based on requested color
        rgb = ColorRGBA()
        if color == 'red':
            rgb.r, rgb.g, rgb.b = 1.0, 0.0, 0.0
        elif color == 'blue':
            rgb.r, rgb.g, rgb.b = 0.0, 0.0, 1.0
        elif color == 'green':
            rgb.r, rgb.g, rgb.b = 0.0, 1.0, 0.0
        else:
            rgb.r, rgb.g, rgb.b = 0.0, 0.0, 0.0
        
        rgb.a = 1.0
        
        # DB21M has 5 LEDs
        pattern.rgb_vals = [rgb] * 5
        
        # Publish pattern
        self.led_pub.publish(pattern)
        
        return SetLEDColorResponse(
            success=True,
            message=f"LED color set to {color}"
        )

    def on_shutdown(self):
        """Turn off LEDs on shutdown"""
        pattern = LEDPattern()
        pattern.header.stamp = rospy.Time.now()
        rgb = ColorRGBA()
        rgb.a = 1.0
        pattern.rgb_vals = [rgb] * 5
        self.led_pub.publish(pattern)

if __name__ == '__main__':
    node = LEDServiceNode(node_name='led_service_node')
    rospy.spin()