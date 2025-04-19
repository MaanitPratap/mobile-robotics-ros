#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from led_service.srv import SetLEDColor, SetLEDColorResponse

class LEDServiceNode(DTROS):
    def __init__(self, node_name):
        super(LEDServiceNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.vehicle_name = os.environ['VEHICLE_NAME']
  
        self.led_publisher = rospy.Publisher(
            f'/{self.vehicle_name}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1
        )
        
        # Color definitions with both names and RGB values
        self.colors = {
            'off':    {'name': 'off',    'rgb': ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)},
            'red':    {'name': 'red',    'rgb': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)},
            'green':  {'name': 'green',  'rgb': ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)},
            'blue':   {'name': 'blue',   'rgb': ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)},
            'white':  {'name': 'white',  'rgb': ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)},
            'yellow': {'name': 'yellow', 'rgb': ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)},
            'purple': {'name': 'purple', 'rgb': ColorRGBA(r=0.5, g=0.0, b=0.5, a=1.0)},
            'cyan':   {'name': 'cyan',   'rgb': ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)},
            'orange': {'name': 'orange', 'rgb': ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)}
        }
        
        # LED indices - Duckiebot has 5 LEDs:
        # 0: front-left, 4: front-right, 3: back-left, 1: back-right, 2: center
        self.led_indices = {
            'front_left': 0,
            'front_right': 4,
            'back_left': 3,
            'back_right': 1,
            'center': 2
        }
        
        # LED patterns for specific behaviors
        self.patterns = {
            # Standard patterns (all LEDs same color)
            'off': self._create_all_pattern('off'),
            'red': self._create_all_pattern('red'),
            'green': self._create_all_pattern('green'),
            'blue': self._create_all_pattern('blue'),
            'white': self._create_all_pattern('white'),
            'yellow': self._create_all_pattern('yellow'),
            'purple': self._create_all_pattern('purple'),
            'cyan': self._create_all_pattern('cyan'),
            'orange': self._create_all_pattern('orange'),
            
            # Special patterns for specific behaviors
            'following': self._create_special_pattern('yellow', 'following'),
            'searching': self._create_special_pattern('green', 'searching'),
            'warning': self._create_special_pattern('red', 'warning')
        }
        
        self.srv = rospy.Service('set_led_color', SetLEDColor, self.handle_set_led_color)
        
        rospy.loginfo("LED service node initialized")
        
    def _create_all_pattern(self, color_name):
        """Create a pattern with all LEDs set to the same color"""
        if color_name not in self.colors:
            rospy.logwarn(f"Color {color_name} not defined, using 'off'")
            color_name = 'off'
            
        color_data = self.colors[color_name]
        pattern = LEDPattern()
        pattern.color_list = [color_data['name']] * 5
        pattern.rgb_vals = [color_data['rgb']] * 5
        pattern.frequency = 0.0
        pattern.frequency_mask = [0] * 5  # No blinking
        pattern.color_mask = [1] * 5      # All LEDs on
        return pattern
    
    def _create_special_pattern(self, color_name, pattern_type):
        """Create special patterns for specific behaviors"""
        if color_name not in self.colors:
            rospy.logwarn(f"Color {color_name} not defined, using 'white'")
            color_name = 'white'
            
        color_data = self.colors[color_name]
        
        pattern = LEDPattern()
        
        if pattern_type == 'following':
            # Following pattern: all LEDs yellow, with front ones blinking
            pattern.color_list = [color_data['name']] * 5
            pattern.rgb_vals = [color_data['rgb']] * 5
            pattern.frequency = 2.0  # 2 Hz blinking
            # Only front LEDs blink
            pattern.frequency_mask = [1, 0, 0, 0, 1]  # Front LEDs blink
            pattern.color_mask = [1] * 5  # All LEDs on
            
        elif pattern_type == 'searching':
            # Searching pattern: green with center LED blinking
            pattern.color_list = [color_data['name']] * 5
            pattern.rgb_vals = [color_data['rgb']] * 5
            pattern.frequency = 1.0  # 1 Hz blinking
            # Only center LED blinks
            pattern.frequency_mask = [0, 0, 1, 0, 0]  # Center LED blinks
            pattern.color_mask = [1] * 5  # All LEDs on
            
        elif pattern_type == 'warning':
            # Warning pattern: all red LEDs blinking
            pattern.color_list = [color_data['name']] * 5
            pattern.rgb_vals = [color_data['rgb']] * 5
            pattern.frequency = 4.0  # 4 Hz blinking (faster for warning)
            pattern.frequency_mask = [1] * 5  # All LEDs blink
            pattern.color_mask = [1] * 5  # All LEDs on
            
        else:
            # Default to all color without blinking
            return self._create_all_pattern(color_name)
            
        return pattern

    def handle_set_led_color(self, req):
        """Handle SetLEDColor service requests"""
        pattern = None
        
        # Check if the requested color matches a pattern
        if req.color in self.patterns:
            pattern = self.patterns[req.color]
            rospy.loginfo(f"Using pattern for {req.color}")
        else:
            rospy.logwarn(f"Requested color/pattern {req.color} not found. Using 'off'")
            pattern = self.patterns['off']
   
        try:
            self.led_publisher.publish(pattern)
            rospy.loginfo(f"Published LED pattern for: {req.color}")
            return SetLEDColorResponse(success=True)
        except Exception as e:
            rospy.logerr(f"Failed to publish LED pattern: {e}")
            return SetLEDColorResponse(success=False)

    def on_shutdown(self):
        """Turn off all LEDs when shutting down"""
        self.led_publisher.publish(self.patterns['off'])
        rospy.loginfo("LED service shutting down, all LEDs turned off")

if __name__ == '__main__':
    node = LEDServiceNode(node_name='led_service_node')
    rospy.spin()