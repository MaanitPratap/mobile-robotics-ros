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
            'yellow': {'name': 'yellow', 'rgb': ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)}
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
            
            # Special patterns for specific behaviors
            'right_signal': self._create_side_pattern('blue', 'right'),  # For blue line
            'left_signal': self._create_side_pattern('green', 'left')    # For green line
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
        
    def _create_side_pattern(self, color_name, side):
        """Create a pattern with only left or right side LEDs lit"""
        if color_name not in self.colors:
            rospy.logwarn(f"Color {color_name} not defined, using 'white'")
            color_name = 'white'
            
        color_data = self.colors[color_name]
        off_data = self.colors['off']
        
        pattern = LEDPattern()
        # Default all LEDs to off
        pattern.color_list = [off_data['name']] * 5
        pattern.rgb_vals = [off_data['rgb']] * 5
        pattern.frequency = 0.0
        pattern.frequency_mask = [0] * 5  # No blinking
        pattern.color_mask = [1] * 5      # All LEDs considered
        
        # Set specific LEDs based on side
        if side == 'right':
            # Set front-right and back-right LEDs
            idx_front = self.led_indices['front_right']
            idx_back = self.led_indices['back_right']
            pattern.color_list[idx_front] = color_data['name']
            pattern.color_list[idx_back] = color_data['name']
            pattern.rgb_vals[idx_front] = color_data['rgb']
            pattern.rgb_vals[idx_back] = color_data['rgb']
        elif side == 'left':
            # Set front-left and back-left LEDs
            idx_front = self.led_indices['front_left']
            idx_back = self.led_indices['back_left']
            pattern.color_list[idx_front] = color_data['name']
            pattern.color_list[idx_back] = color_data['name']
            pattern.rgb_vals[idx_front] = color_data['rgb']
            pattern.rgb_vals[idx_back] = color_data['rgb']
        
        return pattern

    def handle_set_led_color(self, req):
        """Handle SetLEDColor service requests"""
        pattern = None
        
        # Check if the requested color matches a standard pattern
        if req.color in self.patterns:
            if req.color == 'blue':
                pattern = self.patterns['right_signal']
                rospy.loginfo("Using right signal pattern for blue line")
            # For green line, use left side signal pattern
            elif req.color == 'green':
                pattern = self.patterns['left_signal']
                rospy.loginfo("Using left signal pattern for green line")
            # For any other color request, use the all pattern if color exists
            elif req.color in self.colors:
                pattern = self._create_all_pattern(req.color)
                rospy.loginfo(f"Using all pattern for {req.color}")
            else:  
                pattern = self.patterns[req.color]
                rospy.loginfo(f"Using standard pattern for {req.color}")
        # For blue line, use right side signal pattern

        # Default fallback
        else:
            rospy.logwarn(f"Requested color {req.color} not found. Using 'off'")
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