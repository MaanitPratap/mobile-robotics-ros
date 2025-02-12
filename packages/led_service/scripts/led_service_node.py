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
        }
        
        self.srv = rospy.Service('set_led_color', SetLEDColor, self.handle_set_led_color)
        
        rospy.loginfo("LED service node initialized")
        

    def handle_set_led_color(self, req):
        if req.color not in self.colors:
            rospy.logwarn(f"Requested color {req.color} not found. Using white.")
            color_data = self.colors['white']
        else:
            color_data = self.colors[req.color]
   
        pattern = LEDPattern()
        
        pattern.color_list = [color_data['name']] * 5
        pattern.rgb_vals = [color_data['rgb']] * 5
        
        pattern.frequency = 0.0
        pattern.frequency_mask = [0] * 5    # lights should not blink
        pattern.color_mask = [1] * 5 
        
        try:
            self.led_publisher.publish(pattern)
            rospy.loginfo(f"Published LED pattern with color: {req.color}")
            return SetLEDColorResponse(success=True)
        except Exception as e:
            rospy.logerr(f"Failed to publish LED pattern: {e}")
            return SetLEDColorResponse(success=False)

    def on_shutdown(self):
        pattern = LEDPattern()
        color_data = self.colors['off']
        pattern.color_list = [color_data['name']] * 5
        pattern.rgb_vals = [color_data['rgb']] * 5
        pattern.frequency = 0.0
        pattern.frequency_mask = [0] * 5
        pattern.color_mask = [1] * 5
        self.led_publisher.publish(pattern)

if __name__ == '__main__':
    node = LEDServiceNode(node_name='led_service_node')
    rospy.spin()