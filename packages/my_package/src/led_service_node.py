#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_srvs.srv import SetBool, SetBoolResponse
from duckietown_msgs.msg import LEDPattern

class LEDServiceNode(DTROS):
    def __init__(self, node_name):
        super(LEDServiceNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        vehicle_name = os.environ['VEHICLE_NAME']
        led_topic = f"/{vehicle_name}/led_emitter_node/led_pattern"
        self._publisher = rospy.Publisher(led_topic, LEDPattern, queue_size=1)
        self._service = rospy.Service('set_led_color', SetBool, self.handle_set_led_color)

    def handle_set_led_color(self, req):
        pattern = LEDPattern()
        if req.data:
            pattern.rgb_vals = [0, 0, 255] * 5  # Set all LEDs to blue
        else:
            pattern.rgb_vals = [0, 255, 0] * 5  # Set all LEDs to green
        self._publisher.publish(pattern)
        return SetBoolResponse(success=True, message="LED color changed")

if __name__ == '__main__':
    node = LEDServiceNode(node_name='led_service_node')
    rospy.spin()