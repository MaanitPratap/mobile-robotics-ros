#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from led_service.srv import SetLEDColor, SetLEDColorRequest
import time

class DPatternWrapper(DTROS):
    def __init__(self, node_name):
        super(DPatternWrapper, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # State definitions
        self.STATE_STOP = 1
        self.STATE_TRACING = 2
        self.STATE_RETURN = 3
        self.current_state = self.STATE_STOP
        
        # LED colors for states
        self.STOP_COLOR = "blue"
        self.TRACING_COLOR = "green"
        self.RETURN_COLOR = "red"
        
        # LED service client
        rospy.loginfo("Waiting for LED service...")
        rospy.wait_for_service('set_led_color')
        self.led_service = rospy.ServiceProxy('set_led_color', SetLEDColor)
        rospy.loginfo("LED service connected!")

    def set_state(self, state):
        """Set the current state and update LED color."""
        self.current_state = state
        color = {
            self.STATE_STOP: self.STOP_COLOR,
            self.STATE_TRACING: self.TRACING_COLOR,
            self.STATE_RETURN: self.RETURN_COLOR
        }[state]
        
        try:
            # Create and send LED service request
            req = SetLEDColorRequest()
            req.color = color
            response = self.led_service(req)
            if not response.success:
                rospy.logwarn(f"Failed to set LED color to {color}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"LED service call failed: {e}")

    def run(self):
        """Main run function that handles states."""
        # Create instance of original DPatternNode but don't run it yet
        from wheel_d_node import DPatternMainNode
        d_pattern = DPatternMainNode(node_name='d_pattern_node')
        
        # State 1: Initial stop with blue LED
        self.set_state(self.STATE_STOP)
        rospy.loginfo("Initial stop state - waiting 5 seconds")
        rospy.sleep(5.0)
        
        # State 2: Tracing D pattern with green LED
        self.set_state(self.STATE_TRACING)
        rospy.loginfo("Starting D pattern trace")
        d_pattern.execute_d_pattern()  # Execute the original D pattern
        
        # State 3: Final stop with red LED
        self.set_state(self.STATE_RETURN)
        rospy.loginfo("Return state - waiting 5 seconds")
        rospy.sleep(5.0)
        
        # Back to stop state
        self.set_state(self.STATE_STOP)
        rospy.loginfo("Final stop state")

if __name__ == '__main__':
    node = DPatternWrapper(node_name='d_pattern_wrapper')
    node.run()