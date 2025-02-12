#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from led_service.srv import SetLEDColor, SetLEDColorRequest
import time
import rosnode

class DPatternNode(DTROS):
    def __init__(self, node_name):
        super(DPatternNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        vehicle_name = os.environ['VEHICLE_NAME']
        self.wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)

        # Add state management
        self.STATE_STOP = 1
        self.STATE_TRACING = 2
        self.STATE_RETURN = 3
        self.current_state = self.STATE_STOP
        
        # LED colors for states
        self.STOP_COLOR = "blue"
        self.TRACING_COLOR = "green"
        self.RETURN_COLOR = "green"
        
        # LED service client
        rospy.loginfo("Waiting for LED service...")
        rospy.wait_for_service('set_led_color')
        self.led_service = rospy.ServiceProxy('set_led_color', SetLEDColor)
        rospy.loginfo("LED service connected!")

        # Configuration for D pattern
        self.straight_speed = 0.65  # Base speed for straight lines
        # Calibration factors for straight paths
        self.straight_speed_left = 0.72  # Slightly faster left wheel
        self.straight_speed_right = 0.69   # Base right wheel speed
        
        # Distances and durations
        self.long_straight_distance = 2  # Length of the long straight line
        self.vertical_distance = 1.5      # Length of vertical line
        self.connecting_distance = 0.9    # Length of connecting straight line
        
        # Turn and curve parameters
        self.turn_duration = 0.45        # Duration for 90-degree turn
        self.turn_speed = 0.5            # Base speed for turning
        self.first_d_curve_duration = 2.4  # Time for first D curve
        self.second_d_curve_duration = 2.6 # Time for second D curve
        
        # Different wheel speeds for curved motion
        self.curve_speed_outer = 0.7     # Outer wheel speed during curve
        self.curve_speed_inner = 0.3     # Inner wheel speed during curve
        rospy.on_shutdown(self.on_shutdown)

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

    def move(self, vel_left, vel_right, duration, movement_type):
        """Execute a movement with specified wheel velocities for a given duration."""
        message = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        start_time = time.time()
        
        while time.time() - start_time < duration and not rospy.is_shutdown():
            self._publisher.publish(message)
            time_elapsed = time.time() - start_time
            rospy.loginfo(f"Executing {movement_type}: {(time_elapsed/duration*100):.1f}% complete")
            rospy.sleep(0.1)
        
        self.stop()

    def stop(self):
        """Stop the Duckiebot."""
        stop_message = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_message)
        rospy.sleep(0.1)

    def on_shutdown(self):
        """Cleanup function called on shutdown."""
        if not hasattr(self, '_shutdown_executed'):
            self._shutdown_executed = True  # Flag to prevent multiple executions
            
            self.stop()  # Make sure robot stops
            
            # Send shutdown request to LED service node
            try:
                rospy.loginfo("Sending shutdown request to LED service node...")
                nodes = rosnode.get_node_names()
                led_node = '/led_service_node'  # Corrected node name
                if led_node in nodes:
                    # Set blue color one final time before shutdown
                    self.set_state(self.STATE_STOP)
                    rospy.sleep(0.5)  # Give time for the LED command to process
                    
                    # Now kill the LED node
                    rosnode.kill_nodes([led_node])
                    rospy.loginfo("LED service node shutdown request sent successfully")
                else:
                    rospy.logwarn("LED service node not found")
            except Exception as e:
                rospy.logerr(f"Error shutting down LED service node: {e}")
                
            rospy.loginfo("D pattern node shutdown cleanly")

    def execute_d_pattern(self):
        """Execute the complete D pattern in the correct sequence."""
        # Step 1: Initial longest straight line
        long_straight_duration = self.long_straight_distance / self.straight_speed
        self.move(self.straight_speed_left, self.straight_speed_right, 
                 long_straight_duration, "initial long line")
        rospy.sleep(0.5)

        # Step 2: 90-degree clockwise turn
        self.move(self.turn_speed, -self.turn_speed, 
                 self.turn_duration, "90-degree turn")
        rospy.sleep(0.5)

        # Step 3: Vertical straight line
        vertical_duration = self.vertical_distance / self.straight_speed
        self.move(self.straight_speed_left, self.straight_speed_right,
                 vertical_duration, "vertical line")
        rospy.sleep(0.5)

        # Step 4: First D curve
        self.move(self.curve_speed_outer, self.curve_speed_inner,
                 self.first_d_curve_duration, "first D curve")
        rospy.sleep(0.5)

        # Step 5: Connecting straight line
        connecting_duration = self.connecting_distance / self.straight_speed
        self.move(self.straight_speed_left, self.straight_speed_right,
                 connecting_duration, "connecting line")
        rospy.sleep(0.5)

        # Step 6: Second D curve
        self.move(self.curve_speed_outer, self.curve_speed_inner,
                 self.second_d_curve_duration, "second D curve")
        rospy.sleep(0.5)

        # Step 7: Final straight line back to start
        self.move(self.straight_speed_left, self.straight_speed_right,
                 vertical_duration, "final straight line")
        rospy.sleep(0.5)

        # Step 2: 90-degree clockwise turn
        self.move(self.turn_speed, -self.turn_speed, 
                 self.turn_duration, "90-degree turn")
        rospy.sleep(0.5)

    def run(self):
        """Main run function."""
        rospy.sleep(1.0)  # Initial pause to ensure everything is ready
        
        # State 1: Initial stop
        self.set_state(self.STATE_STOP)
        rospy.loginfo("Initial stop state - waiting 5 seconds")
        rospy.sleep(5.0)
        
        # State 2: Tracing D pattern
        self.set_state(self.STATE_TRACING)
        rospy.loginfo("Starting D pattern trace")
        self.execute_d_pattern()
        
        # State 3: Return state
        self.set_state(self.STATE_RETURN)
        rospy.loginfo("Return state - waiting 5 seconds")
        rospy.sleep(5.0)
        
        # Final stop state
        self.set_state(self.STATE_STOP)
        rospy.loginfo("Pattern completed")

if __name__ == '__main__':
    start_time = time.time()
    node = DPatternNode(node_name='d_pattern_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        if not hasattr(node, '_shutdown_executed'):
            node.on_shutdown()