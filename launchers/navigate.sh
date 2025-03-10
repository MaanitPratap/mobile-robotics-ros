#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# First start the behavior controller which provides the service
rosrun my_package lane_based_behavior_controller.py &
sleep 3  # Give the service time to start up

# Start the LED service
rosrun led_service led_service_node.py &
sleep 2  # Give the LED service time to start up

# Finally, start the lane detection node which will call the services
rosrun my_package lane_detection_node.py &

# wait for app to end
dt-launchfile-join