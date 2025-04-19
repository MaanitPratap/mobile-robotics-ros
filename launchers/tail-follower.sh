#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init


# Start the LED service
rosrun led_service led_service_node.py &
sleep 2  # Give the LED service time to start up

# launch follower node
rosrun my_package tail_follower_node.py &

# wait for app to end
dt-launchfile-join