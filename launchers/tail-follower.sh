#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# Set the parking stall parameter (1-4)
# Stall 1 and 2 are on the right side, 3 and 4 are on the left side
export PARKING_STALL=3

# Start the LED service
rosrun led_service led_service_node.py &
sleep 2  # Give the LED service time to start up

# Launch follower node with the parking stall parameter
rosrun my_package tail_follower_node.py --stall $PARKING_STALL &

# wait for app to end
dt-launchfile-join