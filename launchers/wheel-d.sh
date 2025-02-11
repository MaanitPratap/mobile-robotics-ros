#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch LED service node first (with & for background execution)
rosrun led_service led_service_node.py &
sleep 2  # Give the service time to start up

# launch main node
rosrun my_package wheel_d_node.py &

# wait for app to end
dt-launchfile-join