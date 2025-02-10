#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# # launch LED service node
# rosrun my_package led_service_node.py &

# launch main node
rosrun my_package wheel_d_node.py

# wait for app to end
dt-launchfile-join