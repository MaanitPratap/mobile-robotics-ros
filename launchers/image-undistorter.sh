#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init


rosrun my_package lane_detection_node.py

# wait for app to end
dt-launchfile-join