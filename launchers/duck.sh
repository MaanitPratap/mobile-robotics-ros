#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch follower node
rosrun my_package duck_detection.py

# wait for app to end
dt-launchfile-join