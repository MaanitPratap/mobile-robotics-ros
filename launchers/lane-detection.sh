#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# rosrun my_package color_detector.py &
# sleep 3  # Ensure service starts properly
# # launch subscriber
# rosrun my_package image_undistorter.py &

# # Print usage instructions
# echo "=== Color Detection Services ==="
# echo "To update HSV values: rosservice call /color_detector/update_hsv_values"
# echo "============================"

rosrun my_package lane_detection_node.py

# wait for app to end
dt-launchfile-join