#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch parking node
rosrun my_package parking.py

# wait for app to end
dt-launchfile-join