#!/bin/bash

case "$1" in
  "1"|"2"|"3"|"4")
    rosparam set /parking_test_node/select_stall_$1 true
    echo "Selected parking stall $1"
    ;;
  "p"|"park")
    rosparam set /parking_test_node/start_parking true
    echo "Starting parking maneuver"
    ;;
  "s"|"stop")
    rosparam set /parking_test_node/stop true
    echo "Stopping"
    ;;
  "r"|"reset")
    rosparam set /parking_test_node/reset true
    echo "Resetting"
    ;;
  *)
    echo "Usage: $0 [1-4|p|s|r]"
    echo "  1-4: Select parking stall"
    echo "  p: Start parking"
    echo "  s: Stop"
    echo "  r: Reset"
    ;;
esac