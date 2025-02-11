cmake_minimum_required(VERSION 3.0.2)
project(led_service)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  duckietown_msgs
)

# Generate services
add_service_files(
  FILES
  SetLEDColor.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  message_runtime
  duckietown_msgs
)