cmake_minimum_required(VERSION 3.0.2)
project(migrave_person_state_estimation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roslint
  migrave_ros_msgs
  migrave_common
)

roslint_python()
catkin_python_setup()
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

