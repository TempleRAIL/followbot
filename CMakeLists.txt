cmake_minimum_required(VERSION 3.0.2)

project(followbot)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# message and service dependencies
set(MESSAGE_DEPENDENCIES
  std_msgs
  geometry_msgs
  trajectory_msgs
  sensor_msgs
  nav_msgs
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  ${MESSAGE_DEPENDENCIES}
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  DIRECTORY msg
  FILES
  MGSMarker.msg
  MGSMeasurement.msg
  MGSMeasurements.msg
  PathSegment.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  ${MESSAGE_DEPENDENCIES}
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  rospy 
  message_runtime
  ${MESSAGE_DEPENDENCIES}
)


#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  src/calibration/follower_color_filter.py
  src/calibration/follower_line_finder.py
  src/calibration/follower_opencv.py
  src/calibration/hsv_color_finder.py
  src/controller/auto_restart.py
  src/controller/follower_p_controller.py
  src/detector/follower_detector.py
  src/detector/marker_identifier.py
  src/detector/segment_mapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY config images launch meshes robots urdf world
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
