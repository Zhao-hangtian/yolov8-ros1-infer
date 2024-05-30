cmake_minimum_required(VERSION 3.0.2)
project(my_image_publisher)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(my_image_publisher src/my_image_publisher.cpp)
target_link_libraries(my_image_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
