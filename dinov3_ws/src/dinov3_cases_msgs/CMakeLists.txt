cmake_minimum_required(VERSION 3.8)
project(dinov3_cases_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Point2D.msg"
  "msg/Detection3DPoints.msg"
  "msg/Detection3DPointsArray.msg"
  DEPENDENCIES std_msgs geometry_msgs vision_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
