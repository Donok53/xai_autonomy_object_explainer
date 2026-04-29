cmake_minimum_required(VERSION 3.0.2)
project(xai_driving_explainer)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rosbag
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/build_bag_event_index.py
  scripts/driving_camera_overlay_viewer.py
  scripts/driving_scene_detector_node.py
  scripts/driving_vlm_inference_node.py
  scripts/driving_vlm_prompt_builder.py
  scripts/tune_camera_lidar_extrinsic.py
  scripts/xai_driving_explainer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
