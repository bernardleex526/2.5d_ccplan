cmake_minimum_required(VERSION 3.16)
project(hill_patrol_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(kinematic_follower SHARED src/kinematic_follower.cc)
target_compile_features(kinematic_follower PRIVATE cxx_std_17)
target_link_libraries(kinematic_follower PRIVATE ignition-gazebo6::ignition-gazebo6 yaml-cpp)
target_compile_options(kinematic_follower PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS kinematic_follower LIBRARY DESTINATION lib)

# World products are generated from the sourced parameter files at build time.
set(WORLD_DIR ${CMAKE_CURRENT_BINARY_DIR}/worlds)
set(WORLD_OUTPUTS)
foreach(world hill_world hill_world_small)
  set(stamp ${WORLD_DIR}/${world}.sdf)
  add_custom_command(
    OUTPUT ${stamp}
    COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/hill_patrol_sim/generate_world.py
            ${CMAKE_CURRENT_SOURCE_DIR}/config/${world}.yaml --output-dir ${WORLD_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hill_patrol_sim/generate_world.py
            ${CMAKE_CURRENT_SOURCE_DIR}/config/${world}.yaml
    COMMENT "Generating ${world}")
  list(APPEND WORLD_OUTPUTS ${stamp})
endforeach()
add_custom_target(worlds ALL DEPENDS ${WORLD_OUTPUTS})

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/cloud_adapter scripts/cloud_merger scripts/gait_rule_monitor
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config models DESTINATION share/${PROJECT_NAME})
install(DIRECTORY ${WORLD_DIR} DESTINATION share/${PROJECT_NAME})
ament_package()
