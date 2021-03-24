cmake_minimum_required(VERSION 3.2)
project(Project_Runnaway)

#add_executable(
#  main main.cc
#)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/modules
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/xtensor/include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/xtl/include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11/include
)
find_package(pybind11 REQUIRED)
pybind11_add_module(go main.cc)
