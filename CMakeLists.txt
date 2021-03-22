cmake_minimum_required(VERSION 3.2)
project(project)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/Eigen)

pybind11_add_module(lil_test modules/main.cc)
