cmake_minimum_required(VERSION 3.2)
project(example LANGUAGES CXX)

#add_executable(eigen-demo modules/main.cc)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Eigen
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Eigen)

# python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(lil_test modules/main.cc)

add_test(NAME pybindings_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/test.py)

#need pythonpath to import module
set_tests_properties (pybindings_test PROPERTIES ENVIRONMENT
    "PYTHONPATH=${project_BINARY_DIR}")
