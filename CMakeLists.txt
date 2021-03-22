cmake_minimum_required(VERSION 3.2)
project(project)

#add_executable(eigen-demo modules/main.cc)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Eigen
  )

pybind11_add_module(example modules/main.cc)

# to pythonpath to import module

message("${pybind11_VERSION}")
message("${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

enable_testing()

add_test(NAME pybindings_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/test.py)
#need pythonpath to import module
set_tests_properties (pybindings_test PROPERTIES ENVIRONMENT
    "PYTHONPATH=${project_BINARY_DIR}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Eigen)

