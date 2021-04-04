cmake_minimum_required(VERSION 3.2)
project(Van_Eyk)

add_executable(
  vaneyk src/tensor.cc
)

include_directories(
  ${cmake_source_dir}/include
)
