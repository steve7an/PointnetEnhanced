cmake_minimum_required(VERSION 2.8)
 
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
 
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
 
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(inner_product SHARED tf_grouping.cpp)
 
#include_directories(${Tensorflow_INCLUDE_DIRS})
#add_library(inner_product_grad SHARED inner_product_grad.cc)