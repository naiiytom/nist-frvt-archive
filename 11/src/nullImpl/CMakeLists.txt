cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

# Build the shared libraries
add_library (frvt_11_aigen_001 SHARED nullimplfrvt11.cpp RetinaFace.cpp ArcFace.cpp FaceAlign.cpp)
target_link_libraries (frvt_11_aigen_001 opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui opencv_video mxnet)

