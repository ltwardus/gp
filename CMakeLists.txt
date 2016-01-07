cmake_minimum_required(VERSION 3.2)
project(GeneticPixelizator)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 2.4 REQUIRED)

add_executable(gp src/main.cc)
target_link_libraries(gp ${OpenCV_LIBS})
