cmake_minimum_required(VERSION 3.20)
project(sgp4_propagator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

add_library(sgp4_core STATIC
    src/sgp4_model.cpp
    src/sgp4_propagator.cpp
    src/wgs84_converter.cpp
    src/batch_propagator.cpp
)

target_include_directories(sgp4_core PUBLIC include)

pybind11_add_module(_sgp4_binding
    src/pybind_module.cpp
)

target_link_libraries(_sgp4_binding PRIVATE sgp4_core)
target_include_directories(_sgp4_binding PRIVATE include)
