cmake_minimum_required(VERSION 3.7)
project(neural_network)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.c
        neural_net.c
        neural_net.h)

add_executable(neural_network ${SOURCE_FILES})
target_link_libraries(neural_network m)