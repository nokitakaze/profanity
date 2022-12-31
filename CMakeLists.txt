cmake_minimum_required(VERSION 3.23)
project(profanity)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -mmmx")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -Wextra -mmmx")

add_executable(profanity Dispatcher.cpp Mode.cpp precomp.cpp profanity.cpp SpeedSample.cpp)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
target_link_libraries("profanity" "-L \"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64/*.lib\"")
add_custom_command(
        TARGET profanity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/keccak.cl
        ${CMAKE_CURRENT_BINARY_DIR}/keccak.cl)
add_custom_command(
        TARGET profanity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/profanity.cl
        ${CMAKE_CURRENT_BINARY_DIR}/profanity.cl)
add_custom_command(
        TARGET profanity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/elliptic.cl
        ${CMAKE_CURRENT_BINARY_DIR}/elliptic.cl)
#add_custom_command(
#        TARGET profanity POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E remove
#        "${CMAKE_CURRENT_BINARY_DIR}/cache-opencl.*")
