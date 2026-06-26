cmake_minimum_required(VERSION 3.22.1)
project("qwen3tts")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime headers from AAR dependency
# The ONNX Runtime AAR provides headers in its aar
find_library(android-log log)

add_library(qwen3tts SHARED
    native_bridge.cpp
)

target_link_libraries(qwen3tts
    ${android-log}
)
