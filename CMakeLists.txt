cmake_minimum_required(VERSION 4.2)
project(audio_mic_onnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(audio_mic_onnx
        src/main.cpp
        src/emotion_smoother.cpp
        src/emotion_model.cpp
        src/change_detector.cpp
        src/audio_recorder.cpp
)

target_include_directories(audio_mic_onnx PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
)

find_package(portaudio CONFIG REQUIRED)
find_package(SndFile CONFIG REQUIRED)

find_package(portaudio CONFIG REQUIRED)
find_package(SndFile CONFIG REQUIRED)

target_link_libraries(audio_mic_onnx PRIVATE
        portaudio
        SndFile::sndfile
)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
target_include_directories(audio_mic_onnx PRIVATE
        "${ONNXRUNTIME_ROOT}/include"
)

find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
)

target_link_libraries(audio_mic_onnx PRIVATE
        "${ONNXRUNTIME_LIBRARY}"
)

add_custom_command(TARGET audio_mic_onnx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:audio_mic_onnx>"
)

set_target_properties(audio_mic_onnx PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
