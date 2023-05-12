cmake_minimum_required(VERSION 3.0.0)
project(normalizercxx VERSION 0.1.0)

# if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # workaround for clangd+mingw
# 	set(AXX_TARGET_TRIPLE "i686-w64-mingw32")
# 	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=${AXX_TARGET_TRIPLE}")
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=${AXX_TARGET_TRIPLE}")
# endif()

if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell -Ofast -ffast-math")
endif()

find_package(SndFile REQUIRED)
find_package(SampleRate REQUIRED)


add_executable(normalizercxx main.cpp)
set_property(TARGET normalizercxx PROPERTY CXX_STANDARD 20)
set_property(TARGET normalizercxx PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(normalizercxx PRIVATE SndFile::sndfile SampleRate::samplerate)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
