cmake_minimum_required(VERSION 3.16)
project(yolo_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Genera compile_commands.json para IntelliSense de VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find OpenCV (must include dnn module) ───────────────────
# Priority: user-specified OpenCV_DIR → pkg-config path → system default
if(NOT OpenCV_DIR)
    # Try pkg-config to find the custom installation
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_OPENCV QUIET opencv4)
        if(PC_OPENCV_FOUND)
            # Derive OpenCV_DIR from prefix returned by pkg-config
            set(OpenCV_DIR "${PC_OPENCV_PREFIX}/lib/cmake/opencv4"
                CACHE PATH "OpenCV cmake config directory" FORCE)
            message(STATUS "Found OpenCV via pkg-config at: ${OpenCV_DIR}")
        endif()
    endif()
endif()

find_package(OpenCV 4.2 REQUIRED
    COMPONENTS core imgproc highgui video videoio dnn)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs   : ${OpenCV_LIBS}")

# ── Optional: CUDA toolkit for compile-time checks ─────────
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA toolkit found: ${CUDA_VERSION}")
    add_compile_definitions(CUDA_TOOLKIT_FOUND)
else()
    message(STATUS "CUDA toolkit not found (runtime check via OpenCV will still work)")
endif()

# ── Source files ─────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/YoloDetector.cpp
    src/OutputParser.cpp
    src/Benchmark.cpp
    src/SystemMonitor.cpp
    src/Utils.cpp
)

add_executable(yolo_benchmark ${SOURCES})

target_include_directories(yolo_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_benchmark PRIVATE
    ${OpenCV_LIBS}
    pthread
    stdc++fs   # std::filesystem on GCC < 9
)

# ── Compiler flags ───────────────────────────────────────────
target_compile_options(yolo_benchmark PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# ── Installation ─────────────────────────────────────────────
install(TARGETS yolo_benchmark RUNTIME DESTINATION bin)

# ── Status summary ───────────────────────────────────────────
message(STATUS "")
message(STATUS "=== Build configuration ===")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID})")
message(STATUS "  C++ std  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV   : ${OpenCV_VERSION}")
if(CUDA_FOUND)
    message(STATUS "  CUDA     : ${CUDA_VERSION}")
endif()
message(STATUS "===========================")
message(STATUS "")
message(STATUS "Build command:  cmake -B build && cmake --build build -j$(nproc)")
message(STATUS "")
