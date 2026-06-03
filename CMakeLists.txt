cmake_minimum_required(VERSION 3.18)
project(neural_decoder_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Torch — resolved from the active Python environment
# ---------------------------------------------------------------------------
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

message(STATUS "Torch version:  ${Torch_VERSION}")
message(STATUS "Torch libs:     ${TORCH_LIBRARIES}")

# ---------------------------------------------------------------------------
# Runtime static library
# All non-main C++ logic lives here. Both the main executable and Google Test
# link against this target — nothing testable goes in main.cpp.
# ---------------------------------------------------------------------------
add_library(ndr_runtime STATIC
  src/cpp/runtime/model_runner.cpp
)

target_include_directories(ndr_runtime PUBLIC
  ${CMAKE_SOURCE_DIR}/src/cpp
  ${CMAKE_SOURCE_DIR}/src/cpp/third_party
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(ndr_runtime PUBLIC
  ${TORCH_LIBRARIES}
)

target_compile_options(ndr_runtime PRIVATE
  -Wall
  -Wextra
  -O2
)

set_target_properties(ndr_runtime PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(neural_decoder_runtime
  src/cpp/main.cpp
)

target_link_libraries(neural_decoder_runtime PRIVATE
  ndr_runtime
)

set_target_properties(neural_decoder_runtime PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# ---------------------------------------------------------------------------
# Tests (Google Test — uncomment when tests/cpp is ready)
# ---------------------------------------------------------------------------
# enable_testing()
# find_package(GTest REQUIRED)
# add_subdirectory(tests/cpp)