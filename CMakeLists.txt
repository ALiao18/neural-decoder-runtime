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
# Main executable (stub — src/cpp/main.cpp must exist)
# ---------------------------------------------------------------------------
add_executable(neural_decoder_runtime
  src/cpp/main.cpp
)

target_include_directories(neural_decoder_runtime PRIVATE
  ${TORCH_INCLUDE_DIRS}
  src/cpp
)

target_link_libraries(neural_decoder_runtime PRIVATE
  ${TORCH_LIBRARIES}
)

# Required on macOS to find torch dylibs at runtime
set_target_properties(neural_decoder_runtime PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
target_compile_options(neural_decoder_runtime PRIVATE
  -Wall
  -Wextra
  -O2
)

# ---------------------------------------------------------------------------
# Tests (Google Test — wired up when tests/cpp exists)
# ---------------------------------------------------------------------------
# Uncomment when you're ready to add tests:
#
# enable_testing()
# find_package(GTest REQUIRED)
# add_subdirectory(tests/cpp)