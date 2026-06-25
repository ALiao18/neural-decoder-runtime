add_executable(ndr_tests
  test_zscore.cpp
)
target_link_libraries(ndr_tests PRIVATE
  ndr_runtime
  GTest::gtest_main
)
target_include_directories(ndr_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/src/cpp
)

target_compile_definitions(ndr_tests PRIVATE
    NDR_TEST_ARTIFACTS_DIR="${CMAKE_SOURCE_DIR}/artifacts"
    NDR_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/cpp/fixtures"
)

include(GoogleTest)
gtest_discover_tests(ndr_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)