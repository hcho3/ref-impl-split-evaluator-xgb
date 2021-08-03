cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(reference_split_evaluator VERSION 0.0.1 LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG        1.13.0
)
FetchContent_MakeAvailable(thrust)
thrust_create_target(Thrust HOST CPP DEVICE CPP)

add_executable(reference_split_evaluator)
target_include_directories(reference_split_evaluator PRIVATE include)
target_link_libraries(reference_split_evaluator PRIVATE gtest Thrust)
target_sources(reference_split_evaluator
  PRIVATE
  src/evaluator.cc
  include/param.h
  include/evaluator.h
  include/scan.h
  include/reduce.h
  include/helpers.h
  test/test_main.cc
  test/test_evaluate_splits.cc
  test/test_iterator.cc
  test/test_scan.cc
  test/test_reduce.cc)
set_target_properties(reference_split_evaluator PROPERTIES CXX_STANDARD 20)
target_compile_options(reference_split_evaluator PUBLIC -Wall -Wextra -pedantic)
