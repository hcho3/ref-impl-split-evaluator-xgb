cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(reference_split_evaluator VERSION 0.0.1 LANGUAGES C CXX)

add_executable(reference_split_evaluator)
target_include_directories(reference_split_evaluator PRIVATE include)
target_sources(reference_split_evaluator
  PRIVATE
  src/main.cc
  src/evaluator.cc
  include/param.h
  include/evaluator.h
  include/iterator.h
  include/scan.h
  include/reduce.h
  include/helpers.h)
set_target_properties(reference_split_evaluator PROPERTIES CXX_STANDARD 20)
target_compile_options(reference_split_evaluator PUBLIC -Wall -Wextra -pedantic)
