cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(reference_split_evaluator VERSION 0.0.1 LANGUAGES C CXX)

add_executable(reference_split_evaluator)
target_sources(reference_split_evaluator PRIVATE main.cc param.h evaluator.h iterator.h scan.h
        reduce.h helpers.h evaluator.cc)
set_target_properties(reference_split_evaluator PROPERTIES CXX_STANDARD 20)
target_compile_options(reference_split_evaluator PUBLIC -Wall -Wextra -pedantic)
