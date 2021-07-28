
About
=====
This is a "reference" implementation of the proposed refactor of GPU split evaluation kernel in XGBoost: https://github.com/trivialfis/xgboost/tree/rework-evaluation. The reference implementation is simple to understand and debug since it uses a single CPU thread.

How to build and run
====================
Use CMake to build. Make sure to have a recent C++ compiler that supports C++20.

```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
make
./reference_split_evaluator --verbose
```
