//
// Created by phcho on 7/27/21.
//

#ifndef HELPERS_H_
#define HELPERS_H_

#include <span>

template <typename VectorT, typename T = typename VectorT::value_type>
std::span<T> ToSpan(VectorT& vec) {
  return {std::begin(vec), std::end(vec)};
}

#endif  // HELPERS_H_
