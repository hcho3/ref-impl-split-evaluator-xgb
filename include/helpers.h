#ifndef HELPERS_H_
#define HELPERS_H_

#include <algorithm>
#include <span>

template <typename VectorT, typename T = typename VectorT::value_type>
inline std::span<T> ToSpan(VectorT& vec) {
  return {std::begin(vec), std::end(vec)};
}

template <typename IteratorT>
inline std::size_t SegmentId(IteratorT first, IteratorT last, std::size_t idx) {
  return std::upper_bound(first, last, idx) - 1 - first;
}

template <typename T>
inline std::size_t SegmentId(std::span<T> segments_ptr, std::size_t idx) {
  return SegmentId(segments_ptr.begin(), segments_ptr.end(), idx);
}

#endif  // HELPERS_H_
