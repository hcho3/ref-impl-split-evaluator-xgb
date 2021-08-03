#ifndef SCAN_H_
#define SCAN_H_

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/tuple.h>
#include <type_traits>
#include <cstddef>

template <typename InputIterT, typename OutputIterT, typename ScanOpT>
inline void InclusiveScan(InputIterT in, OutputIterT out, ScanOpT scan_op, std::size_t size) {
  // Homebrew Scan implementation, since Thurst does not implement thrust::inclusive_scan
  // for host CPP target.
  if (size == 0) {
    return;
  }
  auto acc = *in;
  *out = acc;
  ++in;
  ++out;
  for (std::size_t i = 1; i < size; ++i) {
    acc = scan_op(acc, *in);
    *out = acc;
    ++in;
    ++out;
  }
}

template <typename KeyT, typename ValueT, typename ScanOpT>
struct ScanByKeyOp {
  ScanOpT scan_op;

  thrust::tuple<KeyT, ValueT> operator()(
      const thrust::tuple<KeyT, ValueT>& lhs,
      const thrust::tuple<KeyT, ValueT>& rhs) {
    if (thrust::get<0>(lhs) != thrust::get<0>(rhs)) {
      return rhs;
    }
    return thrust::make_tuple(thrust::get<0>(lhs),
        scan_op(thrust::get<1>(lhs), thrust::get<1>(rhs)));
  }
};

// aka "segmented scan"
template <typename KeyIterT, typename InputIterT, typename OutputIterT, typename ScanOpT>
inline void InclusiveScanByKey(KeyIterT key_in, InputIterT value_in, OutputIterT value_out,
                               ScanOpT scan_op, std::size_t size) {
  using KeyT = std::remove_reference_t<decltype(*key_in)>;
  using ValueT = std::remove_reference_t<decltype(*value_in)>;
  auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(key_in, value_in));
  auto out_transform_iter = thrust::make_transform_output_iterator(
      value_out, [](const auto& tup) { return thrust::get<1>(tup); });
  InclusiveScan(zip_iter, out_transform_iter, ScanByKeyOp<KeyT, ValueT, ScanOpT>{scan_op}, size);
}

#endif  // SCAN_H_
