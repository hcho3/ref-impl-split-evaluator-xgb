#ifndef SCAN_H_
#define SCAN_H_

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

#endif  // SCAN_H_
