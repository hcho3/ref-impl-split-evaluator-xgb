#ifndef SCAN_H_
#define SCAN_H_

#include <type_traits>
#include <cstddef>

template <typename InputIterT, typename OutputIterT, typename ScanOpT>
void InclusiveScan(InputIterT in, OutputIterT out, ScanOpT ScanOp, std::size_t size) {
  if (size == 0) {
    return;
  }
  auto acc = *in;
  *out = acc;
  ++in;
  ++out;
  for (std::size_t i = 1; i < size; ++i) {
    acc = ScanOp(acc, *in);
    *out = acc;
    ++in;
    ++out;
  }
}

#endif  // SCAN_H_
