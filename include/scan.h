#ifndef SCAN_H_
#define SCAN_H_

#include <numeric>
#include <cstddef>

template <typename InputIterT, typename OutputIterT, typename ScanOpT>
void InclusiveScan(InputIterT in, OutputIterT out, ScanOpT scan_op, std::size_t size) {
  std::inclusive_scan(in, in + static_cast<std::ptrdiff_t>(size), out, scan_op);
}

#endif  // SCAN_H_
