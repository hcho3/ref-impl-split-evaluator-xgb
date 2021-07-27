#ifndef SCAN_H_
#define SCAN_H_

#include <type_traits>
#include <cstddef>

template <typename InputIterT, typename OutputIterT, typename ScanOpT>
void InclusiveScan(InputIterT in, OutputIterT out, ScanOpT ScanOp, std::size_t size) {
  using ElemT = std::invoke_result_t<decltype(&InputIterT::Next), InputIterT>;
  ElemT acc = in.Next();
  for (std::size_t i = 0; i < size; ++i) {
    out.Next(acc);
    acc = ScanOp(acc, in.Next());
  }
}

#endif  // SCAN_H_
