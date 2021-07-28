#ifndef SCAN_H_
#define SCAN_H_

#include <type_traits>
#include <cstddef>

template <typename InputIterT, typename OutputIterT, typename ScanOpT>
void InclusiveScan(InputIterT in, OutputIterT out, ScanOpT ScanOp, std::size_t size) {
  using ElemT = std::invoke_result_t<decltype(&InputIterT::Get), InputIterT>;
  if (size == 0) {
    return;
  }
  ElemT acc = in.Get();
  out.Set(acc);
  in.Next();
  out.Next();
  for (std::size_t i = 1; i < size; ++i) {
    acc = ScanOp(acc, in.Get());
    out.Set(acc);
    in.Next();
    out.Next();
  }
}

#endif  // SCAN_H_
