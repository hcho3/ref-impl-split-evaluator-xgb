#ifndef ITERATOR_H_
#define ITERATOR_H_

#include <tuple>
#include <type_traits>

template <typename T, int increment>
class CountingIterator {
 public:
  explicit CountingIterator(T init) : cur_(init) {}
  T Next() {
    T ret = cur_;
    cur_ += increment;
    return ret;
  }

 private:
  T cur_;
};

template <typename InputIterT, typename FuncT, typename TransformedOutputT>
class TransformIterator {
 public:
  TransformIterator(InputIterT begin, FuncT func) : input_iter_(begin), func_(func) {}
  TransformedOutputT Next() {
    return func_(input_iter_.Next());
  }

 private:
  InputIterT input_iter_;
  FuncT func_;
};

template <typename FirstInputIterT, typename SecondInputIterT, typename FirstOutputT,
    typename SecondOutputT>
class ZipIterator {
 public:
  ZipIterator(FirstInputIterT first_begin, SecondInputIterT second_begin)
      : first_input_iter_(first_begin), second_input_iter_(second_begin) {}
  std::tuple<FirstOutputT, SecondOutputT> Next() {
    return std::make_tuple(first_input_iter_.Next(), second_input_iter_.Next());
  }

 private:
  FirstInputIterT first_input_iter_;
  SecondInputIterT second_input_iter_;
};

template <typename IteratorT>
class OutputIterator {
 public:
  using OutputT =
      std::remove_reference_t<std::invoke_result_t<decltype(&IteratorT::operator*), IteratorT>>;
  explicit OutputIterator(IteratorT begin) : cur_(begin) {}
  void Next(OutputT e) {
    *(cur_++) = e;
  }

 private:
  IteratorT cur_;
};

template <typename Tu>
class DiscardIterator {
 public:
  using OutputT = Tu;
  DiscardIterator() = default;
  template <typename DummyT>
  void Next(DummyT) {}
};

template <typename OutputIterT, typename FuncT>
class TransformOutputIterator {
 public:
  using OutputT = typename OutputIterT::OutputT;
  TransformOutputIterator(OutputIterT begin, FuncT func) : output_iter_(begin), func_(func) {}
  void Next(OutputT e) {
    output_iter_.Next(func_(e));
  }

 private:
  OutputIterT output_iter_;
  FuncT func_;
};

template <typename T>
using ForwardCountingIterator = CountingIterator<T, +1>;

template <typename T>
using BackwardCountingIterator = CountingIterator<T, -1>;

template <typename T>
ForwardCountingIterator<T> MakeForwardCountingIterator(T init) {
  return ForwardCountingIterator<T>(init);
}

template <typename T>
BackwardCountingIterator<T> MakeBackwardCountingIterator(T init) {
  return BackwardCountingIterator<T>(init);
}

template <typename InputIterT, typename FuncT>
decltype(auto) MakeTransformIterator(InputIterT begin, FuncT func) {
  using OutputT = std::invoke_result_t<decltype(&InputIterT::Next), InputIterT>;
  using TransformedOutputT = std::invoke_result_t<FuncT, OutputT>;
  return TransformIterator<InputIterT, FuncT, TransformedOutputT>(begin, func);
}

template <typename OutputIterT, typename FuncT>
decltype(auto) MakeTransformOutputIterator(OutputIterT begin, FuncT func) {
  return TransformOutputIterator<OutputIterT, FuncT>(begin, func);
}

template <typename FirstInputIterT, typename SecondInputIterT>
decltype(auto) MakeZipIterator(FirstInputIterT first_begin, SecondInputIterT second_begin) {
  using FirstOutputT = std::invoke_result_t<decltype(&FirstInputIterT::Next), FirstInputIterT>;
  using SecondOutputT = std::invoke_result_t<decltype(&SecondInputIterT::Next), SecondInputIterT>;
  return ZipIterator<FirstInputIterT, SecondInputIterT, FirstOutputT, SecondOutputT>(
      first_begin, second_begin);
}

#endif  // ITERATOR_H_
