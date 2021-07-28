#ifndef ITERATOR_H_
#define ITERATOR_H_

#include <stdexcept>
#include <tuple>
#include <type_traits>

template <typename IteratorT>
class InputIterator {   // light wrapper around std InputIterator
 public:
  using OutputT =
      std::remove_reference_t<std::invoke_result_t<decltype(&IteratorT::operator*), IteratorT>>;
  explicit InputIterator(IteratorT begin, IteratorT end) : cur_(begin), end_(end) {}
  OutputT Get() {
    if (cur_ == end_) {
      throw std::out_of_range("Attempting to access beyond the last element");
    }
    return *cur_;
  }
  void Next() {
    ++cur_;
  }

 private:
  IteratorT cur_;
  IteratorT end_;
};

template <typename T, int increment>
class CountingIterator {
 public:
  explicit CountingIterator(T init) : cur_(init) {}
  T Get() {
    return cur_;
  }
  void Next() {
    cur_ += increment;
  }

 private:
  T cur_;
};

template <typename InputIterT, typename FuncT, typename TransformedOutputT>
class TransformIterator {
 public:
  TransformIterator(InputIterT begin, FuncT func) : input_iter_(begin), func_(func) {}
  TransformedOutputT Get() {
    return func_(input_iter_.Get());
  }
  void Next() {
    input_iter_.Next();
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
  std::tuple<FirstOutputT, SecondOutputT> Get() {
    return std::make_tuple(first_input_iter_.Get(), second_input_iter_.Get());
  }
  void Next() {
    first_input_iter_.Next();
    second_input_iter_.Next();
  }

 private:
  FirstInputIterT first_input_iter_;
  SecondInputIterT second_input_iter_;
};

template <typename IteratorT>
class OutputIterator {   // light wrapper around std OutputIterator
 public:
  using OutputT =
      std::remove_reference_t<std::invoke_result_t<decltype(&IteratorT::operator*), IteratorT>>;
  OutputIterator(IteratorT begin, IteratorT end) : cur_(begin), end_(end) {}
  void Set(OutputT e) {
    if (cur_ == end_) {
      throw std::out_of_range("Attempting to access beyond the last element");
    }
    *cur_ = e;
  }
  void Next() {
    ++cur_;
  }

 private:
  IteratorT cur_;
  IteratorT end_;
};

template <typename Tu>
class DiscardIterator {
 public:
  using OutputT = Tu;
  DiscardIterator() = default;
  template <typename DummyT>
  void Set(DummyT) {}
  void Next() {}
};

template <typename OutputIterT, typename FuncT>
class TransformOutputIterator {
 public:
  using OutputT = typename OutputIterT::OutputT;
  TransformOutputIterator(OutputIterT begin, FuncT func) : output_iter_(begin), func_(func) {}
  void Set(OutputT e) {
    output_iter_.Set(func_(e));
  }
  void Next() {
    output_iter_.Next();
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
  using OutputT = std::invoke_result_t<decltype(&InputIterT::Get), InputIterT>;
  using TransformedOutputT = std::invoke_result_t<FuncT, OutputT>;
  return TransformIterator<InputIterT, FuncT, TransformedOutputT>(begin, func);
}

template <typename OutputIterT, typename FuncT>
decltype(auto) MakeTransformOutputIterator(OutputIterT begin, FuncT func) {
  return TransformOutputIterator<OutputIterT, FuncT>(begin, func);
}

template <typename FirstInputIterT, typename SecondInputIterT>
decltype(auto) MakeZipIterator(FirstInputIterT first_begin, SecondInputIterT second_begin) {
  using FirstOutputT = std::invoke_result_t<decltype(&FirstInputIterT::Get), FirstInputIterT>;
  using SecondOutputT = std::invoke_result_t<decltype(&SecondInputIterT::Get), SecondInputIterT>;
  return ZipIterator<FirstInputIterT, SecondInputIterT, FirstOutputT, SecondOutputT>(
      first_begin, second_begin);
}

#endif  // ITERATOR_H_
