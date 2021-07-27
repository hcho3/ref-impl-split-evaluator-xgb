#ifndef REDUCE_H_
#define REDUCE_H_

#include <cstddef>

template <typename KeyIterator, typename ValueIterator, typename KeyOutputIterator,
    typename ValueOutputIterator, typename BinaryPredicate, typename BinaryOp>
void ReduceByKey(KeyIterator keys_first, std::size_t size, ValueIterator values_first,
                 KeyOutputIterator keys_output, ValueOutputIterator values_output,
                 BinaryPredicate binary_pred, BinaryOp binary_op) {
  if (size == 0) {
    return;
  }
  using KeyT = std::invoke_result_t<decltype(&KeyIterator::Next), KeyIterator>;
  using ValueT = std::invoke_result_t<decltype(&ValueIterator::Next), ValueIterator>;
  KeyT prev_key = keys_first.Next();
  ValueT acc = values_first.Next();
  std::size_t last_reset_idx = 0;
  for (std::size_t i = 1; i < size; ++i) {
    KeyT current_key = keys_first.Next();
    ValueT current_value = values_first.Next();
    if (binary_pred(prev_key, current_key)) {
      acc = binary_op(acc, current_value);
    } else {
      keys_output.Next(prev_key);
      values_output.Next(acc);
      prev_key = current_key;
      acc = current_value;
      last_reset_idx = i;
    }
  }
  if (last_reset_idx != size) {
    keys_output.Next(prev_key);
    values_output.Next(acc);
  }
}

#endif  // REDUCE_H_
