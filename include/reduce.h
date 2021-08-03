#ifndef REDUCE_H_
#define REDUCE_H_

#include <cstddef>

template <typename KeyIterator, typename ValueIterator, typename KeyOutputIterator,
    typename ValueOutputIterator, typename BinaryPredicate, typename BinaryOp>
void ReduceByKey(KeyIterator keys_first, KeyIterator keys_last, ValueIterator values_first,
                 KeyOutputIterator keys_output, ValueOutputIterator values_output,
                 BinaryPredicate binary_pred, BinaryOp binary_op) {
  if (keys_first == keys_last) {
    return;
  }
  auto prev_key = *keys_first;
  auto acc = *values_first;
  KeyIterator last_reset_pt = keys_first;
  ++keys_first;
  ++values_first;
  for (; keys_first != keys_last; ++keys_first, ++values_first) {
    auto current_key = *keys_first;
    auto current_value = *values_first;
    if (binary_pred(prev_key, current_key)) {
      acc = binary_op(acc, current_value);
    } else {
      *keys_output = prev_key;
      ++keys_output;
      *values_output = acc;
      ++values_output;
      prev_key = current_key;
      acc = current_value;
      last_reset_pt = keys_first;
    }
  }
  if (last_reset_pt != keys_last) {
    *keys_output = prev_key;
    *values_output = acc;
  }
}

#endif  // REDUCE_H_
