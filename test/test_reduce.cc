#include <gtest/gtest.h>
#include <vector>
#include "iterator.h"
#include "reduce.h"

TEST(Reduce, ReduceByKey) {
  std::vector<int> keys{1, 3, 3, 3, 2, 2, 1};
  std::vector<int> values{9, 8, 7, 6, 5, 4, 3};
  std::vector<int> out_keys(keys.size());
  std::vector<int> out_values(values.size());
  auto key_iter = InputIterator(keys.begin(), keys.end());
  auto value_iter = InputIterator(values.begin(), values.end());
  auto out_key_iter = OutputIterator(out_keys.begin(), out_keys.end());
  auto out_value_iter = OutputIterator(out_values.begin(), out_values.end());
  auto binary_pred = [](int x, int y) { return x == y; };
  auto binary_op = [](int x, int y) { return x + y; };
  ReduceByKey(key_iter, keys.size(), value_iter, out_key_iter, out_value_iter, binary_pred,
              binary_op);

  std::vector<int> expected_out_keys{1, 3, 2, 1};
  std::vector<int> expected_out_values{9, 21, 9, 3};
  for (std::size_t i = 0; i < expected_out_keys.size(); ++i) {
    EXPECT_EQ(out_keys.at(i), expected_out_keys.at(i));
  }
  for (std::size_t i = 0; i < expected_out_values.size(); ++i) {
    EXPECT_EQ(out_values.at(i), expected_out_values.at(i));
  }
}
