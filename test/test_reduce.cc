#include "reduce.h"
#include <gtest/gtest.h>
#include <vector>

TEST(Reduce, ReduceByKey) {
  std::vector<int> keys{1, 3, 3, 3, 2, 2, 1};
  std::vector<int> values{9, 8, 7, 6, 5, 4, 3};
  std::vector<int> out_keys(keys.size());
  std::vector<int> out_values(values.size());
  auto binary_pred = [](int x, int y) { return x == y; };
  auto binary_op = [](int x, int y) { return x + y; };
  ReduceByKey(keys.begin(), keys.end(), values.begin(), out_keys.begin(), out_values.begin(),
              binary_pred, binary_op);

  std::vector<int> expected_out_keys{1, 3, 2, 1};
  std::vector<int> expected_out_values{9, 21, 9, 3};
  for (std::size_t i = 0; i < expected_out_keys.size(); ++i) {
    EXPECT_EQ(out_keys.at(i), expected_out_keys.at(i));
  }
  for (std::size_t i = 0; i < expected_out_values.size(); ++i) {
    EXPECT_EQ(out_values.at(i), expected_out_values.at(i));
  }
}
