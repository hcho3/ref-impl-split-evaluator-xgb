#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "param.h"
#include "evaluator.h"
#include "scan.h"
#include "helpers.h"

extern bool g_verbose_flag;

namespace {

struct EvaluateSplitsExample {
  std::vector<bst_feature_t> feature_set;
  std::vector<uint32_t> feature_segments;
  std::vector<float> feature_values;
  std::vector<float> feature_min_values;
  std::vector<GradientPair> feature_histogram_left;
  std::vector<GradientPair> feature_histogram_right;
  GradientPair parent_sum;
  TrainingParam param;
};

std::tuple<EvaluateSplitInputs, EvaluateSplitInputs> GetTreeStumpExample(EvaluateSplitsExample& data) {
  std::vector<bst_feature_t> feature_set{0, 1};
  std::vector<uint32_t> feature_segments{0, 2, 4};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0};
  std::vector<float> feature_min_values{0.0, 0.0};
  std::vector<GradientPair> feature_histogram_left{
      {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}
  };
  std::vector<GradientPair> feature_histogram_right{
      {-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}
  };

  GradientPair parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  data = {feature_set, feature_segments, feature_values, feature_min_values, feature_histogram_left,
          feature_histogram_right, parent_sum, param};

  EvaluateSplitInputs left{
      1,
      data.parent_sum,
      data.param,
      ToSpan(data.feature_set),
      {},
      ToSpan(data.feature_segments),
      ToSpan(data.feature_values),
      ToSpan(data.feature_min_values),
      ToSpan(data.feature_histogram_left)};
  EvaluateSplitInputs right{
      2,
      data.parent_sum,
      data.param,
      ToSpan(data.feature_set),
      {},
      ToSpan(data.feature_segments),
      ToSpan(data.feature_values),
      ToSpan(data.feature_min_values),
      ToSpan(data.feature_histogram_right)};

  return std::make_tuple(left, right);
}

}  // anonymous namespace

TEST(EvaluateSplits, ScanValueOp) {
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs left = std::get<0>(split_inputs);
  EvaluateSplitInputs right = std::get<1>(split_inputs);

  auto map_to_left_right = [&left](uint64_t idx) {
    const auto left_hist_size = static_cast<uint64_t>(left.gradient_histogram.size());
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kLeftChild, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kRightChild, idx - left_hist_size};
    }
  };

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto for_count_iter = MakeForwardCountingIterator<uint64_t>(0);
  auto for_loc_iter = MakeTransformIterator(for_count_iter, map_to_left_right);
  auto rev_count_iter = MakeBackwardCountingIterator<uint64_t>(static_cast<uint64_t>(size) - 1);
  auto rev_loc_iter = MakeTransformIterator(rev_count_iter, map_to_left_right);
  auto zip_loc_iter = MakeZipIterator(for_loc_iter, rev_loc_iter);

  SplitEvaluator evaluator;
  auto scan_input_iter = MakeTransformIterator(zip_loc_iter, ScanValueOp{left, right, evaluator});
  for (std::size_t i = 0; i < size; ++i) {
    auto [fw, bw] = scan_input_iter.Get();
    if (g_verbose_flag) {
      std::cout << "forward: " << fw << std::endl << "backward: " << bw << std::endl;
    }
    scan_input_iter.Next();
  }
}

TEST(EvaluateSplits, EvaluateSplitsInclusiveScan) {
  SplitEvaluator evaluator;
  EvaluateSplitsExample example;
  auto [left, right] = GetTreeStumpExample(example);
  std::vector<ScanComputedElem> out_scan =
      EvaluateSplitsFindOptimalSplitsViaScan(evaluator, left, right);
  if (g_verbose_flag) {
    for (auto e : out_scan) {
      std::cout << e << std::endl;
    }
  }
  EXPECT_EQ(out_scan.size(), 4);
  // Left child
  EXPECT_FLOAT_EQ(out_scan[0].best_loss_chg, 1.0f);
  EXPECT_EQ(out_scan[0].best_findex, 0);
  EXPECT_FLOAT_EQ(out_scan[0].left_sum.sum_grad, -0.5);
  EXPECT_FLOAT_EQ(out_scan[0].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[0].right_sum.sum_grad, 0.5);
  EXPECT_FLOAT_EQ(out_scan[0].right_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[1].best_loss_chg, 4.0f);
  EXPECT_EQ(out_scan[1].best_findex, 1);
  EXPECT_FLOAT_EQ(out_scan[1].left_sum.sum_grad, -1.0);
  EXPECT_FLOAT_EQ(out_scan[1].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[1].right_sum.sum_grad, 1.0);
  EXPECT_FLOAT_EQ(out_scan[1].right_sum.sum_hess, 0.5);
  // Right child
  EXPECT_FLOAT_EQ(out_scan[2].best_loss_chg, 4.0f);
  EXPECT_EQ(out_scan[2].best_findex, 0);
  EXPECT_FLOAT_EQ(out_scan[2].left_sum.sum_grad, -1.0);
  EXPECT_FLOAT_EQ(out_scan[2].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[2].right_sum.sum_grad, 1.0);
  EXPECT_FLOAT_EQ(out_scan[2].right_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[3].best_loss_chg, 1.0f);
  EXPECT_EQ(out_scan[3].best_findex, 1);
  EXPECT_FLOAT_EQ(out_scan[3].left_sum.sum_grad, -0.5);
  EXPECT_FLOAT_EQ(out_scan[3].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[3].right_sum.sum_grad, 0.5);
  EXPECT_FLOAT_EQ(out_scan[3].right_sum.sum_hess, 0.5);
}
