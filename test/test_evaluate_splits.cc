#include <gtest/gtest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "param.h"
#include "evaluator.h"
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

thrust::tuple<EvaluateSplitInputs, EvaluateSplitInputs>
GetTreeStumpExample(EvaluateSplitsExample& data) {
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

  return thrust::make_tuple(left, right);
}

void TestEvaluateSingleSplit(bool is_categorical) {
  std::vector<SplitCandidate> out_splits(1);
  GradientPair parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  std::vector<bst_feature_t> feature_set{0, 1};
  std::vector<uint32_t> feature_segments{0, 2, 4};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0};
  std::vector<float> feature_min_values{0.0, 0.0};
  // Setup gradients so that second feature gets higher gain
  std::vector<GradientPair> feature_histogram{
    {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};

  std::vector<FeatureType> feature_types(feature_set.size(),
                                         FeatureType::kCategorical);
  EvaluateSplitInputs input{
    1,
    parent_sum,
    param,
    ToSpan(feature_set),
    (is_categorical ? ToSpan(feature_types) : std::span<FeatureType>{}),
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};
  SplitEvaluator evaluator;
  EvaluateSingleSplit(ToSpan(out_splits), evaluator, input);

  if (g_verbose_flag) {
    for (SplitCandidate e : out_splits) {
      std::cout << e << std::endl;
    }
  }

  SplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 1);
  if (result.dir == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(result.fvalue, 11.0);
  } else {
    EXPECT_FLOAT_EQ(result.fvalue, 12.0);
  }
  EXPECT_FLOAT_EQ(result.left_sum.sum_grad + result.right_sum.sum_grad,
                  parent_sum.grad_);
  EXPECT_FLOAT_EQ(result.left_sum.sum_hess + result.right_sum.sum_hess,
                  parent_sum.hess_);
}

void TestEvaluateSingleSplitWithMissing(bool is_categorical) {
  std::vector<SplitCandidate> out_splits(1);
  GradientPair parent_sum{1.0, 1.5};
  TrainingParam param;

  std::vector<bst_feature_t> feature_set{0};
  std::vector<uint32_t> feature_segments{0, 2};
  std::vector<float> feature_values{1.0, 2.0};
  std::vector<float> feature_min_values{0.0};
  std::vector<GradientPair> feature_histogram{{-0.5, 0.5}, {0.5, 0.5}};
  // The sum of gradient for the data points that lack a value for Feature 0 is (1.0, 0.5)

  std::vector<FeatureType> feature_types(feature_set.size(),
                                         FeatureType::kCategorical);
  EvaluateSplitInputs input{
    1,
    parent_sum,
    param,
    ToSpan(feature_set),
    (is_categorical ? ToSpan(feature_types) : std::span<FeatureType>{}),
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};

  SplitEvaluator evaluator;
  EvaluateSingleSplit(ToSpan(out_splits), evaluator, input);

  if (g_verbose_flag) {
    for (SplitCandidate e : out_splits) {
      std::cout << e << std::endl;
    }
  }

  SplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 0);
  if (is_categorical) {
    // If the feature is categorical, forward and backward passes yield identical split candidate.
    // So result.dir can be kRightDir or kLeftDir.
    EXPECT_FLOAT_EQ(result.fvalue, 1.0);
    // (One-hot) categorical splits specify a single matching category. The data points whose
    // feature values match this single category are associated with the **right** child node;
    // all the other data points are associated with the left child node.
    // Thus, the right_sum variable is set to the gradient sum of all data points whose feature
    // value is identical to the matching category.
    EXPECT_FLOAT_EQ(result.left_sum.sum_grad, 1.5);
    EXPECT_FLOAT_EQ(result.left_sum.sum_hess, 1.0);
    EXPECT_EQ(result.right_sum.sum_grad, -0.5);
    EXPECT_EQ(result.right_sum.sum_hess, 0.5);
    EXPECT_FLOAT_EQ(result.loss_chg, 2.75 - Sqr(parent_sum.grad_) / parent_sum.hess_);
  } else {
    EXPECT_FLOAT_EQ(result.fvalue, 1.0);
    EXPECT_EQ(result.dir, DefaultDirection::kRightDir);
    EXPECT_FLOAT_EQ(result.left_sum.sum_grad, -0.5);
    EXPECT_FLOAT_EQ(result.left_sum.sum_hess, 0.5);
    EXPECT_EQ(result.right_sum.sum_grad, 1.5);
    EXPECT_EQ(result.right_sum.sum_hess, 1.0);
    EXPECT_FLOAT_EQ(result.loss_chg, 2.75 - Sqr(parent_sum.grad_) / parent_sum.hess_);
  }
}

}  // anonymous namespace

TEST(EvaluateSplits, ScanValueOp) {
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs right = thrust::get<1>(split_inputs);

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
  auto for_count_iter = thrust::make_counting_iterator<uint64_t>(0);
  auto for_loc_iter = thrust::make_transform_iterator(for_count_iter, map_to_left_right);
  auto rev_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint64_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto rev_loc_iter = thrust::make_transform_iterator(rev_count_iter, map_to_left_right);
  auto zip_loc_iter = thrust::make_zip_iterator(thrust::make_tuple(for_loc_iter, rev_loc_iter));

  SplitEvaluator evaluator;
  auto scan_input_iter = thrust::make_transform_iterator(
      zip_loc_iter, ScanValueOp{left, right, evaluator});
  for (std::size_t i = 0; i < size; ++i) {
    auto fw = thrust::get<0>(*scan_input_iter);
    auto bw = thrust::get<1>(*scan_input_iter);
    if (g_verbose_flag) {
      std::cout << "forward: " << fw << std::endl << "backward: " << bw << std::endl;
    }
    ++scan_input_iter;
  }
}

TEST(EvaluateSplits, EvaluateSplitsInclusiveScan) {
  SplitEvaluator evaluator;
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs right = thrust::get<1>(split_inputs);
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
  if (out_scan[0].best_direction == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_scan[0].best_fvalue, 1.0f);
  } else {
    EXPECT_FLOAT_EQ(out_scan[0].best_fvalue, 2.0f);
  }
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
  if (out_scan[1].best_direction == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_scan[1].best_fvalue, 11.0f);
  } else {
    EXPECT_FLOAT_EQ(out_scan[1].best_fvalue, 12.0f);
  }

  // Right child
  EXPECT_FLOAT_EQ(out_scan[2].best_loss_chg, 4.0f);
  EXPECT_EQ(out_scan[2].best_findex, 0);
  EXPECT_FLOAT_EQ(out_scan[2].left_sum.sum_grad, -1.0);
  EXPECT_FLOAT_EQ(out_scan[2].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[2].right_sum.sum_grad, 1.0);
  EXPECT_FLOAT_EQ(out_scan[2].right_sum.sum_hess, 0.5);
  if (out_scan[2].best_direction == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_scan[2].best_fvalue, 1.0f);
  } else {
    EXPECT_FLOAT_EQ(out_scan[2].best_fvalue, 2.0f);
  }

  EXPECT_FLOAT_EQ(out_scan[3].best_loss_chg, 1.0f);
  EXPECT_EQ(out_scan[3].best_findex, 1);
  EXPECT_FLOAT_EQ(out_scan[3].left_sum.sum_grad, -0.5);
  EXPECT_FLOAT_EQ(out_scan[3].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_scan[3].right_sum.sum_grad, 0.5);
  EXPECT_FLOAT_EQ(out_scan[3].right_sum.sum_hess, 0.5);
  if (out_scan[3].best_direction == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_scan[3].best_fvalue, 11.0f);
  } else {
    EXPECT_FLOAT_EQ(out_scan[3].best_fvalue, 12.0f);
  }
}

TEST(EvaluateSplits, E2ETreeStump) {
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs right = thrust::get<1>(split_inputs);
  std::vector<SplitCandidate> out_splits(2);
  SplitEvaluator evaluator;
  EvaluateSplits(ToSpan(out_splits), evaluator, left, right);

  if (g_verbose_flag) {
    for (SplitCandidate e : out_splits) {
      std::cout << e << std::endl;
    }
  }

  SplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_FLOAT_EQ(result_left.loss_chg, 4.0f);
  if (result_left.dir == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(result_left.fvalue, 11.0f);
  } else {
    EXPECT_FLOAT_EQ(result_left.fvalue, 12.0f);
  }

  SplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_FLOAT_EQ(result_right.loss_chg, 4.0f);
  if (result_right.dir == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(result_right.fvalue, 1.0f);
  } else {
    EXPECT_FLOAT_EQ(result_right.fvalue, 2.0f);
  }
}

TEST(EvaluateSplits, EvaluateSingleSplit) {
  TestEvaluateSingleSplit(false);
}

TEST(EvaluateSplits, EvaluateSingleCategoricalSplit) {
  TestEvaluateSingleSplit(true);
}

TEST(EvaluateSplits, EvaluateSingleSplitWithMissing) {
  TestEvaluateSingleSplitWithMissing(false);
}

TEST(EvaluateSplits, EvaluateSingleCategoricalSplitWithMissing) {
  TestEvaluateSingleSplitWithMissing(true);
}

TEST(EvaluateSplits, EvaluateSingleSplitEmpty) {
  SplitCandidate nonzeroed;
  nonzeroed.findex = 1;
  nonzeroed.loss_chg = 1.0;

  std::vector<SplitCandidate> out_split(1);
  out_split[0] = nonzeroed;

  SplitEvaluator evaluator;
  EvaluateSingleSplit(ToSpan(out_split), evaluator, EvaluateSplitInputs{});

  SplitCandidate result = out_split[0];
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 produces the best split, but the algorithm must account for feature sampling
TEST(EvaluateSplits, EvaluateSplitsWithFeatureSampling) {
  std::vector<SplitCandidate> out_splits(2);
  GradientPair parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  std::vector<bst_feature_t> feature_set_left{2};
  std::vector<bst_feature_t> feature_set_right{1};
  std::vector<uint32_t> feature_segments{0, 2, 4, 6};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0, 100.0, 200.0};
  std::vector<float> feature_min_values{0.0, 10.0, 98.0};
  std::vector<GradientPair> feature_histogram{
    {-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}
  };
  EvaluateSplitInputs left{
    1,
    parent_sum,
    param,
    ToSpan(feature_set_left),
    {},
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};
  EvaluateSplitInputs right{
    2,
    parent_sum,
    param,
    ToSpan(feature_set_right),
    {},
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};

  SplitEvaluator evaluator;
  EvaluateSplits(ToSpan(out_splits), evaluator, left, right);

  EXPECT_EQ(out_splits[0].findex, 2);
  if (out_splits[0].dir == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_splits[0].fvalue, 100.0f);
  } else {
    EXPECT_FLOAT_EQ(out_splits[0].fvalue, 200.0f);
  }
  EXPECT_FLOAT_EQ(out_splits[0].loss_chg, 4.0f);
  EXPECT_FLOAT_EQ(out_splits[0].left_sum.sum_grad, -1.0);
  EXPECT_FLOAT_EQ(out_splits[0].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_splits[0].right_sum.sum_grad, 1.0);
  EXPECT_FLOAT_EQ(out_splits[0].right_sum.sum_hess, 0.5);

  EXPECT_EQ(out_splits[1].findex, 1);
  if (out_splits[1].dir == DefaultDirection::kRightDir) {
    EXPECT_FLOAT_EQ(out_splits[1].fvalue, 11.0f);
  } else {
    EXPECT_FLOAT_EQ(out_splits[1].fvalue, 12.0f);
  }
  EXPECT_FLOAT_EQ(out_splits[1].loss_chg, 1.0f);
  EXPECT_FLOAT_EQ(out_splits[1].left_sum.sum_grad, -0.5);
  EXPECT_FLOAT_EQ(out_splits[1].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_splits[1].right_sum.sum_grad, 0.5);
  EXPECT_FLOAT_EQ(out_splits[1].right_sum.sum_hess, 0.5);
}
