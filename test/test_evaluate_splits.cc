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
  std::vector<GradStats> feature_histogram_left;
  std::vector<GradStats> feature_histogram_right;
  GradStats parent_sum;
  TrainingParam param;
};

thrust::tuple<EvaluateSplitInputs<GradStats>, EvaluateSplitInputs<GradStats>>
GetTreeStumpExample(EvaluateSplitsExample& data) {
  std::vector<bst_feature_t> feature_set{0, 1};
  std::vector<uint32_t> feature_segments{0, 2, 4};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0};
  std::vector<float> feature_min_values{0.0, 0.0};
  std::vector<GradStats> feature_histogram_left{
      {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}
  };
  std::vector<GradStats> feature_histogram_right{
      {-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}
  };

  GradStats parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  data = {feature_set, feature_segments, feature_values, feature_min_values, feature_histogram_left,
          feature_histogram_right, parent_sum, param};

  EvaluateSplitInputs<GradStats> left{
      1,
      data.parent_sum,
      data.param,
      ToSpan(data.feature_set),
      {},
      ToSpan(data.feature_segments),
      ToSpan(data.feature_values),
      ToSpan(data.feature_min_values),
      ToSpan(data.feature_histogram_left)};
  EvaluateSplitInputs<GradStats> right{
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
  GradStats parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  std::vector<bst_feature_t> feature_set{0, 1};
  std::vector<uint32_t> feature_segments{0, 2, 4};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0};
  std::vector<float> feature_min_values{0.0, 0.0};
  // Setup gradients so that second feature gets higher gain
  std::vector<GradStats> feature_histogram{
    {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};

  std::vector<FeatureType> feature_types(feature_set.size(),
                                         FeatureType::kCategorical);
  EvaluateSplitInputs<GradStats> input{
    1,
    parent_sum,
    param,
    ToSpan(feature_set),
    (is_categorical ? ToSpan(feature_types) : std::span<FeatureType>{}),
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};
  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSingleSplit(ToSpan(out_splits), evaluator, input);

  if (g_verbose_flag) {
    for (SplitCandidate e : out_splits) {
      std::cout << e << std::endl;
    }
  }

  SplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 1);
  if (result.dir == DefaultDirection::kRightDir) {
    EXPECT_EQ(result.fvalue, 11.0);
  } else {
    EXPECT_EQ(result.fvalue, 12.0);
  }
}

void TestEvaluateSingleSplitWithMissing(bool is_categorical) {
  std::vector<SplitCandidate> out_splits(1);
  GradStats parent_sum{1.0, 1.5};
  TrainingParam param;

  std::vector<bst_feature_t> feature_set{0};
  std::vector<uint32_t> feature_segments{0, 2};
  std::vector<float> feature_values{1.0, 2.0};
  std::vector<float> feature_min_values{0.0};
  std::vector<GradStats> feature_histogram{{-0.5, 0.5}, {0.5, 0.5}};
  // The sum of gradient for the data points that lack a value for Feature 0 is (1.0, 0.5)

  std::vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  EvaluateSplitInputs<GradStats> input{
    1,
    parent_sum,
    param,
    ToSpan(feature_set),
    (is_categorical ? ToSpan(feature_types) : std::span<FeatureType>{}),
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};

  SplitEvaluator<TrainingParam> evaluator;
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
    EXPECT_EQ(result.fvalue, 1.0f);
    // (One-hot) categorical splits specify a single matching category. The data points whose
    // feature values match this single category are associated with the **right** child node;
    // all the other data points are associated with the left child node.
    // Thus, the right_sum variable is set to the gradient sum of all data points whose feature
    // value is identical to the matching category.
    EXPECT_FLOAT_EQ(result.left_sum.sum_grad, 1.5);
    EXPECT_FLOAT_EQ(result.left_sum.sum_hess, 1.0);
    EXPECT_EQ(result.right_sum.sum_grad, -0.5);
    EXPECT_EQ(result.right_sum.sum_hess, 0.5);
    EXPECT_FLOAT_EQ(result.loss_chg, 2.75 - Sqr(parent_sum.sum_grad) / parent_sum.sum_hess);
  } else {
    EXPECT_EQ(result.fvalue, 1.0f);
    EXPECT_EQ(result.dir, DefaultDirection::kRightDir);
    EXPECT_FLOAT_EQ(result.left_sum.sum_grad, -0.5);
    EXPECT_FLOAT_EQ(result.left_sum.sum_hess, 0.5);
    EXPECT_EQ(result.right_sum.sum_grad, 1.5);
    EXPECT_EQ(result.right_sum.sum_hess, 1.0);
    EXPECT_FLOAT_EQ(result.loss_chg, 2.75 - Sqr(parent_sum.sum_grad) / parent_sum.sum_hess);
  }
}

}  // anonymous namespace

TEST(EvaluateSplits, ScanValueOp) {
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs<GradStats> left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs<GradStats> right = thrust::get<1>(split_inputs);

  auto map_to_hist_bin = [&left](uint32_t idx) {
    const auto left_hist_size = static_cast<uint32_t>(left.gradient_histogram.size());
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{0, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{1, idx - left_hist_size};
    }
  };

  SplitEvaluator<TrainingParam> evaluator;
  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto forward_count_iter = thrust::make_counting_iterator<uint32_t>(0);
  auto forward_bin_iter = thrust::make_transform_iterator(forward_count_iter, map_to_hist_bin);
  auto forward_scan_input_iter = thrust::make_transform_iterator(
      forward_bin_iter, ScanValueOp<GradStats>{true, left, right, evaluator});

  for (std::size_t i = 0; i < size; ++i) {
    if (g_verbose_flag) {
      std::cout << "forward: " << (*forward_scan_input_iter) << std::endl;
    }
    ++forward_scan_input_iter;
  }

  auto backward_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint32_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto backward_bin_iter = thrust::make_transform_iterator(backward_count_iter, map_to_hist_bin);
  auto backward_scan_input_iter = thrust::make_transform_iterator(
      backward_bin_iter, ScanValueOp<GradStats>{false, left, right, evaluator});
  for (std::size_t i = 0; i < size; ++i) {
    if (g_verbose_flag) {
      std::cout << "backward: " << (*backward_scan_input_iter) << std::endl;
    }
    ++backward_scan_input_iter;
  }
}

TEST(EvaluateSplits, EvaluateSplitsInclusiveScan) {
  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs<GradStats> left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs<GradStats> right = thrust::get<1>(split_inputs);
  std::vector<ScanComputedElem<GradStats>> out_scan =
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
    EXPECT_EQ(out_scan[0].best_fvalue, 1.0f);
    EXPECT_FLOAT_EQ(out_scan[0].best_partial_sum.sum_grad, -0.5);
    EXPECT_FLOAT_EQ(out_scan[0].best_partial_sum.sum_hess, 0.5);
  } else {
    EXPECT_EQ(out_scan[0].best_fvalue, 2.0f);
    EXPECT_FLOAT_EQ(out_scan[0].best_partial_sum.sum_grad, 0.5);
    EXPECT_FLOAT_EQ(out_scan[0].best_partial_sum.sum_hess, 0.5);
  }

  EXPECT_FLOAT_EQ(out_scan[1].best_loss_chg, 4.0f);
  EXPECT_EQ(out_scan[1].best_findex, 1);
  if (out_scan[1].best_direction == DefaultDirection::kRightDir) {
    EXPECT_EQ(out_scan[1].best_fvalue, 11.0f);
    EXPECT_FLOAT_EQ(out_scan[1].best_partial_sum.sum_grad, -1.0);
    EXPECT_FLOAT_EQ(out_scan[1].best_partial_sum.sum_hess, 0.5);
  } else {
    EXPECT_EQ(out_scan[1].best_fvalue, 12.0f);
    EXPECT_FLOAT_EQ(out_scan[1].best_partial_sum.sum_grad, 1.0);
    EXPECT_FLOAT_EQ(out_scan[1].best_partial_sum.sum_hess, 0.5);
  }

  // Right child
  EXPECT_FLOAT_EQ(out_scan[2].best_loss_chg, 4.0f);
  EXPECT_EQ(out_scan[2].best_findex, 0);
  if (out_scan[2].best_direction == DefaultDirection::kRightDir) {
    EXPECT_EQ(out_scan[2].best_fvalue, 1.0f);
    EXPECT_FLOAT_EQ(out_scan[2].best_partial_sum.sum_grad, -1.0);
    EXPECT_FLOAT_EQ(out_scan[2].best_partial_sum.sum_hess, 0.5);
  } else {
    EXPECT_EQ(out_scan[2].best_fvalue, 2.0f);
    EXPECT_FLOAT_EQ(out_scan[2].best_partial_sum.sum_grad, 1.0);
    EXPECT_FLOAT_EQ(out_scan[2].best_partial_sum.sum_hess, 0.5);
  }

  EXPECT_FLOAT_EQ(out_scan[3].best_loss_chg, 1.0f);
  EXPECT_EQ(out_scan[3].best_findex, 1);
  if (out_scan[3].best_direction == DefaultDirection::kRightDir) {
    EXPECT_EQ(out_scan[3].best_fvalue, 11.0f);
    EXPECT_FLOAT_EQ(out_scan[3].best_partial_sum.sum_grad, -0.5);
    EXPECT_FLOAT_EQ(out_scan[3].best_partial_sum.sum_hess, 0.5);
  } else {
    EXPECT_EQ(out_scan[3].best_fvalue, 12.0f);
    EXPECT_FLOAT_EQ(out_scan[3].best_partial_sum.sum_grad, 0.5);
    EXPECT_FLOAT_EQ(out_scan[3].best_partial_sum.sum_hess, 0.5);
  }
}

TEST(EvaluateSplits, E2ETreeStump) {
  EvaluateSplitsExample example;
  auto split_inputs = GetTreeStumpExample(example);
  EvaluateSplitInputs left = thrust::get<0>(split_inputs);
  EvaluateSplitInputs right = thrust::get<1>(split_inputs);
  std::vector<SplitCandidate> out_splits(2);
  SplitEvaluator<TrainingParam> evaluator;
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
    EXPECT_EQ(result_left.fvalue, 11.0f);
  } else {
    EXPECT_EQ(result_left.fvalue, 12.0f);
  }

  SplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_FLOAT_EQ(result_right.loss_chg, 4.0f);
  if (result_right.dir == DefaultDirection::kRightDir) {
    EXPECT_EQ(result_right.fvalue, 1.0f);
  } else {
    EXPECT_EQ(result_right.fvalue, 2.0f);
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

  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSingleSplit(ToSpan(out_split), evaluator, EvaluateSplitInputs<GradStats>{});

  SplitCandidate result = out_split[0];
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 produces the best split, but the algorithm must account for feature sampling
TEST(EvaluateSplits, EvaluateSplitsWithFeatureSampling) {
  std::vector<SplitCandidate> out_splits(2);
  GradStats parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  std::vector<bst_feature_t> feature_set_left{2};
  std::vector<bst_feature_t> feature_set_right{1};
  std::vector<uint32_t> feature_segments{0, 2, 4, 6};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0, 100.0, 200.0};
  std::vector<float> feature_min_values{0.0, 10.0, 98.0};
  std::vector<GradStats> feature_histogram{
    {-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}
  };
  EvaluateSplitInputs<GradStats> left{
    1,
    parent_sum,
    param,
    ToSpan(feature_set_left),
    {},
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};
  EvaluateSplitInputs<GradStats> right{
    2,
    parent_sum,
    param,
    ToSpan(feature_set_right),
    {},
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};

  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSplits(ToSpan(out_splits), evaluator, left, right);

  EXPECT_EQ(out_splits[0].findex, 2);
  if (out_splits[0].dir == DefaultDirection::kRightDir) {
    EXPECT_EQ(out_splits[0].fvalue, 100.0f);
  } else {
    EXPECT_EQ(out_splits[0].fvalue, 200.0f);
  }
  EXPECT_FLOAT_EQ(out_splits[0].loss_chg, 4.0f);
  EXPECT_FLOAT_EQ(out_splits[0].left_sum.sum_grad, -1.0);
  EXPECT_FLOAT_EQ(out_splits[0].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_splits[0].right_sum.sum_grad, 1.0);
  EXPECT_FLOAT_EQ(out_splits[0].right_sum.sum_hess, 0.5);

  EXPECT_EQ(out_splits[1].findex, 1);
  if (out_splits[1].dir == DefaultDirection::kRightDir) {
    EXPECT_EQ(out_splits[1].fvalue, 11.0f);
  } else {
    EXPECT_EQ(out_splits[1].fvalue, 12.0f);
  }
  EXPECT_FLOAT_EQ(out_splits[1].loss_chg, 1.0f);
  EXPECT_FLOAT_EQ(out_splits[1].left_sum.sum_grad, -0.5);
  EXPECT_FLOAT_EQ(out_splits[1].left_sum.sum_hess, 0.5);
  EXPECT_FLOAT_EQ(out_splits[1].right_sum.sum_grad, 0.5);
  EXPECT_FLOAT_EQ(out_splits[1].right_sum.sum_hess, 0.5);
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(EvaluateSplits, EvaluateSingleSplitBreakTies) {
  std::vector<SplitCandidate> out_splits(1);
  GradStats parent_sum{0.0, 1.0};
  TrainingParam param{0.0f};

  std::vector<bst_feature_t> feature_set{0, 1};
  std::vector<uint32_t> feature_segments{0, 2, 4};
  std::vector<float> feature_values{1.0, 2.0, 11.0, 12.0};
  std::vector<float> feature_min_values{0.0, 10.0};
  std::vector<GradStats> feature_histogram{
    {-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  EvaluateSplitInputs<GradStats> input{
    1,
    parent_sum,
    param,
    ToSpan(feature_set),
    {},
    ToSpan(feature_segments),
    ToSpan(feature_values),
    ToSpan(feature_min_values),
    ToSpan(feature_histogram)};

  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSingleSplit(ToSpan(out_splits), evaluator, input);

  SplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
}

TEST(EvaluateSplits, E2ETreeStumpSecondExample) {
  std::vector<SplitCandidate> out_splits(1);
  GradStats parent_sum{6.4, 12.8};
  TrainingParam param{0.01f};

  std::vector<bst_feature_t> feature_set{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint32_t> feature_segments{0, 3, 6, 9, 12, 15, 18, 21, 24};
  std::vector<float> feature_values{0.30f, 0.67f, 1.64f,
                                    0.32f, 0.77f, 1.95f,
                                    0.29f, 0.70f, 1.80f,
                                    0.32f, 0.75f, 1.85f,
                                    0.18f, 0.59f, 1.69f,
                                    0.25f, 0.74f, 2.00f,
                                    0.26f, 0.74f, 1.98f,
                                    0.26f, 0.71f, 1.83f};
  std::vector<float> feature_min_values{0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
  std::vector<GradStats> feature_histogram{
      {0.8314f, 0.7147f}, {1.7989f, 3.7312f}, {3.3846f, 3.4598f},
      {2.9277f, 3.5886f}, {1.8429f, 2.4152f}, {1.2443f, 1.9019f},
      {1.6380f, 2.9174f}, {1.5657f, 2.5107f}, {2.8111f, 2.4776f},
      {2.1322f, 3.0651f}, {3.2927f, 3.8540f}, {0.5899f, 0.9866f},
      {1.5185f, 1.6263f}, {2.0686f, 3.1844f}, {2.4278f, 3.0950f},
      {1.5105f, 2.1403f}, {2.6922f, 4.2217f}, {1.8122f, 1.5437f},
      {0.0000f, 0.0000f}, {4.3245f, 5.7955f}, {1.6903f, 2.1103f},
      {2.4012f, 4.4754f}, {3.6136f, 3.4303f}, {0.0000f, 0.0000f}};
  EvaluateSplitInputs<GradStats> input{
      1,
      parent_sum,
      param,
      ToSpan(feature_set),
      {},
      ToSpan(feature_segments),
      ToSpan(feature_values),
      ToSpan(feature_min_values),
      ToSpan(feature_histogram)};

  SplitEvaluator<TrainingParam> evaluator;
  EvaluateSingleSplit(ToSpan(out_splits), evaluator, input);

  SplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 7);
  EXPECT_FLOAT_EQ(result.fvalue, 0.71f);
}
