#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "param.h"
#include "evaluator.h"
#include "helpers.h"

TEST(EvaluateSplits, TreeStump) {
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

  EvaluateSplitInputs input_left{
      1,
      parent_sum,
      param,
      ToSpan(feature_set),
      {},
      ToSpan(feature_segments),
      ToSpan(feature_values),
      ToSpan(feature_min_values),
      ToSpan(feature_histogram_left)};
  EvaluateSplitInputs input_right{
      2,
      parent_sum,
      param,
      ToSpan(feature_set),
      {},
      ToSpan(feature_segments),
      ToSpan(feature_values),
      ToSpan(feature_min_values),
      ToSpan(feature_histogram_right)};

  std::vector<SplitCandidate> out_splits(2);
  SplitEvaluator evaluator;
  EvaluateSplits(ToSpan(out_splits), evaluator, input_left, input_right);

  SplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);
  EXPECT_FLOAT_EQ(result_left.loss_chg, 4.0f);

  SplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
  EXPECT_FLOAT_EQ(result_right.loss_chg, 4.0f);
}
