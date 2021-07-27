#include <iostream>
#include <vector>
#include <cstdint>
#include "param.h"
#include "evaluator.h"
#include "helpers.h"

int main() {
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
  for (SplitCandidate c : out_splits) {
    std::cout << "findex = " << c.findex << ", fvalue = " << c.fvalue
      << ", loss_chg = " << c.loss_chg << std::endl;
  }

  return 0;
}
