#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <iostream>
#include <cstdint>
#include "param.h"
#include "evaluator.h"
#include "scan.h"
#include "helpers.h"

extern bool g_verbose_flag;

namespace {

struct SplitData {
  std::vector<bst_feature_t> feature_set;
  std::vector<uint32_t> feature_segments;
  std::vector<float> feature_values;
  std::vector<float> feature_min_values;
  std::vector<GradientPair> feature_histogram_left;
  std::vector<GradientPair> feature_histogram_right;
  GradientPair parent_sum;
  TrainingParam param;
};

std::tuple<EvaluateSplitInputs, EvaluateSplitInputs> GetTreeStumpExample(SplitData& data) {
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
  {
    SplitEvaluator evaluator;
    SplitData example;
    auto [left, right] = GetTreeStumpExample(example);
    auto for_counting = MakeForwardCountingIterator<uint64_t>(0ul);
    auto for_value_iter = MakeTransformIterator(for_counting,
                                                ScanValueOp<true>{left, right, evaluator});
    std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
    for (std::size_t i = 0; i < size; ++i) {
      auto e = for_value_iter.Get();
      if (g_verbose_flag) {
        std::cout << e << std::endl;
      }
      EXPECT_EQ(e.idx, i);
      EXPECT_EQ(e.candidate.dir, kRightDir);
      std::size_t idx =
          (i < left.gradient_histogram.size() ? i : i - left.gradient_histogram.size());
      EXPECT_EQ(e.candidate.findex, SegmentId(ToSpan(example.feature_segments), idx));
      EXPECT_EQ(e.candidate.fvalue, example.feature_values[idx]);
      if (example.feature_segments[e.candidate.findex] == idx) {
        // Elements at start of each segments gets a valid loss_chg
        EXPECT_GE(e.candidate.loss_chg, 0.0f);
      } else {
        // Other elements get invalid loss_chg
        EXPECT_LT(e.candidate.loss_chg, 0.0f);
      }
      GradientPair expected_grad;
      if (i < left.gradient_histogram.size()) {
        // left branch
        expected_grad = left.gradient_histogram[idx];
      } else {
        // right branch
        expected_grad = right.gradient_histogram[idx];
      }
      EXPECT_FLOAT_EQ(e.grad.grad_, expected_grad.grad_);
      EXPECT_FLOAT_EQ(e.grad.hess_, expected_grad.hess_);
      for_value_iter.Next();
    }
  }
  {
    SplitEvaluator evaluator;
    SplitData example;
    auto [left, right] = GetTreeStumpExample(example);
    std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
    auto rev_counting = MakeBackwardCountingIterator<uint64_t>(size - 1);
    auto rev_value_iter = MakeTransformIterator(rev_counting,
                                                ScanValueOp<false>{left, right, evaluator});
    for (std::size_t i = size - 1; i <= size - 1; --i) {
      auto e = rev_value_iter.Get();
      if (g_verbose_flag) {
        std::cout << e << std::endl;
      }
      EXPECT_EQ(e.idx, i);
      EXPECT_EQ(e.candidate.dir, kLeftDir);
      std::size_t idx =
          (i < left.gradient_histogram.size() ? i : i - left.gradient_histogram.size());
      EXPECT_EQ(e.candidate.findex, SegmentId(ToSpan(example.feature_segments), idx));
      EXPECT_EQ(e.candidate.fvalue, example.feature_values[idx]);
      if (example.feature_segments[e.candidate.findex + 1] - 1 == idx) {
        // Elements at start of each segments gets a valid loss_chg
        EXPECT_GE(e.candidate.loss_chg, 0.0f);
      } else {
        // Other elements get invalid loss_chg
        EXPECT_LT(e.candidate.loss_chg, 0.0f);
      }
      GradientPair expected_grad;
      if (i < left.gradient_histogram.size()) {
        // left branch
        expected_grad = left.gradient_histogram[idx];
      } else {
        // right branch
        expected_grad = right.gradient_histogram[idx];
      }
      EXPECT_FLOAT_EQ(e.grad.grad_, expected_grad.grad_);
      EXPECT_FLOAT_EQ(e.grad.hess_, expected_grad.hess_);
      rev_value_iter.Next();
    }
  }
}

TEST(EvaluateSplits, EvaluateSplitsInclusiveScan) {
  if (!g_verbose_flag) {
    return;
  }
  SplitEvaluator evaluator;
  SplitData example;
  auto [left, right] = GetTreeStumpExample(example);
  auto value_iter = EvaluateSplitsGetIterator(evaluator, left, right);
  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  std::vector<std::tuple<ScanElem, ScanElem>> out_scan(size);
  auto out_iter = OutputIterator(out_scan.begin(), out_scan.end());
  InclusiveScan(value_iter, out_iter, ScanOp{left, right, evaluator}, size);
  for (auto e : out_scan) {
    auto [x, y] = e;
    std::cout << "forward: " << x << std::endl;
    std::cout << "backward: " << y << std::endl;
  }
}

TEST(EvaluateSplits, FindOptimalSplitsViaScan) {
  SplitEvaluator evaluator;
  SplitData example;
  auto [left, right] = GetTreeStumpExample(example);
  std::vector<ScanElem> scan_elems = EvaluateSplitsFindOptimalSplitsViaScan(evaluator, left, right);

  for (ScanElem e : scan_elems) {
    std::cout << e << std::endl;
  }
}

TEST(EvaluateSplits, E2ETreeStump) {
  SplitData example;
  auto [left, right] = GetTreeStumpExample(example);
  std::vector<SplitCandidate> out_splits(2);
  SplitEvaluator evaluator;
  EvaluateSplits(ToSpan(out_splits), evaluator, left, right);

  SplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);
  EXPECT_FLOAT_EQ(result_left.loss_chg, 4.0f);

  SplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
  EXPECT_FLOAT_EQ(result_right.loss_chg, 4.0f);
}
