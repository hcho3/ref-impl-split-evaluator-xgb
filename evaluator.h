#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include <span>
#include <limits>
#include "param.h"

template <typename T>
inline static T Sqr(T a) {
  return a * a;
}

struct EvaluateSplitInputs {
  int nidx;
  GradientPair parent_sum;
  TrainingParam param;
  std::span<const bst_feature_t> feature_set;
  std::span<FeatureType const> feature_types;
  std::span<const uint32_t> feature_segments;
  std::span<const float> feature_values;
  std::span<const float> min_fvalue;
  std::span<const GradientPair> gradient_histogram;
};

struct SplitCandidate {
  float loss_chg{std::numeric_limits<float>::lowest()};
  DefaultDirection dir{kLeftDir};
  int findex{-1};
  float fvalue{0};
  bool is_cat{false};

  GradientPair left_sum;
  GradientPair right_sum;

  bool Update(const SplitCandidate& other, const TrainingParam& param) {
    if (other.loss_chg > loss_chg &&
        other.left_sum.hess_ >= param.min_child_weight &&
        other.right_sum.hess_ >= param.min_child_weight) {
      *this = other;
      return true;
    }
    return false;
  }

  bool Update(float loss_chg_in, DefaultDirection dir_in,
              float fvalue_in, int findex_in,
              GradientPair left_sum_in,
              GradientPair right_sum_in,
              bool cat,
              const TrainingParam& param) {
    if (loss_chg_in > loss_chg &&
        left_sum_in.hess_ >= param.min_child_weight &&
        right_sum_in.hess_ >= param.min_child_weight) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = fvalue_in;
      is_cat = cat;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
      return true;
    }
    return false;
  }
};

struct SplitEvaluator {
  double CalcSplitGain(const TrainingParam& param, bst_node_t,
                       bst_feature_t,
                       GradStats const& left,
                       GradStats const& right) const {
    double gain = CalcGain(param, left) + CalcGain(param, right);
    return gain;
  }

  double CalcGain(const TrainingParam&, const GradStats& stats) const {
    if (stats.sum_hess <= 0) {
      return 0.0;
    }
    return Sqr(stats.sum_grad) / stats.sum_hess;
  }
};

void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right);

#endif  // EVALUATOR_H_
