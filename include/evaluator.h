#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "param.h"
#include "helpers.h"
#include <span>
#include <limits>
#include <vector>
#include <thrust/tuple.h>

template <typename T>
inline static T Sqr(T a) {
  return a * a;
}

inline bool IsCat(std::span<FeatureType const> ft, bst_feature_t fidx) {
  return !ft.empty() && ft[fidx] == FeatureType::kCategorical;
}

template <typename GradientSumT>
struct EvaluateSplitInputs {
  int nidx;
  GradientSumT parent_sum;
  TrainingParam param;
  std::span<const bst_feature_t> feature_set;
  std::span<FeatureType const> feature_types;
  std::span<const uint32_t> feature_segments;
  std::span<const float> feature_values;
  std::span<const float> min_fvalue;
  std::span<const GradientSumT> gradient_histogram;
};

template <typename ParamT>
struct SplitEvaluator {
  double CalcSplitGain(const ParamT& param, bst_node_t,
                       bst_feature_t,
                       const GradStats& left,
                       const GradStats& right) const {
    double gain = CalcGain(param, left) + CalcGain(param, right);
    return gain;
  }

  double CalcGain(const ParamT&, const GradStats& stats) const {
    if (stats.sum_hess <= 0) {
      return 0.0;
    }
    return Sqr(stats.sum_grad) / stats.sum_hess;
  }
};

struct EvaluateSplitsHistEntry {
  uint32_t node_idx;
  uint32_t hist_idx;
  bool forward;

  friend std::ostream& operator<<(std::ostream& os, const EvaluateSplitsHistEntry& m);
};

struct SplitCandidate {
  float loss_chg{std::numeric_limits<float>::lowest()};
  DefaultDirection dir{kLeftDir};
  int findex{-1};
  float fvalue{0};
  bool is_cat{false};

  GradStats left_sum, right_sum;

  friend std::ostream& operator<<(std::ostream& os, const SplitCandidate& m);
  bool Update(const SplitCandidate& other, const TrainingParam& param);
};

template <typename GradientSumT>
struct ScanElem {
  uint32_t node_idx;
  uint32_t hist_idx;
  int32_t findex{-1};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  bool forward{true};
  GradientSumT gpair{0.0, 0.0};
  GradientSumT partial_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};

  template <typename X>
  friend std::ostream& operator<<(std::ostream& os, const ScanElem<X>& m);
};

template <typename GradientSumT>
struct ScanValueOp {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  SplitEvaluator<TrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  ScanElemT MapEvaluateSplitsHistEntryToScanElem(
      EvaluateSplitsHistEntry entry,
      EvaluateSplitInputs<GradientSumT> split_input);
  ScanElemT operator() (EvaluateSplitsHistEntry entry);
};

template <typename GradientSumT>
struct ScanOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  SplitEvaluator<TrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  ScanElemT DoIt(ScanElemT lhs, ScanElemT rhs);
  ScanElemT operator() (ScanElemT lhs, ScanElemT rhs);
};

template <typename GradientSumT>
struct ReduceElem {
  GradientSumT partial_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};
  float loss_chg{std::numeric_limits<float>::lowest()};
  int32_t findex{-1};
  uint32_t node_idx{0};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  DefaultDirection direction{DefaultDirection::kLeftDir};

  template <typename X>
  friend std::ostream& operator<<(std::ostream& os, const ReduceElem<X>& m);
};

template <typename GradientSumT>
struct ReduceValueOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  SplitEvaluator<TrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;
  using ReduceElemT = ReduceElem<GradientSumT>;

  ReduceElemT DoIt(ScanElemT e);
  ReduceElemT operator() (ScanElemT e);
};

template <typename GradientSumT>
std::vector<ReduceElem<GradientSumT>>
EvaluateSplitsGenerateSplitCandidatesViaScan(
    SplitEvaluator<TrainingParam> evaluator,
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right);

template <typename GradientSumT>
void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator<TrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right);

template <typename GradientSumT>
void EvaluateSingleSplit(std::span<SplitCandidate> out_split,
                         SplitEvaluator<TrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input);

#endif  // EVALUATOR_H_
