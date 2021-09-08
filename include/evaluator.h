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

enum class ChildNodeIndicator : int8_t {
  kLeftChild, kRightChild
};

struct EvaluateSplitsHistEntry {
  ChildNodeIndicator indicator;
  uint64_t hist_idx;

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
struct ScanComputedElem {
  GradientSumT left_sum{0.0, 0.0};
  GradientSumT right_sum{0.0, 0.0};
  GradientSumT parent_sum{0.0, 0.0};
  GradientSumT best_left_sum{0.0, 0.0};
  GradientSumT best_right_sum{0.0, 0.0};
  float best_loss_chg{std::numeric_limits<float>::lowest()};
  int32_t best_findex{-1};
  bool is_cat{false};
  float best_fvalue{std::numeric_limits<float>::quiet_NaN()};
  DefaultDirection best_direction{DefaultDirection::kLeftDir};

  template <typename X>
  friend std::ostream& operator<<(std::ostream& os, const ScanComputedElem<X>& m);
  bool Update(GradientSumT left_sum_in,
              GradientSumT right_sum_in,
              GradientSumT parent_sum_in,
              float loss_chg_in,
              int32_t findex_in,
              bool is_cat_in,
              float fvalue_in,
              DefaultDirection dir_in,
              const TrainingParam& param);
};

template <typename GradientSumT>
struct ScanElem {
  ChildNodeIndicator indicator{ChildNodeIndicator::kLeftChild};
  uint64_t hist_idx;
  GradientSumT gpair{0.0, 0.0};
  int32_t findex{-1};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  std::span<ScanComputedElem<GradientSumT>> computed_result{};

  template <typename X>
  friend std::ostream& operator<<(std::ostream& os, const ScanElem<X>& m);
};

template <typename GradientSumT>
struct ScanValueOp {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  SplitEvaluator<TrainingParam> evaluator;
  std::span<ScanComputedElem<GradientSumT>> scratch_area;

  using ScanElemT = ScanElem<GradientSumT>;

  template <bool forward>
  ScanElemT MapEvaluateSplitsHistEntryToScanElem(
      EvaluateSplitsHistEntry entry,
      EvaluateSplitInputs<GradientSumT> split_input);
  thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup);
};

template <typename GradientSumT>
struct ScanOp {
  EvaluateSplitInputs<GradientSumT> left, right;
  SplitEvaluator<TrainingParam> evaluator;

  using ScanElemT = ScanElem<GradientSumT>;

  template<bool forward>
  ScanElem<GradientSumT> DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs);
  thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<ScanElemT, ScanElemT> lhs, thrust::tuple<ScanElemT, ScanElemT> rhs);
};

template <typename GradientSumT>
struct WriteScan {
  EvaluateSplitInputs<GradientSumT> left, right;
  SplitEvaluator<TrainingParam> evaluator;
  std::span<ScanComputedElem<GradientSumT>> out_scan;

  using ScanElemT = ScanElem<GradientSumT>;

  template <bool forward>
  void DoIt(ScanElemT e);

  thrust::tuple<ScanElemT, ScanElemT>
  operator() (thrust::tuple<ScanElemT, ScanElemT> e);
};

template <typename GradientSumT>
std::vector<ScanComputedElem<GradientSumT>>
EvaluateSplitsFindOptimalSplitsViaScan(
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
