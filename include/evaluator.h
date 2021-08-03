#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "param.h"
#include <span>
#include <limits>
#include <vector>
#include <thrust/tuple.h>

template <typename T>
inline static T Sqr(T a) {
  return a * a;
}

template <typename IteratorT>
inline std::size_t SegmentId(IteratorT first, IteratorT last, std::size_t idx) {
  return std::upper_bound(first, last, idx) - 1 - first;
}

template <typename T>
inline std::size_t SegmentId(std::span<T> segments_ptr, std::size_t idx) {
  return SegmentId(segments_ptr.begin(), segments_ptr.end(), idx);
}

inline bool IsCat(std::span<FeatureType const> ft, bst_feature_t fidx) {
  return !ft.empty() && ft[fidx] == FeatureType::kCategorical;
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

struct SplitEvaluator {
  double CalcSplitGain(const TrainingParam& param, bst_node_t,
                       bst_feature_t,
                       const GradStats& left,
                       const GradStats& right) const {
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

struct ScanComputedElem {
  GradStats left_sum{0.0, 0.0};
  GradStats right_sum{0.0, 0.0};
  GradStats parent_sum{0.0, 0.0};
  float best_loss_chg{std::numeric_limits<float>::lowest()};
  int32_t best_findex{-1};
  float best_fvalue{std::numeric_limits<float>::quiet_NaN()};
  DefaultDirection best_direction{DefaultDirection::kLeftDir};

  friend std::ostream& operator<<(std::ostream& os, const ScanComputedElem& m);
  bool Update(GradStats left_sum_in,
              GradStats right_sum_in,
              GradStats parent_sum_in,
              float loss_chg_in,
              int32_t findex_in,
              float fvalue_in,
              DefaultDirection dir_in,
              const TrainingParam& param);
};

struct ScanElem {
  ChildNodeIndicator indicator{ChildNodeIndicator::kLeftChild};
  uint64_t hist_idx;
  GradientPair gpair{0.0, 0.0};
  int32_t findex{-1};
  float fvalue{std::numeric_limits<float>::quiet_NaN()};
  bool is_cat{false};
  ScanComputedElem computed_result{};

  friend std::ostream& operator<<(std::ostream& os, const ScanElem& m);
};

struct ScanValueOp {
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  SplitEvaluator evaluator;

  template <bool forward>
  ScanElem MapEvaluateSplitsHistEntryToScanElem(EvaluateSplitsHistEntry entry,
                                                EvaluateSplitInputs split_input);
  thrust::tuple<ScanElem, ScanElem> operator()(
      thrust::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup);
};

struct ScanOp {
  EvaluateSplitInputs left, right;
  SplitEvaluator evaluator;

  template<bool forward>
  ScanElem DoIt(ScanElem lhs, ScanElem rhs);
  thrust::tuple<ScanElem, ScanElem>
  operator() (thrust::tuple<ScanElem, ScanElem> lhs, thrust::tuple<ScanElem, ScanElem> rhs);
};

struct WriteScan {
  EvaluateSplitInputs left, right;
  SplitEvaluator evaluator;
  std::span<ScanComputedElem> out_scan;
  template <bool forward>
  void DoIt(ScanElem e);

  thrust::tuple<ScanElem, ScanElem>
  operator() (thrust::tuple<ScanElem, ScanElem> e);
};

std::vector<ScanComputedElem> EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator evaluator,
                                                                     EvaluateSplitInputs left,
                                                                     EvaluateSplitInputs right);
void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right);

#endif  // EVALUATOR_H_
