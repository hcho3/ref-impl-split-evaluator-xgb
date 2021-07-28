#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include <span>
#include <limits>
#include <vector>
#include "param.h"
#include "iterator.h"

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

  friend auto operator<<(std::ostream& os, SplitCandidate const& m) -> std::ostream& {
    std::string dir_s = (m.dir == kLeftDir) ? "left" : "right";
    os << "(loss_chg: " << m.loss_chg << ", dir: " << dir_s << ", findex: " << m.findex
       << ", fvalue: " << m.fvalue << ", is_cat: " << m.is_cat << ", left_sum: " << m.left_sum
       << ", right_sum: " << m.right_sum << ")";
    return os;
  }

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

struct ScanElem {
  std::size_t idx;
  GradientPair grad;
  SplitCandidate candidate;

  friend auto operator<<(std::ostream& os, ScanElem const& m) -> std::ostream& {
    os << "(idx: " << m.idx << ", grad: " << m.grad << ", candidate: " << m.candidate << ")";
    return os;
  }
};

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

template <bool forward>
struct ScanValueOp {
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  SplitEvaluator evaluator;

  ScanElem operator()(std::size_t idx);
};

struct ScanOp {
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  SplitEvaluator evaluator;

  ScanOp(EvaluateSplitInputs l, EvaluateSplitInputs r, SplitEvaluator e)
      : left(std::move(l)), right(std::move(r)), evaluator(std::move(e)) {}

  template <bool forward, bool is_cat>
  SplitCandidate DoIt(EvaluateSplitInputs input, std::size_t idx,
                      GradientPair l_gpair, GradientPair r_gpair,
                      SplitCandidate l_split, bst_feature_t fidx) const;
  template <bool forward>
  ScanElem Scan(const ScanElem& l, const ScanElem& r) const;
  using Ty = std::tuple<ScanElem, ScanElem>;
  Ty operator()(Ty const &l, Ty const &r) const;
};

struct WriteScan {
  using TupleT = typename DiscardIterator<std::tuple<ScanElem, ScanElem>>::OutputT;
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  std::span<ScanElem> out_scan;
  std::size_t n_features;

  template<bool forward>
  void DoIt(const ScanElem& candidate);
  TupleT operator()(const TupleT& tu);
};

inline decltype(auto) EvaluateSplitsGetIterator(SplitEvaluator evaluator, EvaluateSplitInputs left,
                                                EvaluateSplitInputs right) {
  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto for_counting = MakeForwardCountingIterator<uint64_t>(0ul);
  auto rev_counting = MakeBackwardCountingIterator<uint64_t>(size - 1);
  auto for_value_iter = MakeTransformIterator(for_counting,
                                              ScanValueOp<true>{left, right, evaluator});
  auto rev_value_iter = MakeTransformIterator(rev_counting,
                                              ScanValueOp<false>{left, right, evaluator});

  auto value_iter = MakeZipIterator(for_value_iter, rev_value_iter);
  return value_iter;
}

std::vector<ScanElem> EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator evaluator,
                                                             EvaluateSplitInputs left,
                                                             EvaluateSplitInputs right);

void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right);

#endif  // EVALUATOR_H_
