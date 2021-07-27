#include "param.h"
#include "evaluator.h"
#include "iterator.h"
#include "helpers.h"
#include "scan.h"
#include "reduce.h"
#include <span>
#include <algorithm>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cassert>

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

struct ScanElem {
  std::size_t idx;
  GradientPair grad;
  SplitCandidate candidate;
};

template <bool forward>
struct ScanValueOp {
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  SplitEvaluator evaluator;

  ScanElem operator()(std::size_t idx) {
    ScanElem ret;
    ret.idx = idx;
    float fvalue;
    std::size_t fidx;
    bool is_cat;
    float loss_chg;
    if (idx < left.gradient_histogram.size()) {
      // left node
      ret.grad = left.gradient_histogram[idx];
      fvalue = left.feature_values[idx];
      fidx = SegmentId(left.feature_segments, idx);
      is_cat = IsCat(left.feature_types, fidx);

      float parent_gain = evaluator.CalcGain(left.param,GradStats{left.parent_sum});
          // FIXME: get it out
      float gain = evaluator.CalcSplitGain(left.param, left.nidx, fidx,
                                           GradStats{ret.grad},
                                           GradStats{left.parent_sum - ret.grad});
      loss_chg = gain - parent_gain;
    } else {
      // right node
      idx -= left.gradient_histogram.size();
      ret.grad = right.gradient_histogram[idx];
      fvalue = right.feature_values[idx];
      fidx = SegmentId(right.feature_segments, idx);
      is_cat = IsCat(right.feature_types, fidx);

      float parent_gain = evaluator.CalcGain(right.param,GradStats{right.parent_sum});
          // FIXME: get it out
      float gain = evaluator.CalcSplitGain(right.param, right.nidx, fidx,
                                           GradStats{ret.grad},
                                           GradStats{left.parent_sum - ret.grad});
      loss_chg = gain - parent_gain;
    }
    if (forward) {
      ret.candidate.Update(
          loss_chg, kRightDir, fvalue, fidx, GradientPair{ret.grad},
          GradientPair{left.parent_sum - ret.grad}, is_cat, left.param);
    } else {
      ret.candidate.Update(loss_chg, kLeftDir, fvalue, fidx,
                           GradientPair{left.parent_sum - ret.grad},
                           GradientPair{ret.grad}, is_cat, left.param);
    }
    return ret;
  }
};

struct WriteScan {
  using TupleT = typename DiscardIterator<std::tuple<ScanElem, ScanElem>>::OutputT;
  EvaluateSplitInputs left;
  EvaluateSplitInputs right;
  std::span<ScanElem> out_scan;
  std::size_t n_features;

  template <bool forward>
  void DoIt(const ScanElem& candidate) {
    std::size_t offset = 0;
    std::size_t beg_idx = 0;
    std::size_t end_idx = 0;

    auto fidx = candidate.candidate.findex;
    auto idx = candidate.idx;

    if (idx < left.gradient_histogram.size()) {
      // left node
      beg_idx = left.feature_segments[fidx];
      auto f_size = left.feature_segments[fidx + 1] - beg_idx;
      f_size = f_size == 0 ? 0 : f_size - 1;
      end_idx = beg_idx + f_size;
    } else {
      // right node
      beg_idx = right.feature_segments[fidx];
      auto f_size = right.feature_segments[fidx + 1] - beg_idx;
      f_size = f_size == 0 ? 0 : f_size - 1;
      end_idx = beg_idx + f_size;
      offset = n_features * 2;
    }
    if (forward) {
      if (end_idx == idx) {
        out_scan[offset + fidx] = candidate;
      }
    } else {
      if (beg_idx == idx) {
        out_scan[offset + n_features + fidx] = candidate;
      }
    }
  }

  TupleT operator()(const TupleT& tu) {
    const ScanElem& fw = std::get<0>(tu);
    const ScanElem& bw = std::get<1>(tu);
    if (fw.candidate.findex != -1) {
      DoIt<true>(fw);
    }
    if (bw.candidate.findex != -1) {
      DoIt<false>(bw);
    }
    return {};  // discard
  }
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
      SplitCandidate l_split, bst_feature_t fidx) const {
    SplitCandidate best;
    float gain = evaluator.CalcSplitGain(
        input.param, input.nidx, fidx, GradStats{l_gpair}, GradStats{r_gpair});
    best.Update(l_split, input.param);
    float parent_gain = evaluator.CalcGain(input.param, GradStats{input.parent_sum});
        // FIXME: get it out
    float loss_chg = gain - parent_gain;
    float fvalue = input.feature_values[idx];

    if (forward) {
      best.Update(loss_chg, kRightDir, fvalue, fidx, GradientPair{l_gpair},
                  GradientPair{r_gpair}, is_cat, input.param);
    } else {
      best.Update(loss_chg, kLeftDir, fvalue, fidx, GradientPair{r_gpair},
                  GradientPair{l_gpair}, is_cat, input.param);
    }

    return best;
  }

  template <bool forward>
  ScanElem Scan(const ScanElem& l, const ScanElem& r) const {
    SplitCandidate l_split = l.candidate;

    if (l.idx < left.gradient_histogram.size()) {
      // Left node
      auto r_idx = r.idx;

      auto l_fidx = SegmentId(left.feature_segments, l.idx);
      auto r_fidx = SegmentId(left.feature_segments, r.idx);
      /* Segmented scan with 2 segments
       * *****|******
       * 0, 1 |  2, 3
       *   /|_|_/| /|
       * 0, 1 |  2, 5
       * *****|******
       */
      if (l_fidx != r_fidx) {
        // Segmented scan
        return r;
      }

      assert(!left.feature_set.empty());
      if ((left.feature_set.size() != left.feature_segments.size() - 1) &&
          !std::binary_search(left.feature_set.begin(),
                              left.feature_set.end(), l_fidx)) {
        // column sampling
        return {r_idx, r.grad, SplitCandidate{}};
      }

      if (IsCat(left.feature_types, l_fidx)) {
        auto l_gpair = left.gradient_histogram[l.idx];
        auto r_gpair = left.parent_sum - l_gpair;
        auto best = DoIt<forward, true>(left, l.idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, r_gpair, best};
      } else {
        auto l_gpair = l.grad;
        auto r_gpair = left.parent_sum - l_gpair;
        SplitCandidate best = DoIt<forward, false>(left, l.idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, l_gpair + r.grad, best};
      }
    } else {
      // Right node
      assert(left.gradient_histogram.size() == right.gradient_histogram.size());
      auto l_idx = l.idx - left.gradient_histogram.size();
      auto r_idx = r.idx - left.gradient_histogram.size();

      auto l_fidx = SegmentId(right.feature_segments, l_idx);
      auto r_fidx = SegmentId(right.feature_segments, r_idx);
      if (l_fidx != r_fidx) {
        // Segmented scan
        return {r.idx, r.grad, r.candidate};
      }

      assert(!right.feature_segments.empty());
      if ((right.feature_set.size() != right.feature_segments.size()) &&
          !std::binary_search(right.feature_set.begin(),
                              right.feature_set.end(), l_fidx)) {
        // column sampling
        return {r_idx, r.grad, SplitCandidate{}};
      }

      if (IsCat(right.feature_types, l_fidx)) {
        auto l_gpair = right.gradient_histogram[l_idx];
        auto r_gpair = right.parent_sum - l_gpair;
        auto best = DoIt<forward, true>(right, l_idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, r_gpair, best};
      } else {
        auto l_gpair = l.grad;
        auto r_gpair = right.parent_sum - l_gpair;
        auto best = DoIt<forward, false>(right, l_idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, l.grad + r.grad, best};
      }
    }
  }

  using Ty = std::tuple<ScanElem, ScanElem>;

  Ty operator()(Ty const &l, Ty const &r) const {
    auto fw = Scan<true>(std::get<0>(l), std::get<0>(r));
    auto bw = Scan<false>(std::get<1>(l), std::get<1>(r));
    return std::make_tuple(fw, bw);
  }
};

void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("");
  }
  auto n_features = l_n_features + r_n_features;

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto for_counting = MakeForwardCountingIterator<uint64_t>(0ul);
  auto rev_counting = MakeBackwardCountingIterator<uint64_t>(size);
  auto for_value_iter = MakeTransformIterator(for_counting,
                                              ScanValueOp<true>{left, right, evaluator});
  auto rev_value_iter = MakeTransformIterator(rev_counting,
                                              ScanValueOp<false>{left, right, evaluator});

  auto value_iter = MakeZipIterator(for_value_iter, rev_value_iter);
  static_assert(std::is_same_v<
      std::invoke_result_t<decltype(&decltype(value_iter)::Next), decltype(value_iter)>,
      std::tuple<ScanElem, ScanElem>>);
  std::vector<ScanElem> out_scan(n_features * 2);

  auto out_it = MakeTransformOutputIterator(
      DiscardIterator<std::tuple<ScanElem, ScanElem>>(),
      WriteScan{left, right, ToSpan(out_scan), l_n_features});
  InclusiveScan(value_iter, out_it, ScanOp{left, right, evaluator}, size);

  auto reduce_key = MakeTransformIterator(
      MakeForwardCountingIterator<bst_feature_t>(0),
      [=] (bst_feature_t fidx) -> int {
        if (fidx < l_n_features * 2) {
          return 0;  // left node
        } else {
          return 1;  // right node
        }
      });
  auto reduce_val = MakeTransformIterator(
      MakeForwardCountingIterator<std::size_t>(0),
       [out_scan](std::size_t idx) {
        // No need to distinguish left and right node as we are just extracting values.
        ScanElem candidate = out_scan[idx];
        return candidate.candidate;
      });
  ReduceByKey(
      reduce_key, out_scan.size(),
      reduce_val, DiscardIterator<int>(), OutputIterator(out_splits.begin()),
      [](int a, int b) { return (a == b); },
      [=](SplitCandidate l, SplitCandidate r) {
        l.Update(r, left.param);
        return l;
      });
}
