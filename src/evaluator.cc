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
#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cassert>

template <bool forward>
ScanElem
ScanValueOp<forward>::operator() (std::size_t idx) {
  ScanElem ret;
  ret.idx = idx;
  float fvalue;
  std::size_t fidx;
  bool is_cat;
  float loss_chg;
  GradientPair left_sum, right_sum;
  bool is_segment_begin = false;
  if (idx < left.gradient_histogram.size()) {
    // left node
    ret.grad = left.gradient_histogram[idx];
    fvalue = left.feature_values[idx];
    fidx = SegmentId(left.feature_segments, idx);
    is_cat = IsCat(left.feature_types, fidx);
    if ((forward && left.feature_segments[fidx] == idx) ||
        (!forward && left.feature_segments[fidx + 1] - 1 == idx)) {
      is_segment_begin = true;
      if (forward) {
        left_sum = ret.grad;
        right_sum = left.parent_sum - ret.grad;
      } else {
        left_sum = left.parent_sum - ret.grad;
        right_sum = ret.grad;
      }
      float parent_gain = evaluator.CalcGain(left.param, GradStats{left.parent_sum});
      float gain = evaluator.CalcSplitGain(left.param, left.nidx, fidx,
                                           GradStats{left_sum}, GradStats{right_sum});
      loss_chg = gain - parent_gain;
    }
  } else {
    // right node
    idx -= left.gradient_histogram.size();
    ret.grad = right.gradient_histogram[idx];
    fvalue = right.feature_values[idx];
    fidx = SegmentId(right.feature_segments, idx);
    is_cat = IsCat(right.feature_types, fidx);
    if ((forward && right.feature_segments[fidx] == idx) ||
        (!forward && right.feature_segments[fidx + 1] - 1 == idx)) {
      is_segment_begin = true;
      if (forward) {
        left_sum = ret.grad;
        right_sum = right.parent_sum - ret.grad;
      } else {
        left_sum = right.parent_sum - ret.grad;
        right_sum = ret.grad;
      }
      float parent_gain = evaluator.CalcGain(right.param, GradStats{right.parent_sum});
      float gain = evaluator.CalcSplitGain(right.param, right.nidx, fidx,
                                           GradStats{left_sum}, GradStats{right_sum});
      loss_chg = gain - parent_gain;
    }
  }
  if (!is_segment_begin) {
    loss_chg = std::numeric_limits<float>::lowest();
    left_sum = {0, 0};
    right_sum = {0, 0};
  }
  ret.candidate = {loss_chg, (forward ? kRightDir : kLeftDir), static_cast<int>(fidx), fvalue,
                   is_cat, left_sum, right_sum};
  return ret;
}

template <bool forward, bool is_cat>
SplitCandidate
ScanOp::DoIt(EvaluateSplitInputs input, std::size_t idx, GradientPair l_gpair, GradientPair r_gpair,
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
ScanElem
ScanOp::Scan(const ScanElem& l, const ScanElem& r) const {
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

ScanOp::Ty
ScanOp::operator() (ScanOp::Ty const &l, ScanOp::Ty const &r) const {
  auto fw = Scan<true>(std::get<0>(l), std::get<0>(r));
  auto bw = Scan<false>(std::get<1>(l), std::get<1>(r));
  return std::make_tuple(fw, bw);
}

template <bool forward>
void
WriteScan::DoIt(const ScanElem& candidate) {
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

WriteScan::TupleT
WriteScan::operator()(const WriteScan::TupleT& tu) {
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

std::vector<ScanElem> EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator evaluator,
                                                             EvaluateSplitInputs left,
                                                             EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  auto n_features = l_n_features + r_n_features;
  auto value_iter = EvaluateSplitsGetIterator(evaluator, left, right);
  static_assert(std::is_same_v<
      std::invoke_result_t<decltype(&decltype(value_iter)::Get), decltype(value_iter)>,
      std::tuple<ScanElem, ScanElem>>);
  std::vector<ScanElem> out_scan(n_features * 2);

  auto out_it = MakeTransformOutputIterator(
      DiscardIterator<std::tuple<ScanElem, ScanElem>>(),
      WriteScan{left, right, ToSpan(out_scan), l_n_features});

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  InclusiveScan(value_iter, out_it, ScanOp{left, right, evaluator}, size);
  return out_scan;
}

void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("");
  }

  auto out_scan = EvaluateSplitsFindOptimalSplitsViaScan(evaluator, left, right);

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
      reduce_val, DiscardIterator<int>(), OutputIterator(out_splits.begin(), out_splits.end()),
      [](int a, int b) { return (a == b); },
      [=](SplitCandidate l, SplitCandidate r) {
        l.Update(r, left.param);
        return l;
      });
}

template struct ScanValueOp<true>;
template struct ScanValueOp<false>;
template SplitCandidate ScanOp::DoIt<true, true>(EvaluateSplitInputs, std::size_t, GradientPair,
    GradientPair, SplitCandidate, bst_feature_t) const;
template SplitCandidate ScanOp::DoIt<true, false>(EvaluateSplitInputs, std::size_t, GradientPair,
    GradientPair, SplitCandidate, bst_feature_t) const;
template SplitCandidate ScanOp::DoIt<false, true>(EvaluateSplitInputs, std::size_t, GradientPair,
    GradientPair, SplitCandidate, bst_feature_t) const;
template SplitCandidate ScanOp::DoIt<false, false>(EvaluateSplitInputs, std::size_t, GradientPair,
    GradientPair, SplitCandidate, bst_feature_t) const;
template ScanElem ScanOp::Scan<true>(const ScanElem&, const ScanElem&) const;
template ScanElem ScanOp::Scan<false>(const ScanElem&, const ScanElem&) const;
template void WriteScan::DoIt<true>(const ScanElem&);
template void WriteScan::DoIt<false>(const ScanElem&);
