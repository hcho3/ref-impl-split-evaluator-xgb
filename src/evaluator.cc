#include "evaluator.h"
#include "helpers.h"
#include "scan.h"
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <cstddef>
#include <algorithm>
#include <ostream>
#include <string>

template <typename GradientSumT>
void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator<TrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  // Handle empty (trivial) input
  if (l_n_features == 0 && r_n_features == 0) {
    std::fill(out_splits.begin(), out_splits.end(), SplitCandidate{});
    return;
  }

  auto out_scan = EvaluateSplitsFindOptimalSplitsViaScan(evaluator, left, right);

  auto reduce_key = thrust::make_transform_iterator(
      thrust::make_counting_iterator<bst_feature_t>(0),
      [=] (bst_feature_t i) -> int {
        if (i < l_n_features) {
          return 0;  // left node
        } else {
          return 1;  // right node
        }
      });
  auto reduce_val = thrust::make_transform_iterator(
      thrust::make_counting_iterator<std::size_t>(0),
      [&out_scan](std::size_t i) {
        ScanComputedElem<GradientSumT> c = out_scan.at(i);
        GradientSumT left_sum, right_sum;
        if (c.is_cat) {
          left_sum = c.parent_sum - c.best_partial_sum;
          right_sum = c.best_partial_sum;
        } else {
          if (c.best_direction == DefaultDirection::kRightDir) {
            left_sum = c.best_partial_sum;
            right_sum = c.parent_sum - c.best_partial_sum;
          } else {
            left_sum = c.parent_sum - c.best_partial_sum;
            right_sum = c.best_partial_sum;
          }
        }
        return SplitCandidate{c.best_loss_chg, c.best_direction, c.best_findex,
                              c.best_fvalue, c.is_cat, left_sum, right_sum};
      });
  thrust::reduce_by_key(
      reduce_key, reduce_key + static_cast<std::ptrdiff_t>(out_scan.size()), reduce_val,
      thrust::make_discard_iterator(), out_splits.data(),
      [](int a, int b) { return (a == b); },
      [=](SplitCandidate l, SplitCandidate r) {
        l.Update(r, left.param);
        return l;
      });
}

template <typename GradientSumT>
void EvaluateSingleSplit(std::span<SplitCandidate> out_split,
                         SplitEvaluator<TrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

template <typename GradientSumT>
std::vector<ScanComputedElem<GradientSumT>>
EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator<TrainingParam> evaluator,
                                       EvaluateSplitInputs<GradientSumT> left,
                                       EvaluateSplitInputs<GradientSumT> right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  auto map_to_hist_bin = [&left](uint32_t idx) {
    const auto left_hist_size = static_cast<uint32_t>(left.gradient_histogram.size());
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{0, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{1, idx - left_hist_size};
    }
  };

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto forward_count_iter = thrust::make_counting_iterator<uint32_t>(0);
  auto forward_bin_iter = thrust::make_transform_iterator(forward_count_iter, map_to_hist_bin);
  auto forward_scan_input_iter = thrust::make_transform_iterator(
      forward_bin_iter, ScanValueOp<GradientSumT>{true, left, right, evaluator});

  std::vector<ScanComputedElem<GradientSumT>> out_scan(l_n_features + r_n_features);
  auto forward_scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{true, left, right, evaluator, ToSpan(out_scan)});
  InclusiveScan(forward_scan_input_iter, forward_scan_out_iter,
                ScanOp<GradientSumT>{true, left, right, evaluator}, size);

  auto backward_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint32_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto backward_bin_iter = thrust::make_transform_iterator(backward_count_iter, map_to_hist_bin);
  auto backward_scan_input_iter = thrust::make_transform_iterator(
      backward_bin_iter, ScanValueOp<GradientSumT>{false, left, right, evaluator});
  auto backward_scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{false, left, right, evaluator, ToSpan(out_scan)});
  InclusiveScan(backward_scan_input_iter, backward_scan_out_iter,
                ScanOp<GradientSumT>{false, left, right, evaluator}, size);

  return out_scan;
}

template <typename GradientSumT>
ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::MapEvaluateSplitsHistEntryToScanElem(
    EvaluateSplitsHistEntry entry,
    EvaluateSplitInputs<GradientSumT> split_input) {
  ScanElem<GradientSumT> ret;
  ret.node_idx = entry.node_idx;
  ret.hist_idx = entry.hist_idx;
  ret.gpair = split_input.gradient_histogram[entry.hist_idx];
  ret.findex = static_cast<int32_t>(SegmentId(split_input.feature_segments, entry.hist_idx));
  ret.fvalue = split_input.feature_values[entry.hist_idx];
  ret.is_cat = IsCat(split_input.feature_types, ret.findex);
  if ((forward && split_input.feature_segments[ret.findex] == entry.hist_idx) ||
      (!forward && split_input.feature_segments[ret.findex + 1] - 1 == entry.hist_idx)) {
    /**
     * For the element at the beginning of each segment, compute gradient sums and loss_chg
     * ahead of time. These will be later used by the inclusive scan operator.
     **/
    GradientSumT partial_sum = ret.gpair;
    GradientSumT complement_sum = split_input.parent_sum - partial_sum;
    GradientSumT *left_sum, *right_sum;
    if (ret.is_cat) {
      left_sum = &complement_sum;
      right_sum = &partial_sum;
    } else {
      if (forward) {
        left_sum = &partial_sum;
        right_sum = &complement_sum;
      } else {
        left_sum = &complement_sum;
        right_sum = &partial_sum;
      }
    }
    ret.computed_result.parent_sum = partial_sum;
    ret.computed_result.best_partial_sum = partial_sum;
    ret.computed_result.parent_sum = split_input.parent_sum;
    float parent_gain = evaluator.CalcGain(split_input.param, split_input.parent_sum);
    float gain = evaluator.CalcSplitGain(split_input.param, split_input.nidx, ret.findex,
                                         *left_sum, *right_sum);
    ret.computed_result.best_loss_chg = gain - parent_gain;
    ret.computed_result.best_findex = ret.findex;
    ret.computed_result.best_fvalue = ret.fvalue;
    ret.computed_result.best_direction =
        (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
    ret.computed_result.is_cat = ret.is_cat;
  }

  return ret;
}

template <typename GradientSumT>
ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::operator() (EvaluateSplitsHistEntry entry) {
  return MapEvaluateSplitsHistEntryToScanElem(
      entry, (entry.node_idx == 0 ? this->left : this->right));
}

template <typename GradientSumT>
ScanElem<GradientSumT>
ScanOp<GradientSumT>::DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  ScanElem<GradientSumT> ret;
  ret = rhs;
  ret.computed_result = {};
  if (lhs.findex != rhs.findex || lhs.node_idx != rhs.node_idx) {
    // Segmented Scan
    return rhs;
  }
  if (((lhs.node_idx == 0) &&
       (left.feature_set.size() != left.feature_segments.size()) &&
       !std::binary_search(left.feature_set.begin(),
                           left.feature_set.end(), lhs.findex)) ||
      ((lhs.node_idx == 1) &&
       (right.feature_set.size() != right.feature_segments.size()) &&
       !std::binary_search(right.feature_set.begin(),
                           right.feature_set.end(), lhs.findex))) {
    // Column sampling
    return rhs;
  }

  GradientSumT parent_sum = lhs.computed_result.parent_sum;
  GradientSumT partial_sum, complement_sum;
  GradientSumT *left_sum, *right_sum;
  if (lhs.is_cat) {
    partial_sum = rhs.gpair;
    complement_sum = lhs.computed_result.parent_sum - rhs.gpair;
    left_sum = &complement_sum;
    right_sum = &partial_sum;
  } else {
    partial_sum = lhs.computed_result.partial_sum + rhs.gpair;
    complement_sum = parent_sum - partial_sum;
    if (forward) {
      left_sum = &partial_sum;
      right_sum = &complement_sum;
    } else {
      left_sum = &complement_sum;
      right_sum = &partial_sum;
    }
  }
  bst_node_t nidx = (lhs.node_idx == 0) ? left.nidx : right.nidx;
  float gain = evaluator.CalcSplitGain(left.param, nidx, lhs.findex, *left_sum, *right_sum);
  float parent_gain = evaluator.CalcGain(left.param, parent_sum);
  float loss_chg = gain - parent_gain;
  ret.computed_result = lhs.computed_result;
  ret.computed_result.Update(partial_sum, parent_sum, loss_chg, rhs.findex, rhs.is_cat, rhs.fvalue,
                             (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir),
                             left.param);
  return ret;
}

template <typename GradientSumT>
ScanElem<GradientSumT>
ScanOp<GradientSumT>::operator() (ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  return DoIt(lhs, rhs);
};

template <typename GradientSumT>
void
WriteScan<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
  EvaluateSplitInputs<GradientSumT>& split_input = (e.node_idx == 0) ? left : right;
  std::size_t offset = 0;
  std::size_t n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  if (e.node_idx == 1) {
    offset = n_features;
  }
  if ((!forward && split_input.feature_segments[e.findex] == e.hist_idx) ||
      (forward && split_input.feature_segments[e.findex + 1] - 1 == e.hist_idx)) {
    if (e.computed_result.best_loss_chg > out_scan[offset + e.findex].best_loss_chg) {
      out_scan[offset + e.findex] = e.computed_result;
    }
  }
}

template <typename GradientSumT>
ScanElem<GradientSumT>
WriteScan<GradientSumT>::operator() (ScanElem<GradientSumT> e) {
  DoIt(e);
  return {};  // discard
}

template <typename GradientSumT>
bool
ScanComputedElem<GradientSumT>::Update(
    GradientSumT partial_sum_in,
    GradientSumT parent_sum_in,
    float loss_chg_in,
    int32_t findex_in,
    bool is_cat_in,
    float fvalue_in,
    DefaultDirection dir_in,
    const TrainingParam& param) {
  partial_sum = partial_sum_in;
  parent_sum = parent_sum_in;
  if (loss_chg_in > best_loss_chg &&
      partial_sum_in.sum_hess >= param.min_child_weight &&
      (parent_sum_in.sum_hess - partial_sum_in.sum_hess) >= param.min_child_weight) {
    best_loss_chg = loss_chg_in;
    best_findex = findex_in;
    is_cat = is_cat_in;
    best_fvalue = fvalue_in;
    best_direction = dir_in;
    best_partial_sum = partial_sum_in;
    return true;
  }
  return false;
}

bool
SplitCandidate::Update(const SplitCandidate& other, const TrainingParam& param) {
  if (other.loss_chg > loss_chg &&
      other.left_sum.sum_hess >= param.min_child_weight &&
      other.right_sum.sum_hess >= param.min_child_weight) {
    *this = other;
    return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& os, const EvaluateSplitsHistEntry& m) {
  os << "(node_idx: " << m.node_idx << ", hist_idx: " << m.hist_idx << ")";
  return os;
}

template <typename GradientSumT>
std::ostream& operator<<(std::ostream& os, const ScanComputedElem<GradientSumT>& m) {
  std::string best_direction_str =
      (m.best_direction == DefaultDirection::kLeftDir) ? "left" : "right";
  os << "(is_cat: " << (m.is_cat ? "true" : "false") << ", best_direction: " << best_direction_str
     << ", best_findex: " << m.best_findex << ", best_loss_chg: " << m.best_loss_chg
     << ", best_fvalue: " << m.best_fvalue << ", partial_sum: " << m.partial_sum
     << ", best_partial_sum: " << m.best_partial_sum << ", parent_sum: " << m.parent_sum << ")";
  return os;
}

template <typename GradientSumT>
std::ostream& operator<<(std::ostream& os, const ScanElem<GradientSumT>& m) {
  os << "(node_idx: " << m.node_idx << ", hist_idx: " << m.hist_idx
     << ", gpair: " << m.gpair << ", findex: " << m.findex << ", fvalue: " << m.fvalue
     << ", is_cat: " << (m.is_cat ? "true" : "false")
     << ", computed_result: " << m.computed_result << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SplitCandidate& m) {
  std::string dir_s = (m.dir == DefaultDirection::kLeftDir) ? "left" : "right";
  os << "(loss_chg: " << m.loss_chg << ", dir: " << dir_s << ", findex: " << m.findex
     << ", fvalue: " << m.fvalue << ", is_cat: " << m.is_cat << ", left_sum: " << m.left_sum
     << ", right_sum: " << m.right_sum << ")";
  return os;
}

template void EvaluateSplits(
    std::span<SplitCandidate> out_splits,
    SplitEvaluator<TrainingParam> evaluator,
    EvaluateSplitInputs<GradStats> left,
    EvaluateSplitInputs<GradStats> right);

template void EvaluateSingleSplit(
    std::span<SplitCandidate> out_split,
    SplitEvaluator<TrainingParam> evaluator,
    EvaluateSplitInputs<GradStats> input);

template std::ostream& operator<<(std::ostream& os, const ScanElem<GradStats>& m);
