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
#include <thrust/tuple.h>
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

  auto out_scan = EvaluateSplitsGenerateSplitCandidatesViaScan(evaluator, left, right);

  auto reduce_key = thrust::make_transform_iterator(
      thrust::make_counting_iterator<std::size_t>(0),
      [out_scan=ToSpan(out_scan)] (std::size_t i) {
        return out_scan[i].node_idx;
      });
  auto reduce_val = thrust::make_transform_iterator(
      thrust::make_counting_iterator<std::size_t>(0),
      [out_scan=ToSpan(out_scan)](std::size_t i) {
        const auto& e = out_scan[i];
        GradientSumT left_sum, right_sum;
        if (e.is_cat) {
          left_sum = e.parent_sum - e.partial_sum;
          right_sum = e.partial_sum;
        } else {
          if (e.direction == DefaultDirection::kRightDir) {
            left_sum = e.partial_sum;
            right_sum = e.parent_sum - e.partial_sum;
          } else {
            left_sum = e.parent_sum - e.partial_sum;
            right_sum = e.partial_sum;
          }
        }
        return SplitCandidate{e.loss_chg, e.direction, e.findex,
                              e.fvalue, e.is_cat, left_sum, right_sum};
      });
  /**
   * Perform segmented reduce to find the best split candidate per node.
   * Note that there will be THREE segments:
   * [segment for left child node] [segment for right child node] [segment for left child node]
   * This is due to how we perform forward and backward passes over the gradient histogram.
   */
  std::vector<SplitCandidate> out_reduce(3);
  thrust::reduce_by_key(
      reduce_key, reduce_key + static_cast<std::ptrdiff_t>(out_scan.size()), reduce_val,
      thrust::make_discard_iterator(), out_reduce.begin(),
      [](int a, int b) { return (a == b); },
      [=](SplitCandidate l, SplitCandidate r) {
        l.Update(r, left.param);
        return l;
      });
  if (right.gradient_histogram.empty()) {
    out_splits[0] = out_reduce[0];
  } else {
    out_reduce[0].Update(out_reduce[2], left.param);
    out_splits[0] = out_reduce[0];
    out_splits[1] = out_reduce[1];
  }
}

template <typename GradientSumT>
void EvaluateSingleSplit(std::span<SplitCandidate> out_split,
                         SplitEvaluator<TrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

template <typename GradientSumT>
std::vector<ReduceElem<GradientSumT>>
EvaluateSplitsGenerateSplitCandidatesViaScan(SplitEvaluator<TrainingParam> evaluator,
                                             EvaluateSplitInputs<GradientSumT> left,
                                             EvaluateSplitInputs<GradientSumT> right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  // CHECK_LE(size, static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
  auto count_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint32_t>(0),
      [size = static_cast<uint32_t>(size)] __device__(uint32_t i) {
        // Generate sequence of length (size * 2):
        // 0 1 2 3 ... (size-2) (size-1) (size-1) (size-2) ... 2 1 0
        if (i < size) {
          return thrust::make_tuple(i, true);
        } else if (i < size * 2) {
          // size <= size < size * 2
          return thrust::make_tuple(size * 2 - 1 - i, false);
        } else {
          return thrust::make_tuple(static_cast<uint32_t>(0), false);
            // out-of-bounds, just return 0
        }
      });
  auto map_to_hist_bin = [&left](thrust::tuple<uint32_t, bool> e) {
    // The first (size) outputs will be of the forward pass
    // The following (size) outputs will be of the backward pass
    const auto left_hist_size = static_cast<uint32_t>(left.gradient_histogram.size());
    uint32_t idx = thrust::get<0>(e);
    bool forward = thrust::get<1>(e);
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{0, idx, forward};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{1, idx - left_hist_size, forward};
    }
  };
  auto bin_iter = thrust::make_transform_iterator(count_iter, map_to_hist_bin);
  auto scan_input_iter = thrust::make_transform_iterator(
      bin_iter, ScanValueOp<GradientSumT>{left, right, evaluator});

  std::vector<ReduceElem<GradientSumT>> out_scan(size * 2);
  auto scan_output_iter = thrust::make_transform_output_iterator(
      out_scan.begin(), ReduceValueOp<GradientSumT>{left, right, evaluator});
  InclusiveScan(scan_input_iter, scan_output_iter, ScanOp<GradientSumT>{left, right, evaluator},
                size * 2);

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
  ret.findex = static_cast<int32_t>(SegmentId(split_input.feature_segments, entry.hist_idx));
  if (entry.forward) {
    ret.fvalue = split_input.feature_values[entry.hist_idx];
  } else {
    if (entry.hist_idx > 0) {
      ret.fvalue = split_input.feature_values[entry.hist_idx - 1];
    } else {
      ret.fvalue = split_input.min_fvalue[ret.findex];
    }
  }
  ret.is_cat = IsCat(split_input.feature_types, ret.findex);
  ret.forward = entry.forward;
  ret.gpair = split_input.gradient_histogram[entry.hist_idx];
  ret.parent_sum = split_input.parent_sum;
  if (((entry.node_idx == 0) &&
       (left.feature_set.size() != left.feature_segments.size()) &&
       !std::binary_search(left.feature_set.begin(),
                           left.feature_set.end(), ret.findex)) ||
      ((entry.node_idx == 1) &&
       (right.feature_set.size() != right.feature_segments.size()) &&
       !std::binary_search(right.feature_set.begin(),
                           right.feature_set.end(), ret.findex))) {
    // Column sampling
    return ret;
  }
  ret.partial_sum = ret.gpair;

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
  if (lhs.findex != rhs.findex || lhs.node_idx != rhs.node_idx || lhs.forward != rhs.forward) {
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

  ret = rhs;
  ret.partial_sum = lhs.partial_sum + rhs.gpair;
  return ret;
}

template <typename GradientSumT>
ScanElem<GradientSumT>
ScanOp<GradientSumT>::operator() (ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  return DoIt(lhs, rhs);
};

template <typename GradientSumT>
ReduceElem<GradientSumT>
ReduceValueOp<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
  ReduceElem<GradientSumT> ret;
  if (e.is_cat) {
    ret.partial_sum = e.gpair;
  } else {
    ret.partial_sum = e.partial_sum;
  }
  ret.parent_sum = e.parent_sum;
  {
    GradientSumT left_sum, right_sum;
    if (e.is_cat) {
      left_sum = e.parent_sum - e.gpair;
      right_sum = e.gpair;
    } else {
      if (e.forward) {
        left_sum = e.partial_sum;
        right_sum = e.parent_sum - e.partial_sum;
      } else {
        left_sum = e.parent_sum - e.partial_sum;
        right_sum = e.partial_sum;
      }
    }
    bst_node_t nidx = (e.node_idx == 0) ? left.nidx : right.nidx;
    float gain = evaluator.CalcSplitGain(left.param, nidx, e.findex, left_sum, right_sum);
    float parent_gain = evaluator.CalcGain(left.param, e.parent_sum);
    ret.loss_chg = gain - parent_gain;
  }
  ret.findex = e.findex;
  ret.node_idx = e.node_idx;
  ret.fvalue = e.fvalue;
  ret.is_cat = e.is_cat;
  ret.direction = (e.forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
  return ret;
}

template <typename GradientSumT>
ReduceElem<GradientSumT>
ReduceValueOp<GradientSumT>::operator() (ScanElem<GradientSumT> e) {
  return DoIt(e);
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
  os << "(node_idx: " << m.node_idx << ", hist_idx: " << m.hist_idx
     << ", forward: " << (m.forward ? "true" : "false") << ")";
  return os;
}

template <typename GradientSumT>
std::ostream& operator<<(std::ostream& os, const ReduceElem<GradientSumT>& m) {
  std::string direction_str = (m.direction == DefaultDirection::kLeftDir) ? "left" : "right";
  os << "(partial_sum: " << m.partial_sum << ", parent_sum: " << m.parent_sum
     << ", loss_chg: " << m.loss_chg << ", findex: " << m.findex
     << ", node_idx: " << m.node_idx << ", fvalue: " << m.fvalue
     << ", is_cat: " << m.is_cat << ", direction: " << direction_str << ")";
  return os;
}

template <typename GradientSumT>
std::ostream& operator<<(std::ostream& os, const ScanElem<GradientSumT>& m) {
  os << "(node_idx: " << m.node_idx << ", hist_idx: " << m.hist_idx
     << ", findex: " << m.findex << ", fvalue: " << m.fvalue
     << ", is_cat: " << (m.is_cat ? "true" : "false")
     << ", forward: " << (m.forward ? "true" : "false")
     << ", gpair: " << m.gpair
     << ", partial_sum: " << m.partial_sum << ", parent_sum: " << m.parent_sum << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SplitCandidate& m) {
  std::string dir_s = (m.dir == kLeftDir) ? "left" : "right";
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

template std::ostream& operator<<(std::ostream& os, const ReduceElem<GradStats>& m);
template std::ostream& operator<<(std::ostream& os, const ScanElem<GradStats>& m);
