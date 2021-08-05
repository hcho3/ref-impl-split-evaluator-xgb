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
#include <ostream>
#include <string>

void EvaluateSplits(std::span<SplitCandidate> out_splits,
                    SplitEvaluator evaluator,
                    EvaluateSplitInputs left,
                    EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
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
      [&out_scan, &left](std::size_t i) {
        ScanComputedElem c = out_scan.at(i);
        bool is_cat = IsCat(left.feature_types, c.best_findex);
        return SplitCandidate{c.best_loss_chg, c.best_direction, c.best_findex,
                              c.best_fvalue, is_cat, c.left_sum, c.right_sum};
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

void EvaluateSingleSplit(std::span<SplitCandidate> out_split,
                         SplitEvaluator evaluator,
                         EvaluateSplitInputs input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

std::vector<ScanComputedElem> EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator evaluator,
                                                                     EvaluateSplitInputs left,
                                                                     EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  auto map_to_left_right = [&left](uint64_t idx) {
    const auto left_hist_size = static_cast<uint64_t>(left.gradient_histogram.size());
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kLeftChild, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kRightChild, idx - left_hist_size};
    }
  };

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto for_count_iter = thrust::make_counting_iterator<uint64_t>(0);
  auto for_loc_iter = thrust::make_transform_iterator(for_count_iter, map_to_left_right);
  auto rev_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint64_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto rev_loc_iter = thrust::make_transform_iterator(rev_count_iter, map_to_left_right);
  auto zip_loc_iter = thrust::make_zip_iterator(thrust::make_tuple(for_loc_iter, rev_loc_iter));

  auto scan_input_iter = thrust::make_transform_iterator(
      zip_loc_iter, ScanValueOp{left, right, evaluator});
  std::vector<ScanComputedElem> out_scan(l_n_features + r_n_features);
  auto scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan{left, right, evaluator, ToSpan(out_scan)});
  InclusiveScan(scan_input_iter, scan_out_iter, ScanOp{left, right, evaluator}, size);
  return out_scan;
}

template <bool forward>
ScanElem
ScanValueOp::MapEvaluateSplitsHistEntryToScanElem(EvaluateSplitsHistEntry entry,
                                                  EvaluateSplitInputs split_input) {
  ScanElem ret;
  ret.indicator = entry.indicator;
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
    if (forward) {
      ret.computed_result.left_sum = GradStats{ret.gpair};
      ret.computed_result.right_sum = GradStats{split_input.parent_sum} - GradStats{ret.gpair};
    } else {
      ret.computed_result.left_sum = GradStats{split_input.parent_sum} - GradStats{ret.gpair};
      ret.computed_result.right_sum = GradStats{ret.gpair};
    }
    ret.computed_result.parent_sum = GradStats{split_input.parent_sum};
    float parent_gain = evaluator.CalcGain(split_input.param, GradStats{split_input.parent_sum});
    float gain = evaluator.CalcSplitGain(split_input.param, split_input.nidx, ret.findex,
                                         ret.computed_result.left_sum,
                                         ret.computed_result.right_sum);
    ret.computed_result.best_loss_chg = gain - parent_gain;
    ret.computed_result.best_findex = ret.findex;
    ret.computed_result.best_fvalue = ret.fvalue;
    ret.computed_result.best_direction =
        (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
  }

  return ret;
}

thrust::tuple<ScanElem, ScanElem>
ScanValueOp::operator() (
    thrust::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup) {
  const auto& fw = thrust::get<0>(entry_tup);
  const auto& bw = thrust::get<1>(entry_tup);
  ScanElem ret_fw, ret_bw;
  ret_fw = MapEvaluateSplitsHistEntryToScanElem<true>(
      fw,
      (fw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  ret_bw = MapEvaluateSplitsHistEntryToScanElem<false>(
      bw,
      (bw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  return thrust::make_tuple(ret_fw, ret_bw);
}

template <bool forward>
ScanElem
ScanOp::DoIt(ScanElem lhs, ScanElem rhs) {
  ScanElem ret;
  ret = rhs;
  ret.computed_result = {};
  if (lhs.findex != rhs.findex || lhs.indicator != rhs.indicator) {
    // Segmented Scan
    return rhs;
  }
  if (((lhs.indicator == ChildNodeIndicator::kLeftChild) &&
       (left.feature_set.size() != left.feature_segments.size()) &&
       !std::binary_search(left.feature_set.begin(),
                           left.feature_set.end(), lhs.findex)) ||
      ((lhs.indicator == ChildNodeIndicator::kRightChild) &&
       (right.feature_set.size() != right.feature_segments.size()) &&
       !std::binary_search(right.feature_set.begin(),
                           right.feature_set.end(), lhs.findex))) {
    // Column sampling
    // FIXME: Test with column sampling enabled
    return rhs;
  }

  GradStats parent_sum = lhs.computed_result.parent_sum;
  GradStats left_sum, right_sum;
  if (forward) {
    if (lhs.is_cat) {  // FIXME: Must test with categorical splits
      left_sum = lhs.computed_result.parent_sum - GradStats{rhs.gpair};
      right_sum = GradStats{rhs.gpair};
    } else {
      left_sum = lhs.computed_result.left_sum + GradStats{rhs.gpair};
      right_sum = lhs.computed_result.parent_sum - left_sum;
    }
  } else {
    if (lhs.is_cat) {  // FIXME: Must test with categorical splits
      left_sum = lhs.computed_result.parent_sum - GradStats{rhs.gpair};
      right_sum = GradStats{rhs.gpair};
    } else {
      right_sum = lhs.computed_result.right_sum + GradStats{rhs.gpair};
      left_sum = lhs.computed_result.parent_sum - right_sum;
    }
  }
  bst_node_t nidx = (lhs.indicator == ChildNodeIndicator::kLeftChild) ? left.nidx : right.nidx;
  float gain = evaluator.CalcSplitGain(
      left.param, nidx, lhs.findex, left_sum, right_sum);
  float parent_gain = evaluator.CalcGain(left.param, parent_sum);
  float loss_chg = gain - parent_gain;
  ret.computed_result = lhs.computed_result;
  ret.computed_result.Update(left_sum, right_sum, parent_sum,
                             loss_chg, lhs.findex, lhs.fvalue,
                             (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir),
                             left.param);
  return ret;
}

thrust::tuple<ScanElem, ScanElem>
ScanOp::operator() (thrust::tuple<ScanElem, ScanElem> lhs, thrust::tuple<ScanElem, ScanElem> rhs) {
  const auto& lhs_fw = thrust::get<0>(lhs);
  const auto& lhs_bw = thrust::get<1>(lhs);
  const auto& rhs_fw = thrust::get<0>(rhs);
  const auto& rhs_bw = thrust::get<1>(rhs);
  return thrust::make_tuple(DoIt<true>(lhs_fw, rhs_fw), DoIt<false>(lhs_bw, rhs_bw));
};

template <bool forward>
void
WriteScan::DoIt(ScanElem e) {
  EvaluateSplitInputs& split_input =
      (e.indicator == ChildNodeIndicator::kLeftChild) ? left : right;
  std::size_t offset = 0;
  std::size_t n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  if (e.indicator == ChildNodeIndicator::kRightChild) {
    offset = n_features;
  }
  if ((!forward && split_input.feature_segments[e.findex] == e.hist_idx) ||
      (forward && split_input.feature_segments[e.findex + 1] - 1 == e.hist_idx)) {
    if (e.computed_result.best_loss_chg > out_scan[offset + e.findex].best_loss_chg) {
      out_scan[offset + e.findex] = e.computed_result;
    }
  }
}

thrust::tuple<ScanElem, ScanElem>
WriteScan::operator() (thrust::tuple<ScanElem, ScanElem> e) {
  const auto& fw = thrust::get<0>(e);
  const auto& bw = thrust::get<1>(e);
  DoIt<true>(fw);
  DoIt<false>(bw);
  return {};  // discard
}

bool
ScanComputedElem::Update(GradStats left_sum_in,
                         GradStats right_sum_in,
                         GradStats parent_sum_in,
                         float loss_chg_in,
                         int32_t findex_in,
                         float fvalue_in,
                         DefaultDirection dir_in,
                         const TrainingParam& param) {
  if (loss_chg_in > best_loss_chg &&
      left_sum_in.sum_hess >= param.min_child_weight &&
      right_sum_in.sum_hess >= param.min_child_weight) {
    best_loss_chg = loss_chg_in;
    best_findex = findex_in;
    best_fvalue = fvalue_in;
    best_direction = dir_in;
    left_sum = left_sum_in;
    right_sum = right_sum_in;
    parent_sum = parent_sum_in;
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
  std::string indicator_str =
      (m.indicator == ChildNodeIndicator::kLeftChild) ? "kLeftChild" : "kRightChild";
  os << "(indicator: " << indicator_str << ", hist_idx: " << m.hist_idx << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScanComputedElem& m) {
  std::string best_direction_str =
      (m.best_direction == DefaultDirection::kLeftDir) ? "left" : "right";
  os << "(left_sum: " << m.left_sum << ", right_sum: " << m.right_sum
     << ", parent_sum: " << m.parent_sum << ", best_loss_chg: " << m.best_loss_chg
     << ", best_findex: " << m.best_findex << ", best_fvalue: " << m.best_fvalue
     << ", best_direction: " << best_direction_str << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScanElem& m) {
  std::string indicator_str =
      (m.indicator == ChildNodeIndicator::kLeftChild) ? "kLeftChild" : "kRightChild";
  os << "(indicator: " << indicator_str << ", hist_idx: " << m.hist_idx
     << ", findex: " << m.findex<< ", gpair: " << m.gpair << ", fvalue: " << m.fvalue
     << ", is_cat: " << (m.is_cat ? "true" : "false")
     << ", computed_result: " << m.computed_result << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SplitCandidate& m) {
  std::string dir_s = (m.dir == kLeftDir) ? "left" : "right";
  os << "(loss_chg: " << m.loss_chg << ", dir: " << dir_s << ", findex: " << m.findex
     << ", fvalue: " << m.fvalue << ", is_cat: " << m.is_cat << ", left_sum: " << m.left_sum
     << ", right_sum: " << m.right_sum << ")";
  return os;
}
