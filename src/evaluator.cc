#include "evaluator.h"
#include "helpers.h"
#include "scan.h"
#include <ostream>
#include <string>

std::vector<ScanComputedElem> EvaluateSplitsFindOptimalSplitsViaScan(SplitEvaluator evaluator,
                                                                     EvaluateSplitInputs left,
                                                                     EvaluateSplitInputs right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("");
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
  auto for_count_iter = MakeForwardCountingIterator<uint64_t>(0);
  auto for_loc_iter = MakeTransformIterator(for_count_iter, map_to_left_right);
  auto rev_count_iter = MakeBackwardCountingIterator<uint64_t>(static_cast<uint64_t>(size) - 1);
  auto rev_loc_iter = MakeTransformIterator(rev_count_iter, map_to_left_right);
  auto zip_loc_iter = MakeZipIterator(for_loc_iter, rev_loc_iter);

  auto scan_input_iter = MakeTransformIterator(zip_loc_iter, ScanValueOp{left, right, evaluator});
  std::vector<ScanComputedElem> out_scan(l_n_features * 2);
  auto scan_out_iter = MakeTransformOutputIterator(
      DiscardIterator<std::tuple<ScanElem, ScanElem>>(),
      WriteScan{left, right, evaluator, ToSpan(out_scan)});
  InclusiveScan(scan_input_iter, scan_out_iter, ScanOp{left, right, evaluator}, size);
  return out_scan;
}

template <bool forward>
ScanElem
ScanValueOp::MapEvaluateSplitsHistEntryToScanElem(EvaluateSplitsHistEntry entry,
                                                  EvaluateSplitInputs split_input) {
  ScanElem ret;
  ret.valid_entry = true;
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
    ret.computed = true;
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

std::tuple<ScanElem, ScanElem>
ScanValueOp::operator() (
    std::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup) {
  auto [fw, bw] = entry_tup;
  ScanElem ret_fw, ret_bw;
  ret_fw = MapEvaluateSplitsHistEntryToScanElem<true>(
      fw,
      (fw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  ret_bw = MapEvaluateSplitsHistEntryToScanElem<false>(
      bw,
      (bw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  return std::make_tuple(ret_fw, ret_bw);
}

template <bool forward>
ScanElem
ScanOp::DoIt(ScanElem lhs, ScanElem rhs) {
  ScanElem ret;
  ret = rhs;
  ret.computed_result = {};
  ret.computed = true;
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
  if (loss_chg > lhs.computed_result.best_loss_chg) {
    ret.computed_result.best_loss_chg = loss_chg;
    ret.computed_result.best_findex = lhs.findex;
    ret.computed_result.best_fvalue = lhs.fvalue;
    ret.computed_result.best_direction =
        (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
    ret.computed_result.left_sum = left_sum;
    ret.computed_result.right_sum = right_sum;
    ret.computed_result.parent_sum = parent_sum;
  } else {
    ret.computed_result.best_loss_chg = lhs.computed_result.best_loss_chg;
    ret.computed_result.best_findex = lhs.computed_result.best_findex;
    ret.computed_result.best_fvalue = lhs.computed_result.best_fvalue;
    ret.computed_result.best_direction = lhs.computed_result.best_direction;
    ret.computed_result.left_sum = lhs.computed_result.left_sum;
    ret.computed_result.right_sum = lhs.computed_result.right_sum;
    ret.computed_result.parent_sum = lhs.computed_result.parent_sum;
  }
  return ret;
}

std::tuple<ScanElem, ScanElem>
ScanOp::operator() (std::tuple<ScanElem, ScanElem> lhs, std::tuple<ScanElem, ScanElem> rhs) {
  auto [lhs_fw, lhs_bw] = lhs;
  auto [rhs_fw, rhs_bw] = rhs;
  return std::make_tuple(DoIt<true>(lhs_fw, rhs_fw), DoIt<false>(lhs_bw, rhs_bw));
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

std::tuple<ScanElem, ScanElem>
WriteScan::operator() (std::tuple<ScanElem, ScanElem> e) {
  auto [fw, bw] = e;
  DoIt<true>(fw);
  DoIt<false>(bw);
  return {};  // discard
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
     << ", findex: " << m.findex;
  if (m.valid_entry) {
    os << ", gpair: " << m.gpair << ", fvalue: " << m.fvalue
       << ", is_cat: " << (m.is_cat ? "true" : "false");
  }
  if (m.computed) {
    os << ", computed_result: " << m.computed_result;
  }
  os << ")";
  return os;
}
