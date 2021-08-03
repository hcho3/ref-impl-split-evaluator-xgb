#include "scan.h"
#include <gtest/gtest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <vector>
#include <cstddef>

TEST(Scan, InclusiveScan) {
  {
    std::vector<int> vec{1, 0, 2, 2, 1, 3};
    std::vector<int> out(vec.size());
    auto scan_op = [](int x, int y) {
      return x + y;
    };
    InclusiveScan(vec.begin(), out.begin(), scan_op, vec.size());

    std::vector<int> expected_out{1, 1, 3, 5, 6, 9};
    EXPECT_EQ(out, expected_out);
  }
  {
    std::vector<int> vec{-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
    std::vector<int> out(vec.size());
    auto scan_op = [](int x, int y) {
      return std::max(x, y);
    };
    InclusiveScan(vec.begin(), out.begin(), scan_op, vec.size());

    std::vector<int> expected_out{-5, 0, 2, 2, 2, 4, 4, 4, 4, 8};
    EXPECT_EQ(out, expected_out);
  }
}

namespace {

struct SimpleSplitCandidate {
  int fidx;
  float fvalue;
  double loss_chg;
  bool operator==(const SimpleSplitCandidate& rhs) const {
    return (fidx == rhs.fidx) && (fvalue == rhs.fvalue) && (loss_chg == rhs.loss_chg);
  }
  friend std::ostream& operator<<(std::ostream& os, const SimpleSplitCandidate& m) {
    os << "(fidx: " << m.fidx << ", fvalue: " << m.fvalue << ", loss_chg: " << m.loss_chg << ")";
    return os;
  }
};

}  // anonymous namespace

TEST(Scan, InclusiveScanWithTuples) {
  std::vector<SimpleSplitCandidate> vec{
    {0, 2.0f, 0.0}, {1, 11.0f, 4.0}, {0, 1.0f, 1.0}, {1, 12.0f, 2.0}, {1, 10.0f, 4.0}
  };
  std::vector<SimpleSplitCandidate> out(vec.size());
  auto for_count_iter = thrust::make_counting_iterator<std::size_t>(0);
  auto for_iter = thrust::make_transform_iterator(for_count_iter,
                                                  [&vec](std::size_t idx) { return vec.at(idx); });
  auto scan_op = [](SimpleSplitCandidate x, SimpleSplitCandidate y) {
    if (x.loss_chg > y.loss_chg) {
      return x;
    } else if (y.loss_chg > x.loss_chg) {
      return y;
    }
    if (x.fidx != y.fidx) {
      return (x.fidx < y.fidx) ? x : y;
    }
    return (x.fvalue < y.fvalue) ? x : y;
  };
  InclusiveScan(for_iter, out.begin(), scan_op, vec.size());

  std::vector<SimpleSplitCandidate> expected_out{
      {0, 2.0f, 0.0}, {1, 11.0f, 4.0}, {1, 11.0f, 4.0}, {1, 11.0f, 4.0}, {1, 10.0f, 4.0}
  };
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out.at(i), expected_out.at(i)) << "Mismatch at index " << i;
  }
}

TEST(Scan, InclusiveScanWithTuplesForwardBackward) {
  std::vector<SimpleSplitCandidate> vec{
      {0, 2.0f, 0.0}, {1, 11.0f, 4.0}, {0, 1.0f, 1.0}, {1, 10.0f, 4.0}, {1, 12.0f, 2.0}
  };
  std::vector<thrust::tuple<SimpleSplitCandidate, SimpleSplitCandidate>> out(vec.size());
  auto for_count_iter = thrust::make_counting_iterator<std::size_t>(0);
  auto access_fn = [&vec](std::size_t idx) { return vec.at(idx); };
  auto for_iter = thrust::make_transform_iterator(for_count_iter, access_fn);
  auto rev_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator(0) + static_cast<ptrdiff_t>(vec.size()));
  auto rev_iter = thrust::make_transform_iterator(rev_count_iter, access_fn);
  auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(for_iter, rev_iter));
  auto inner_scan_op = [](SimpleSplitCandidate x, SimpleSplitCandidate y) {
    if (x.loss_chg > y.loss_chg) {
      return x;
    } else if (y.loss_chg > x.loss_chg) {
      return y;
    }
    if (x.fidx != y.fidx) {
      return (x.fidx < y.fidx) ? x : y;
    }
    return (x.fvalue < y.fvalue) ? x : y;
  };
  auto scan_op = [&inner_scan_op](
      thrust::tuple<SimpleSplitCandidate, SimpleSplitCandidate> x,
      thrust::tuple<SimpleSplitCandidate, SimpleSplitCandidate> y) {
    return thrust::make_tuple(inner_scan_op(thrust::get<0>(x), thrust::get<0>(y)),
                              inner_scan_op(thrust::get<1>(x), thrust::get<1>(y)));
  };
  InclusiveScan(zip_iter, out.begin(), scan_op, vec.size());

  std::vector<thrust::tuple<SimpleSplitCandidate, SimpleSplitCandidate>> expected_out{
      {{0, 2.0f, 0.0}, {1, 12.0f, 2.0}},
      {{1, 11.0f, 4.0}, {1, 10.0f, 4.0}},
      {{1, 11.0f, 4.0}, {1, 10.0f, 4.0}},
      {{1, 10.0f, 4.0}, {1, 10.0f, 4.0}},
      {{1, 10.0f, 4.0}, {1, 10.0f, 4.0}}
  };
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto [forward, backward] = out.at(i);
    auto [expected_forward, expected_backward] = expected_out.at(i);
    EXPECT_EQ(forward, expected_forward) << "Mismatch at index " << i;
    EXPECT_EQ(backward, expected_backward) << "Mismatch at index " << i;
  }
}
