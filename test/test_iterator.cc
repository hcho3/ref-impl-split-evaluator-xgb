#include <gtest/gtest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/tuple.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cstddef>

TEST(Iterator, CountingIterator) {
  constexpr int32_t init = 5;
  {
    auto for_counting = thrust::make_counting_iterator<int32_t>(init);
    for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(*for_counting, init + static_cast<int32_t>(i));
      ++for_counting;
    }
  }
  {
    auto rev_counting = thrust::make_reverse_iterator(
        thrust::make_counting_iterator<int32_t>(0) + init);
    for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(*rev_counting, init - 1 - static_cast<int32_t>(i));
      ++rev_counting;
    }
  }
  {
    auto rev_counting = thrust::make_reverse_iterator(
        thrust::make_counting_iterator<int32_t>(init));
    for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(*rev_counting, init - 1 - static_cast<int32_t>(i));
      ++rev_counting;
    }
  }
}

TEST(Iterator, TransformIterator) {
  constexpr int32_t init = 15;
  auto for_counting = thrust::make_counting_iterator<int32_t>(init);
  auto transform_iter = thrust::make_transform_iterator(for_counting,
                                                        [](int32_t x) { return x * 100 % 15; });
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(*transform_iter, (init + static_cast<int32_t>(i)) * 100 % 15);
    ++transform_iter;
  }
}

TEST(Iterator, ZipIterator) {
  auto for_counting = thrust::make_counting_iterator<int32_t>(0);
  auto rev_counting = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<int32_t>(0) + 101);
  auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(for_counting, rev_counting));
  for (int i = 0; i < 200; ++i) {
    const auto& e = *zip_iter;
    const auto& x = thrust::get<0>(e);
    const auto& y = thrust::get<1>(e);
    EXPECT_EQ(x, static_cast<int32_t>(i));
    EXPECT_EQ(y, static_cast<int32_t>(100 - i));
    ++zip_iter;
  }
}

TEST(Iterator, TransformOutputIterator) {
  constexpr std::size_t size = 150;
  std::vector<double> vec(size);

  std::vector<double> vec2(size);
  std::uniform_real_distribution<double> unif(0.0,1.0);
  std::mt19937 gen(std::random_device{}());
  std::generate_n(vec2.begin(), size, [&]() { return unif(gen); });

  auto transform_func = [](double x) { return 2 * x - 1.0; };
  auto transform_iter = thrust::make_transform_output_iterator(vec.begin(), transform_func);
  for (std::size_t i = 0; i < size; ++i) {
    *transform_iter = vec2.at(i);
    ++transform_iter;
  }
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(vec.at(i), transform_func(vec2.at(i)));
  }
}

TEST(Iterator, Combination) {
  constexpr std::size_t size = 100;
  std::vector<double> vec(size);
  std::vector<double> vec2(size);

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::mt19937 gen(std::random_device{}());
  std::generate_n(vec.begin(), size, [&]() { return unif(gen); });
  std::generate_n(vec2.begin(), size, [&]() { return unif(gen); });

  auto for_counting = thrust::make_counting_iterator<int32_t>(0);
  auto rev_counting = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<int32_t>(0) + 11);
  auto for_zip_iter = thrust::make_zip_iterator(thrust::make_tuple(for_counting, vec.begin()));
  auto rev_zip_iter = thrust::make_zip_iterator(thrust::make_tuple(rev_counting, vec2.begin()));
  auto for_value_iter = thrust::make_transform_iterator(
      for_zip_iter,
      [](thrust::tuple<int32_t, double> e) {
        return thrust::make_tuple(thrust::get<0>(e) * 10, 50 - thrust::get<1>(e) * 30);
      });
  auto rev_value_iter = thrust::make_transform_iterator(
      rev_zip_iter,
      [](thrust::tuple<int32_t, double> e) {
        return thrust::make_tuple(thrust::get<0>(e) * 10, 60 - thrust::get<1>(e) * 15);
      });
  auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(for_value_iter, rev_value_iter));
  for (std::size_t i = 0; i < size; ++i) {
    const auto& x = thrust::get<0>(*zip_iter);
    const auto& y = thrust::get<1>(*zip_iter);
    double x0 = thrust::get<0>(x);
    double x1 = thrust::get<1>(x);
    double y0 = thrust::get<0>(y);
    double y1 = thrust::get<1>(y);
    EXPECT_EQ(x0, static_cast<int32_t>(i) * 10);
    EXPECT_FLOAT_EQ(x1, 50 - vec.at(i) * 30);
    EXPECT_EQ(y0, (10 - static_cast<int32_t>(i)) * 10);
    EXPECT_FLOAT_EQ(y1, 60 - vec2.at(i) * 15);
    ++zip_iter;
  }
}
