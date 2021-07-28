#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <random>
#include <tuple>
#include <cstdint>
#include <cstddef>
#include "iterator.h"

TEST(Iterator, InputIterator) {
  constexpr std::size_t size = 50;
  std::vector<double> vec(size);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::mt19937 gen(std::random_device{}());
  std::generate_n(vec.begin(), size, [&]() { return unif(gen); });

  auto iter = InputIterator(vec.begin(), vec.end());
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_EQ(iter.Get(), vec.at(i));
    iter.Next();
  }
}

TEST(Iterator, CountingIterator) {
  constexpr int32_t init = 5;
  auto for_counting = MakeForwardCountingIterator<int32_t>(init);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(for_counting.Get(), init + static_cast<int32_t>(i));
    for_counting.Next();
  }
  auto rev_counting = MakeBackwardCountingIterator<int32_t>(init);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rev_counting.Get(), init - static_cast<int32_t>(i));
    rev_counting.Next();
  }
}

TEST(Iterator, TransformIterator) {
  constexpr int32_t init = 15;
  auto for_counting = MakeForwardCountingIterator<int32_t>(init);

  auto transform_iter = MakeTransformIterator(for_counting,
                                              [](int32_t x) { return x * 100 % 15; });
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(transform_iter.Get(), (init + static_cast<int32_t>(i)) * 100 % 15);
    transform_iter.Next();
  }
}

TEST(Iterator, ZipIterator) {
  auto for_counting = MakeForwardCountingIterator<int32_t>(0);
  auto rev_counting = MakeBackwardCountingIterator<int32_t>(100);
  auto zip_iter = MakeZipIterator(for_counting, rev_counting);
  for (int i = 0; i < 200; ++i) {
    auto [x, y] = zip_iter.Get();
    EXPECT_EQ(x, static_cast<int32_t>(i));
    EXPECT_EQ(y, static_cast<int32_t>(100 - i));
    zip_iter.Next();
  }
}

TEST(Iterator, OutputIterator) {
  constexpr std::size_t size = 50;
  std::vector<double> vec(size);

  std::vector<double> expected_vec(size);
  std::uniform_real_distribution<double> unif(0.0,1.0);
  std::mt19937 gen(std::random_device{}());
  std::generate_n(expected_vec.begin(), size, [&]() { return unif(gen); });

  auto out_iter = OutputIterator(vec.begin(), vec.end());
  for (std::size_t i = 0; i < size; ++i) {
    out_iter.Set(expected_vec.at(i));
    out_iter.Next();
  }
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_EQ(vec.at(i), expected_vec.at(i));
  }
}

TEST(Iterator, TransformOutputIterator) {
  constexpr std::size_t size = 150;
  std::vector<double> vec(size);

  std::vector<double> vec2(size);
  std::uniform_real_distribution<double> unif(0.0,1.0);
  std::mt19937 gen(std::random_device{}());
  std::generate_n(vec2.begin(), size, [&]() { return unif(gen); });

  auto out_iter = OutputIterator(vec.begin(), vec.end());
  auto transform_func = [](double x) { return 2 * x - 1.0; };
  auto transform_iter = MakeTransformOutputIterator(out_iter, transform_func);
  for (std::size_t i = 0; i < size; ++i) {
    transform_iter.Set(vec2.at(i));
    transform_iter.Next();
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

  auto for_counting = MakeForwardCountingIterator<int32_t>(0);
  auto rev_counting = MakeBackwardCountingIterator<int32_t>(10);
  auto for_iter = InputIterator(vec.begin(), vec.end());
  auto rev_iter = InputIterator(vec2.begin(), vec2.end());
  auto for_zip_iter = MakeZipIterator(for_counting, for_iter);
  auto rev_zip_iter = MakeZipIterator(rev_counting, rev_iter);
  auto for_value_iter = MakeTransformIterator(
      for_zip_iter,
      [](std::tuple<int32_t, double> x) {
        return std::make_tuple(std::get<0>(x) * 10, 50 - std::get<1>(x) * 30);
      });
  auto rev_value_iter = MakeTransformIterator(
      rev_zip_iter,
      [](std::tuple<int32_t, double> x) {
        return std::make_tuple(std::get<0>(x) * 10, 60 - std::get<1>(x) * 15);
      });
  auto zip_iter = MakeZipIterator(for_value_iter, rev_value_iter);
  for (std::size_t i = 0; i < size; ++i) {
    auto [x, y] = zip_iter.Get();
    EXPECT_EQ(std::get<0>(x), static_cast<int32_t>(i) * 10);
    EXPECT_FLOAT_EQ(std::get<1>(x), 50 - vec.at(i) * 30);
    EXPECT_EQ(std::get<0>(y), (10 - static_cast<int32_t>(i)) * 10);
    EXPECT_FLOAT_EQ(std::get<1>(y), 60 - vec2.at(i) * 15);
    zip_iter.Next();
  }
}
