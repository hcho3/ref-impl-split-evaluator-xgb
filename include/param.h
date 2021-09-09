#ifndef PARAM_H_
#define PARAM_H_

#include <ostream>
#include <cstdint>

struct TrainingParam {
  float min_child_weight;
};

using bst_node_t = int32_t;
using bst_feature_t = uint32_t;

struct GradientPair {
  float grad_;
  float hess_;

  GradientPair &operator+=(const GradientPair& rhs) {
    grad_ += rhs.grad_;
    hess_ += rhs.hess_;
    return *this;
  }

  GradientPair operator+(const GradientPair& rhs) const {
    GradientPair g;
    g.grad_ = grad_ + rhs.grad_;
    g.hess_ = hess_ + rhs.hess_;
    return g;
  }

  GradientPair operator-(const GradientPair& rhs) const {
    return {grad_ - rhs.grad_, hess_ - rhs.hess_};
  }

  friend std::ostream& operator<<(std::ostream& os, const GradientPair& m) {
    os << m.grad_ << "/" << m.hess_;
    return os;
  }
};

struct GradStats {
  double sum_grad;
  double sum_hess;

  GradStats() : sum_grad(0.0), sum_hess(0.0) {}
  GradStats(double grad, double hess) : sum_grad(grad), sum_hess(hess) {}
  explicit GradStats(GradientPair pair) : sum_grad(pair.grad_), sum_hess(pair.hess_) {}

  GradStats operator+(const GradStats& rhs) const {
    return {sum_grad + rhs.sum_grad, sum_hess + rhs.sum_hess};
  }

  GradStats operator-(const GradStats& rhs) const {
    return {sum_grad - rhs.sum_grad, sum_hess - rhs.sum_hess};
  }

  friend std::ostream& operator<<(std::ostream& os, const GradStats& m) {
    os << m.sum_grad << "/" << m.sum_hess;
    return os;
  }
};

enum class FeatureType : uint8_t {
  kNumerical,
  kCategorical
};

enum DefaultDirection {
  kLeftDir = 0,
  kRightDir
};

#endif  // PARAM_H_
