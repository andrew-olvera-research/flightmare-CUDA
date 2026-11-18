#pragma once
#include <cuda_runtime.h>

#include <cfloat>

#include "dynamics_launch.hpp"

namespace flightlib_cuda::v1::detail {

// Normalized substep time. The quadratic interpolates both stored RK endpoints
// and has the constant acceleration used by the translation integrator.
struct Arc {
  double start, end, bend;
  __device__ double linear() const { return (end - start) - bend; }
  __device__ double at(double t) const {
    if (t == 1) return end;
    return fma(t, fma(t, bend, linear()), start);
  }
};

// Certify a clear ball about the initial position once per step. A typical
// open-space lane stays inside it throughout every substep. Nodes outside
// the search window cannot reduce the radius of that ball.
struct Clearance {
  double travel;
  bool hit;
};
__device__ inline Clearance startClearance(StaticScene scene,
                                           const float* start, Mask& valid) {
  double radius = scene.options[0], travel = fmax(.25, radius);
  for (Index i = 0; i < scene.count;) {
    const float* box = scene.nodes + Offset{i} * 10;
    double dx = double(start[0]) - box[0], dy = double(start[1]) - box[1],
           dz = double(start[2]) - box[2];
    if (box[9] == 0) {
      double window = radius + travel;
      if (fabs(dx) <= box[3] + window && fabs(dy) <= box[4] + window &&
          fabs(dz) <= box[5] + window) {
        ++i;
        continue;
      }
    } else {
      double invNorm = rsqrt(double(box[6]) * box[6] + double(box[7]) * box[7]);
      double u = fmax(0., fabs((box[6] * dx + box[7] * dy) * invNorm) - box[3]);
      double v = fmax(0., fabs((box[6] * dy - box[7] * dx) * invNorm) - box[4]);
      double w = fmax(0., fabs(dz) - box[5]);
      double distance = sqrt(u * u + v * v + w * w);
      // Conservative subtraction permits the full solver to decide a graze.
      double scale =
        fmax(radius,
             fmax(fabs(dx) + fabs(dy) + fabs(dz),
                  fmax(double(box[3]), fmax(double(box[4]), double(box[5])))));
      double error = 128 * DBL_EPSILON * scale;
      if (distance < radius - error) return {0, true};
      travel = fmin(travel, fmax(0., distance - radius - error));
    }
    double escape = box[8];
    if (!(escape >= i + 1 && escape <= scene.count &&
          trunc(escape) == escape)) {
      valid = 0;
      break;
    }
    i = static_cast<Index>(escape);
  }
  return {travel, false};
}

__device__ inline bool insideClearance(const Arc* arc, const float* origin,
                                       double travel) {
  double extent = 0;
  for (int i = 0; i < 3; ++i) {
    double displacement =
      fmax(fabs(arc[i].start - origin[i]), fabs(arc[i].end - origin[i])) +
      .25 * fabs(arc[i].bend);
    extent += displacement * displacement;
  }
  return extent < travel * travel;
}

// Stable roots strictly inside the requested interval. Linear motion, parallel
// faces and repeated roots occur routinely in upright racing gates.
__device__ inline int quadraticCuts(double a, double b, double c, double lo,
                                    double hi, double* out) {
  if (a == 0) {
    if (b == 0) return 0;
    double t = -c / b;
    if (t > lo && t < hi) {
      out[0] = t;
      return 1;
    }
    return 0;
  }
  double product = (4 * a) * c;
  double disc = fma(b, b, -product) + fma(-4 * a, c, product);
  if (disc < 0) return 0;
  double q = -0.5 * (b + copysign(sqrt(disc), b));
  if (q == 0) return 0;  // double root at zero, already an endpoint
  int n = 0;
  double u = q / a, v = c / q;
  if (u > lo && u < hi) out[n++] = u;
  if (v > lo && v < hi && v != u) out[n++] = v;
  if (n == 2 && out[1] < out[0]) {
    double tmp = out[0];
    out[0] = out[1];
    out[1] = tmp;
  }
  return n;
}

__device__ inline double distance2(const Arc* local, const float* half,
                                   double t) {
  double result = 0;
  for (int axis = 0; axis < 3; ++axis) {
    double gap = fmax(fabs(local[axis].at(t)) - half[axis], 0.);
    result = fma(gap, gap, result);
  }
  return result;
}

__device__ inline double cubic(const double* k, double t) {
  return fma(t, fma(t, fma(t, k[3], k[2]), k[1]), k[0]);
}

// Between face crossings the squared Euclidean distance to the actual box
// is quartic. Its derivative is cubic. Partition at the cubic's turning
// points, then bracket every negative-to-positive zero (a local minimum).
__device__ inline bool intervalContact(const Arc* local, const float* half,
                                       double lo, double hi, double radius2) {
  double mid = (lo + hi) * .5, k[4] = {};
  for (int axis = 0; axis < 3; ++axis) {
    Arc p = local[axis];
    double point = p.at(mid);
    if (fabs(point) <= half[axis]) continue;
    double c = p.start - copysign(double(half[axis]), point), b = p.linear(),
           a = p.bend;
    k[0] += c * b;
    k[1] += b * b + 2 * a * c;
    k[2] += 3 * a * b;
    k[3] += 2 * a * a;
  }
  double cuts[4] = {lo};
  int count = quadraticCuts(3 * k[3], 2 * k[2], k[1], lo, hi, cuts + 1) + 1;
  cuts[count++] = hi;
  for (int i = 1; i < count; ++i) {
    double left = cuts[i - 1], right = cuts[i];
    if (distance2(local, half, right) <= radius2) return true;
    if (cubic(k, left) >= 0 || cubic(k, right) <= 0) continue;
    // Work to binary64 time resolution. Distance tests avoid polynomial
    // cancellation at the closest approach to a very thin collider.
    for (int j = 0; j < 64; ++j) {
      double t = (left + right) * .5;
      if (t == left || t == right) break;
      if (distance2(local, half, t) <= radius2) return true;
      if (cubic(k, t) < 0)
        left = t;
      else
        right = t;
    }
  }
  return false;
}

__device__ inline bool leafContact(const Arc* world, const float* box,
                                   double radius, Mask& valid) {
  Arc local[3];
  double invNorm = rsqrt(double(box[6]) * box[6] + double(box[7]) * box[7]);
  for (int axis = 0; axis < 2; ++axis) {
    double x = (axis == 0 ? box[6] : -box[7]) * invNorm,
           y = (axis == 0 ? box[7] : box[6]) * invNorm;
    local[axis] = {
      x * (world[0].start - box[0]) + y * (world[1].start - box[1]),
      x * (world[0].end - box[0]) + y * (world[1].end - box[1]),
      x * world[0].bend + y * world[1].bend};
  }
  local[2] = {world[2].start - box[2], world[2].end - box[2], world[2].bend};
  double scale = radius;
  for (int i = 0; i < 3; ++i)
    scale =
      fmax(scale, fmax(double(box[3 + i]),
                       fmax(fabs(local[i].start),
                            fmax(fabs(local[i].end), fabs(local[i].bend)))));
  // Outward binary64 roundoff allowance only; tangency is inclusive.
  double allowance = 64 * DBL_EPSILON * scale;
  if (allowance > fmax(1e-9, 1e-7 * radius)) {
    valid = 0;
    return false;
  }
  radius += allowance;
  double radius2 = radius * radius;
  if (distance2(local, box + 3, 0) <= radius2 ||
      distance2(local, box + 3, 1) <= radius2)
    return true;
  double cuts[14] = {0, 1};  // 3 axes x 2 faces x at most 2 roots
  int n = 2;
  for (int i = 0; i < 3; ++i) {
    Arc p = local[i];
    double margin = fabs(p.bend) * .25;
    double low = fmin(p.start, p.end) - margin,
           high = fmax(p.start, p.end) + margin;
    if (low > box[3 + i] + radius || high < -box[3 + i] - radius) return false;
    for (int sign = -1; sign <= 1; sign += 2) {
      double face = sign * box[3 + i];
      if (low <= face && face <= high)
        n += quadraticCuts(p.bend, p.linear(), p.start - face, 0, 1, cuts + n);
    }
  }
  for (int i = 1; i < n; ++i) {
    double t = cuts[i];
    int j = i;
    while (j && t < cuts[j - 1]) {
      cuts[j] = cuts[j - 1];
      --j;
    }
    cuts[j] = t;
  }
  for (int i = 1; i < n; ++i)
    if (cuts[i] != cuts[i - 1] &&
        intervalContact(local, box + 3, cuts[i - 1], cuts[i], radius2))
      return true;
  return false;
}

__device__ inline bool sweptOverlap(StaticScene scene, const Arc* world,
                                    Mask& valid) {
  double radius = scene.options[0], low[3], high[3];
  for (int i = 0; i < 3; ++i) {
    // Maximum departure from the chord is |bend|/4 at half time.
    double margin = radius + fabs(world[i].bend) * .25;
    low[i] = fmin(world[i].start, world[i].end) - margin;
    high[i] = fmax(world[i].start, world[i].end) + margin;
  }
  for (Index i = 0; i < scene.count;) {
    const float* box = scene.nodes + Offset{i} * 10;
    if (box[9] != 0) {
      // Reject the Z slab before transforming the trajectory into box space.
      if (low[2] <= double(box[2]) + box[5] &&
          high[2] >= double(box[2]) - box[5] &&
          leafContact(world, box, radius, valid))
        return true;
      if (!valid) return false;
    } else {
      bool overlap = true;
      for (int c = 0; c < 3; ++c)
        overlap &= low[c] <= double(box[c]) + box[c + 3] &&
                   high[c] >= double(box[c]) - box[c + 3];
      if (overlap) {
        ++i;
        continue;
      }
    }
    double escape = box[8];
    if (!(escape >= i + 1 && escape <= scene.count &&
          trunc(escape) == escape)) {
      valid = 0;
      return false;
    }
    i = static_cast<Index>(escape);
  }
  return false;
}
}  // namespace flightlib_cuda::v1::detail
