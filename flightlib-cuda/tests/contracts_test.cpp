#include "flightlib_cuda/contracts.hpp"

#include <iostream>
#include <vector>

using namespace flightlib_cuda::v1;

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "Failed line " << __LINE__ << ": " #condition "\n"; return 1; \
} } while (false)

int main() {
  // Deliberately neither a warp multiple nor a single-platform environment.
  const auto batch = makeBatchShape(7, 5);
  CHECK(batch.lanes == 35);
  std::vector<Scalar> storage(kStateSize * batch.lanes, -1);
  Matrix<Scalar> state{storage.data(), {kStateSize, batch.lanes}};
  for (Index e = 0; e < 7; ++e)
    for (Index a = 0; a < 5; ++a) {
      const auto n = batch.lane(e, a);
      CHECK(batch.environment(n) == e);
      state(static_cast<Index>(StateField::QuaternionWxyz), n) = Scalar(n);
    }
  for (Index n = 0; n < 35; ++n) {
    CHECK(storage[3 * 35 + n] == Scalar(n));
    CHECK(storage[2 * 35 + n] == -1);
    CHECK(storage[4 * 35 + n] == -1);
  }
  CHECK(state.stride(0) == 35 && state.stride(1) == 1);
  CHECK(state.size() == 875);
  // Policy tensors and physics tensors deliberately have different layouts.
  std::vector<Scalar> observations(7 * 5 * 12, 0);
  TensorView<Scalar, 3> obs{observations.data(), {7, 5, 12}};
  obs(6, 4, 11) = 123;
  CHECK(observations.back() == 123);
  CHECK(obs.stride(0) == 60 && obs.stride(1) == 12 && obs.stride(2) == 1);
  // 64-bit addressing must not wrap at 4 Gi elements.
  Matrix<Scalar> large{nullptr, {25, 300000000}};
  CHECK(large.offset(24, 299999999) == 7499999999ULL);
  CHECK(checkedBytes<Scalar>(large.size()) == 30000000000ULL);
  for (const auto bad : {BatchShape{0, 1, 0}, BatchShape{1, 0, 0},
                         BatchShape{kInvalidIndex, 1, 0},
                         BatchShape{65536, 65536, 0}}) {
    bool rejected = false;
    try { (void)makeBatchShape(bad.environments, bad.slots_per_environment); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  }
  bool overflow = false;
  try { (void)checkedBytes<Scalar>(std::numeric_limits<Offset>::max()); }
  catch (const std::overflow_error&) { overflow = true; }
  CHECK(overflow);
  const Matrix<const Scalar> read_only{storage.data(), {25, 35}};
  CHECK(read_only(3, 34) == 34);
  static_assert(static_cast<Index>(StateField::AccelerationBias) + 3 == kStateSize);
  static_assert(std::is_same_v<decltype(read_only(0, 0)), const Scalar&>);
  return 0;
}
