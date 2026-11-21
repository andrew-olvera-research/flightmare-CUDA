#pragma once
#include "flightlib_cuda/packet_channel.hpp"
#include <stdexcept>
#include <vector>

namespace packet_fixture {
using namespace flightlib_cuda::packet;
inline void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> class Buffer {
 public:
  explicit Buffer(size_t count) : count_(count) {
    checked(cudaMalloc(&data, count * sizeof(T)));
    try { checked(cudaMemset(data, 0, count * sizeof(T))); }
    catch (...) { cudaFree(data); throw; }
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const std::vector<T>& values) {
    if (values.size() != count_) throw std::invalid_argument("packet fixture shape");
    checked(cudaMemcpy(data, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> read() const {
    std::vector<T> values(count_);
    checked(cudaMemcpy(values.data(), data, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
  T* data = nullptr;
 private:
  size_t count_;
};
struct Channel {
  unsigned links, depth, tick;
  Buffer<Profile> profiles;
  Buffer<Attempt> attempts;
  Buffer<Link> state;
  Buffer<Event> queue, output;
  Buffer<Result> results;
  Channel(unsigned n, unsigned capacity, unsigned dt)
      : links(n), depth(capacity), tick(dt), profiles(n), attempts(n), state(n),
        queue(size_t(n)*capacity), output(size_t(n)*capacity), results(n) {}
  void run(cudaStream_t stream) {
    advanceLinks(links, depth, tick, profiles.data, attempts.data, state.data,
                 queue.data, output.data, results.data, stream);
  }
};
} // namespace packet_fixture
