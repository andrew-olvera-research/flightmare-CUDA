#include "flightlib_cuda/packet_channel.hpp"
#include <stdexcept>

namespace flightlib_cuda::packet {
static __global__ void advance(unsigned links, unsigned depth, unsigned dt_ns,
                        const Profile* profiles, const Attempt* attempts,
                        Link* state, Event* queue, Event* output, Result* results) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= links) return;
  const auto profile = profiles[i];
  const auto attempt = attempts[i];
  auto link = state[i];
  auto* pending = queue + i * depth;
  auto* delivered = output + i * depth;
  Result result{Admission::Idle, 0};
  constexpr Time last = ~Time{0};
  if (dt_ns > last - link.now) {
    results[i] = {Admission::ClockOverflow, 0};
    return;
  }
  if (attempt.bytes) {
    const Time start = link.now > link.tx_free ? link.now : link.tx_free;
    const Time airtime = Time(attempt.bytes) * profile.ns_per_byte;
    if (link.size == depth) {
      result.admission = Admission::Full;
      ++link.rejected_full;
    } else if (airtime > last - start || profile.propagation_ns > last - start - airtime) {
      result.admission = Admission::ClockOverflow;
    } else {
      const Time finish = start + airtime;
      const Time arrival = finish + profile.propagation_ns;
      if (arrival - link.now > attempt.ttl_ns) {
        result.admission = Admission::Deadline;
      } else {
        pending[(Time(link.head) + link.size) % depth] = {attempt.id, arrival};
        ++link.size;
        ++link.accepted;
        link.tx_free = finish;
        result.admission = Admission::Accepted;
      }
    }
  }
  link.now += dt_ns;
  while (link.size && pending[link.head].arrival_ns <= link.now) {
    delivered[result.delivered++] = pending[link.head];
    link.head = (link.head + 1) % depth;
    --link.size;
  }
  link.delivered += result.delivered;
  state[i] = link;
  results[i] = result;
}
void advanceLinks(unsigned links, unsigned depth, unsigned dt_ns,
                  const Profile* profiles, const Attempt* attempts, Link* state,
                  Event* queue, Event* output, Result* results, cudaStream_t stream) {
  if (!links || !depth || !profiles || !attempts || !state || !queue || !output || !results)
    throw std::invalid_argument("invalid packet channel storage");
  advance<<<1+(links-1)/128,128,0,stream>>>(links,depth,dt_ns,profiles,attempts,
                                             state,queue,output,results);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace flightlib_cuda::packet
