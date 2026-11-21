#include "flightlib_cuda/shared_channel.hpp"
#include <stdexcept>

namespace flightlib_cuda::packet {
namespace {
__device__ unsigned backoff(Time row, Time sequence, SharedProfile profile) {
  Time x = row + (Time(profile.seed) << 32) + sequence * 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return (x ^ (x >> 31)) % profile.window_slots;
}

__global__ void step(unsigned n, unsigned dt, SharedProfile p, Medium* media,
                     const Frame* offers, Sender* senders, Reception* results, const std::uint8_t* active) {
  __shared__ unsigned minimum, contenders, longest;
  __shared__ Medium m;
  if (!threadIdx.x) m = media[blockIdx.x];
  __syncthreads();
  const size_t base = size_t(blockIdx.x)*n;
  // Leave enough headroom to represent even a maximum-length transmission.
  const Time reserve = Time(UINT32_MAX)*p.ns_per_byte + p.preamble_ns + p.recovery_ns;
  if (~Time{0}-m.now < reserve || ~Time{0}-m.now-reserve < dt + Time(p.slot_ns)) {
    for (size_t j=threadIdx.x; j<n; j+=blockDim.x)
      results[base+j] = {0,0,0,0,false,false,false,Admission::ClockOverflow};
    return;
  }
  const Time end = m.now + dt;
  for (size_t j=threadIdx.x; j<n; j+=blockDim.x) {
    auto& s = senders[base+j];
    auto offer = offers[base+j];
    auto& r = results[base+j];
    r = {};
    if (active && !active[base+j]) { s = {}; continue; }
    if (!offer.bytes) continue;
    if (s.phase != Phase::Empty) { r.admission = Admission::Full; continue; }
    s.submitted = m.now;
    s.frame = offer;
    s.remaining = backoff(base+j, ++s.sequence, p);
    s.phase = Phase::Waiting;
    r.admission = Admission::Accepted;
  }
  __syncthreads();
  while (m.next_slot < end) {
    if (!threadIdx.x) { minimum = UINT32_MAX; contenders = 0; longest = 0; }
    __syncthreads();
    for (size_t j=threadIdx.x; j<n; j+=blockDim.x) {
      const auto& s = senders[base+j];
      if (s.phase == Phase::Waiting) atomicMin(&minimum, s.remaining);
    }
    __syncthreads();
    if (minimum == UINT32_MAX) {
      if (!threadIdx.x) m.next_slot = end;
      __syncthreads();
      break;
    }
    const Time slots = (end-m.next_slot+p.slot_ns-1)/p.slot_ns;
    const unsigned elapsed = slots < minimum ? unsigned(slots) : minimum;
    const bool transmit = minimum < slots;
    const Time start = m.next_slot + Time(elapsed)*p.slot_ns;
    for (size_t j=threadIdx.x; j<n; j+=blockDim.x) {
      auto& s = senders[base+j];
      if (s.phase != Phase::Waiting) continue;
      s.remaining -= elapsed;
      if (transmit && !s.remaining) {
        atomicAdd(&contenders, 1);
        atomicMax(&longest, s.frame.bytes);
      }
    }
    __syncthreads();
    if (transmit) {
      for (size_t j=threadIdx.x; j<n; j+=blockDim.x) {
        auto& s = senders[base+j];
        if (s.phase == Phase::Waiting && !s.remaining) {
          s.phase = Phase::OnAir;
          s.collided = contenders > 1;
          s.finish = start + p.preamble_ns + Time(s.frame.bytes)*p.ns_per_byte;
        }
      }
    }
    __syncthreads();
    if (!threadIdx.x) {
      const Time occupied = transmit ? Time(longest)*p.ns_per_byte+p.preamble_ns : 0;
      m.next_slot = start + occupied + (transmit ? p.recovery_ns : 0);
      m.transmissions += contenders;
      m.collisions += contenders > 1;
      m.busy_ns += occupied;
    }
    __syncthreads();
  }
  for (size_t j=threadIdx.x; j<n; j+=blockDim.x) {
    auto& s = senders[base+j];
    if (s.phase == Phase::OnAir && s.finish <= end) {
      auto& r = results[base+j];
      r.completed = true;
      r.collided = s.collided;
      r.id = s.frame.id;
      r.arrival = s.finish;
      r.receiver_base = s.frame.receiver_base;
      r.all_peers = s.frame.all_peers;
      r.receivers = s.frame.receivers & ~s.frame.erased;
      if (s.collided) r.receivers = 0;
      if (r.all_peers) r.receivers = s.frame.erased;
      ++s.completions;
      if (!s.collided && r.all_peers) {
        unsigned valid = r.receiver_base < n ? n-r.receiver_base : 0;
        Time exclusions = valid >= 64 ? r.receivers : r.receivers & ((Time{1}<<valid)-1);
        if (j >= r.receiver_base && j-r.receiver_base < 64)
          exclusions &= ~(Time{1}<<(j-r.receiver_base));
        s.received_copies += n-1-__popcll(exclusions);
      } else if (!s.collided) {
        s.received_copies += __popcll(r.receivers);
      }
      s.phase = Phase::Empty;
    }
  }
  if (!threadIdx.x) { m.now = end; media[blockIdx.x] = m; }
}
}
void advanceShared(unsigned domains, unsigned stations, unsigned dt, SharedProfile p,
                   Medium* m, const Frame* f, Sender* s, Reception* r, cudaStream_t stream, const std::uint8_t* active) {
  if (!domains || domains > INT32_MAX || !stations || !dt || !p.window_slots ||
      !p.slot_ns || !p.ns_per_byte || !m || !f || !s || !r)
    throw std::invalid_argument("invalid shared channel configuration");
  step<<<domains,128,0,stream>>>(stations,dt,p,m,f,s,r,active);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace flightlib_cuda::packet
