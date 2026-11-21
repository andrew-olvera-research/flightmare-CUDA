#pragma once
#include "flightlib_cuda/packet_channel.hpp"

namespace flightlib_cuda::packet {
// A medium is a fully connected contention domain. One row per sender.
// receivers and erased are masks relative to receiver_base within the medium.
struct Frame { Time id, receivers, erased; unsigned receiver_base, bytes; std::uint8_t all_peers = 0;
  Time attempt_id{}, handle{}, expiry_ns{}; // absolute simulation ns; zero disables expiry
};
struct SharedProfile {
  unsigned slot_ns, window_slots, ns_per_byte, preamble_ns, recovery_ns, seed;
};
struct Medium { Time now, next_slot, transmissions, collisions, busy_ns; };
enum class Phase : unsigned { Empty, Waiting, OnAir };
struct Sender {
  Frame frame;
  Time submitted, finish, sequence, completions, received_copies;
  unsigned remaining;
  Phase phase;
  bool collided;
};
struct Reception {
  Time id, arrival, receivers;
  unsigned receiver_base;
  bool completed, collided;
  // For all_peers, receivers holds exclusions in the receiver_base mask window.
  bool all_peers;
  Admission admission;
  Time submitted, attempt_id, handle;
  bool expired;
};
// Expand a completion on demand. Full-domain broadcast excludes its source.
__host__ __device__ inline bool receivedBy(const Reception& r, unsigned sender,
                                           unsigned receiver, unsigned stations) {
  if (!r.completed || r.collided || r.expired) return false;
  unsigned offset = receiver - r.receiver_base;
  bool masked = receiver >= r.receiver_base && offset < 64 && ((r.receivers >> offset) & 1);
  return r.all_peers ? receiver < stations && receiver != sender && !masked : masked;
}
// Device arrays: media[domains]; offers, senders and results[domains*stations].
// Zero-initialize persistent media/senders. One offer per sender at tick start
// (zero bytes = idle). Consume one completion per sender before the next call.
// The sender buffer holds one frame until its transmission ends. A full offer
// is rejected independently of a previous frame completing this tick.
// All arrays must be disjoint and retained on the stream's device until done.
void advanceShared(unsigned domains, unsigned stations, unsigned dt_ns, SharedProfile,
                   Medium*, const Frame*, Sender*, Reception*, cudaStream_t, const std::uint8_t* active=nullptr);
// Setup validates ascending tick offsets (first zero, last below dt). Supply the
// immutable device copy and disjoint offers/results laid out [segments,domains,stations].
void advanceSharedSchedule(unsigned domains,unsigned stations,unsigned dt,SharedProfile,
    Medium*,const Frame*,Sender*,Reception*,cudaStream_t,const std::uint8_t* active,
    unsigned segments,const unsigned* offsets);
} // namespace flightlib_cuda::packet
