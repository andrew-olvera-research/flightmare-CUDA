#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Experimental native channel stage. One lane per configured directed link.
// Submit at the previous horizon. Drain through the new horizon inclusively.
// Caller owns resident buffers and consumes the per-link event prefix each tick.
namespace flightlib_cuda::packet {
using Time = std::uint64_t;
struct Profile { unsigned ns_per_byte, propagation_ns; };
struct Attempt { Time id, ttl_ns; unsigned bytes; }; // zero bytes = idle
struct Event { Time id, arrival_ns; };
struct Link { Time now, tx_free, accepted, delivered, rejected_full; unsigned head, size; };
enum class Admission : unsigned { Idle, Accepted, Full, Deadline, ClockOverflow };
struct Result { Admission admission; unsigned delivered; };

// Low-level asynchronous stage. Profiles are immutable while work is pending.
// Provide links entries of each per-link field and links*depth events for each
// slab. State starts zeroed. All arrays must be disjoint, on the stream's
// device, and live until completion. The caller orders submissions on a stream.
void advanceLinks(unsigned links, unsigned depth, unsigned dt_ns,
                  const Profile* profiles, const Attempt* attempts,
                  Link* state, Event* queue, Event* output, Result* results,
                  cudaStream_t stream);
} // namespace flightlib_cuda::packet
