#pragma once
#include "packet_fixture.cuh"
#include "flightlib_cuda/shared_channel.hpp"

namespace packet_fixture {
struct SharedChannel {
  unsigned domains, stations, tick;
  SharedProfile profile;
  Buffer<Medium> media;
  Buffer<Frame> offers;
  Buffer<Sender> senders;
  Buffer<Reception> results;

  SharedChannel(unsigned e, unsigned n, unsigned dt, SharedProfile p)
      : domains(e), stations(n), tick(dt), profile(p), media(e), offers(size_t(e)*n),
        senders(size_t(e)*n), results(size_t(e)*n) {}
  void run(cudaStream_t stream) {
    advanceShared(domains,stations,tick,profile,media.data,offers.data,senders.data,
                  results.data,stream);
  }
};
} // namespace packet_fixture
