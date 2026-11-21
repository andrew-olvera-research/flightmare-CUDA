# Shared airtime and broadcast

The optional native CUDA stage assigns one cooperative block per fully connected medium. Each sender can hold one frame. A new frame draws a deterministic backoff from a fixed window, counts down idle slots and freezes while another frame occupies the channel. Senders starting in the same slot collide. Their longest frame determines the occupied interval. Recovery time is configurable. Each completed transmission exposes its packet ID, end time and compact receiver set. All-peer broadcast excludes its sender and supports receiver-specific erasures in one 64-address window. Group multicast uses an explicit 64-bit recipient mask. Consumers can test membership with receivedBy. Retries, application fan-in and receiver processing belong to the caller.

## Comparison with hardware

The reference is the upper panel of Figure 5 in the University of Cape Town 802.11g hardware study. It compares saturated unicast to a common sink at a fixed 54 Mbps with one through eight contenders. The measured means were read from the plotted curve, rounded to about 0.002 in normalized throughput. We compare successful payload bits per simulated second divided by 54 Mbps. The GPU experiment uses 32 independent domains per count, 108 ms per replicate, 1,000-byte offers refreshed every 9 µs, 16 backoff slots, 148 ns/byte, 32 µs fixed frame overhead and 72 µs recovery. Those timings approximate basic access; ACK exchanges and exponential retry windows are outside this implementation. The model also uses the same recovery setting for broadcast in the cost sweep. The unicast comparison measures aggregate throughput only. Broadcast fan-out, collision ties, airtime accounting, partial slots and domain isolation have deterministic CUDA checks. [Hardware experiment](https://pubs.cs.uct.ac.za/id/eprint/534/1/Final_Version.pdf) · [Comparison results](shared_validation_results.json)

## Reproduction

Use the development container and CMake build directories from [the packet benchmark](packet_benchmark.md). The comparison exits unsuccessfully if any simulated mean differs by over 10% from its digitized hardware mean.

```sh
cmake --build /build/packet-tests --target shared_test
ctest --test-dir /build/packet-tests -R flightlib_cuda.shared --output-on-failure
python3 tools/transport/validate_shared.py /build/packet-tests/shared_test
cmake --build /build/packet-profile --target profile_driver
python3 tools/transport/bench_packet.py /build/packet-profile/profile_driver --shared
```

The native benchmark enqueues the stage after the plant on its simulation stream, with one medium per environment. Each active sender offers one 1,000-byte frame at every 10 ms boundary. Receiver masks stay compact. Expanding or consuming per-agent observations adds workload beyond this measurement. Three shuffled rounds each use 128 warmups and 128 individually timed steps.

| Environments × senders | Plant median µs | Shared unicast median µs | Broadcast median µs | Broadcast p95 µs |
|---|---:|---:|---:|---:|
| 1 × 1 | 20.94 | 26.50 | 26.45 | 56.77 |
| 1 × 100 | 22.82 | 43.90 | 43.66 | 64.61 |
| 1 × 1,000 | 25.76 | 116.77 | 116.62 | 135.55 |
| 100 × 10 | 25.73 | 39.87 | 39.97 | 67.65 |

These RTX 5070 Ti timings are medians of round statistics. The synchronized 1,000-sender case had zero successful transmissions across 256 ticks; it exercises collision handling and is outside the measured radio regime. The 100 × 10 case delivered 143,650 unicasts or 1,292,850 broadcast receiver copies across the same number of ticks.
