# Native packet stage development benchmark

Run inside the CUDA development container from the repository root.

```sh
cmake -S flightlib-cuda -B /build/packet-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DFLIGHTLIB_CUDA_TEST_DEVICE=ON
cmake --build /build/packet-tests --target packet_test
/build/packet-tests/packet_test
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 /build/packet-tests/packet_test
cmake -S tools/profile/native -B /build/packet-profile -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build /build/packet-profile --target profile_driver
python3 tools/transport/bench_packet.py /build/packet-profile/profile_driver > /build/packet-profile/benchmark.jsonl
```

The driver steps the native environment for 10 ms of simulation time, then enqueues one CUDA thread per configured link on the simulation stream. The fixture assigns one directed link to each endpoint. Output is a per-link prefix of packet identities and arrival times. A caller associates link rows with receivers. Each link has an independent transmitter, a four-event FIFO, and one resident 1,200-byte attempt each tick. The ideal profile has zero airtime, the rate profile takes 0.6 ms per attempt, and the overload profile takes 60 ms. All three use zero propagation delay for timing. The reference test varies propagation, packet size, deadline and submission intervals. Profiles are synthetic.

Each run includes 128 warmup steps and 128 individually timed steps. The sweep shuffles the plant-only and channel configurations across three rounds. CUDA event time spans the entire step, including any gaps in kernel submission. The host waits between samples. Packet totals include warmup, while the CUDA API counters cover the timed loop. To compute steady-state offered attempts/s for these fixtures, divide endpoint count by mean measured step time in seconds. Delivery throughput in overload requires subtracting warmup deliveries from the cumulative count.

The producer and receiver buffers are preallocated. The time measurements exclude payload production, receiver fan-in, host export, and policy inference. This first stage exercises independent links. Shared sender airtime, broadcast and trace-calibrated impairments require separate qualification.

For measured reception replay and profile qualification, see [Wi-SUN validation](wisun_validation.md).
