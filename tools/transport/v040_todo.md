# v0.4.0 packet channel working checklist

Sequence and release evidence for issues #22 and #23. An unchecked item is open even when a narrower fixture passes.

1. [ ] Measurement contract. Bristol ITS-G5 road trial, high-power band, OBU 00 and 01 in both directions. Fit the February 11 morning, hold out February 13 morning in full. Count actual TX requests within joint capture coverage. Compare directional reception across distance bands. Reject a profile if absolute delivery error exceeds 0.10 per supported band with at least 200 offers in each split. Report missing GPS, ambiguous joins, scope and clock precision. Deadline qualification requires independent reception timestamps and bounded clock error.
2. [x] Resident public environment integration with optional device state, graph/ordinary steps, explicit streams and masked reset.
3. [x] Timestamped multi-offer timeline, attempt identities, deadlines, CPU oracle and partition invariance.
4. [x] One-frame sender admission, bounded output, channel-only work continuation, and caller-owned payload/reset lifetime. Configurable multi-frame sender FIFO depth remains an extension.
5. [ ] Qualify mixed offered loads, access timing and per-sender service beyond the existing aggregate comparison.
6. [ ] Sparse directional geometry and cross-group hidden-sender overlap implemented. Qualify partial interference and mobility on hardware.
7. [ ] Versioned replay and seeded burst/outage profiles with provenance and validity checks.
8. [ ] Receiver-only timestamped inbox and evaluator-only labels implemented. Deadline and group-completion scorer built. Qualify information age and deadline metrics on the target network.
9. [ ] Independent multi-receiver hardware holdout, uncertainty, packet simulator cross-check.
10. [ ] Qualified-workload GPU cost, useful-delivery scaling, lifecycle and full device sanitizer results.

Validation should exercise the native simulation loop without policy training.

### Native integration checkpoint

Resident shared channel, sparse receiver inbox and full/partial reset tests are implemented. An environment reset clears traffic and backoff while preserving the absolute simulation clock. Per-lane resets discard the lane's sender state and suppress receivers observing packets submitted before their reset. The Python environment can use native capture or ordinary stepping. Verified with 77 Python passes, one skip and five passing CTests on the RTX 5070 Ti. The initial contract permits one offer per sender at each tick boundary.

### Batched timeline checkpoint

Added fixed, arbitrary nanosecond submission offsets within a plant tick. Each offset has one offer slot per sender, a separate admission/completion result, and a reserved receiver event row per subscribed edge. The CUDA block advances the complete schedule in one launch. Frames carry logical and attempt IDs, absolute expiry and opaque payload handle. Receiver observations expose only live deliveries. The evaluator returns completion handles even for in-flight expiry. The caller retains payload storage through completion or episode cancellation. Dynamic output limits, explicit continuation and the independent shared-timeline CPU oracle remain open.

### Continuation, receiver geometry and scaling checkpoint

The native schedule can resume from a device cursor with a segment budget and reusable output slab. Tests compare limited and full processing and verify zero-work and already-completed calls. The environment path reserves enough rows for its entire configured tick. A CPU slot oracle independently checks the public GPU path. Per-edge distance bands and two conditional loss tables now sample receiver outcomes on the GPU. Positions are sampled at the plant tick endpoint. Separate backoff/interference domains, hidden senders, measured calibration and long outages are still open.

On the RTX 5070 Ti a 1 × 1,000 case generated 999 useful receiver observations per broadcast. Across three rounds the median captured plant step took 16.48 µs, shared channel with receiver materialization 32.77 µs, and shared plus sampled receiver logic 34.82 µs (zero loss probabilities). 100 × 10 took 16.45, 22.53 and 24.58 µs respectively. These are one-source-per-medium tests. Reproduce with bench_resident.py against the built Python package. Full Python suite: 80 passed, one skip; native: five passed.

### Hardware qualification checkpoint

Published [resident API and reproduction instructions](resident_channel.md), [measured profile](shared_basic_v1.json), [Bristol fit/holdout](bristol_holdout.json) and [independent packet simulation comparison](ns3_comparison.md). Matched actual offers and receptions by transmitter MAC and packet identity. Distance-only fitting failed four of twelve held-out directional bands (maximum error 12.42 percentage points). GPU replay verified both receiver decisions for 138,426 measured roadside broadcasts. The packet-ID-only preliminary analysis was superseded after discovering repeated identifiers across transmitters.

Final integrated benchmark on the RTX 5070 Ti: at 1 × 1,000, plant median 16.42 µs; sampled channel plus 999 useful inbox entries median 36.86 µs, p95 37.73 µs, total arena 908,352 bytes. In 100 × 10 with 900 useful inbox entries, sampled median 24.58 µs, p95 25.25 µs. Test geometry used one sender per medium with zero sampled loss. Hot-loop timing excludes host downloads. The final build passed 82 Python tests with one skip and five native CTests; a separate host-UBSan build passed five CTests. A 24-cycle public graph construct/replay/destroy soak returned free device bytes and Torch live allocations to baseline. Compute Sanitizer could not initialize the WDDM debugger interface. Its leak and race gates are unverified.

Release decision: open. Needed to qualify decentralized operation: specify the target radio, packet sizes/rates, contention geometry, peer count and information deadline; obtain clock-bounded send and receive measurements in that envelope. Exercise the scorer, independently held-out burst/outage and per-sender timing profiles, and the device sanitizer suite on a supported runner. Physical correctness of the target traffic and an accepted cost budget remain outstanding.
