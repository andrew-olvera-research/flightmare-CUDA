# v0.4.0 packet channel working checklist

## Current release split and test work

**v0.4.0** delivers the native packet engine, measured replay, reproducible fidelity reports, and the #23 continuation and payload-lifetime contracts. A physics or network parity failure blocks release. A deviation from a specifically advertised measured range also requires investigation. Known prediction failures stay visible in the reports. **v0.4.1** owns improvements to prediction across dates, traffic conditions and trajectories, including target-network deadline qualification. Earlier qualification notes below record evidence gathered before this scope change.

### Ordered implementation plan

1. [x] Inventory the network CI path and establish a warm-build baseline. The two native packet CTests passed in 0.52 seconds combined; the public network suite and slot oracle passed nine cases in 1.95 seconds. These timings exclude compilation. The current verify profile labels the entire Python suite as parity. The 67-million-interval soak runs inside the default packet test.
2. [x] Give network parity its own offline CI selection and JUnit result. Keep the independent CPU FIFO reference for directed links and slot-clock reference for shared airtime. Name each failure by behavior and print seed, environment, sender or receiver, tick, and expected/actual at the first mismatch. Leave the frozen plant oracle and its hash manifest intact.
3. [x] Finish #23 with two named acceptance scenarios. Under output and work pressure compare every observable event and cursor horizon to uninterrupted execution, including zero capacity, repeat-after-completion, and one plant step with multiple channel drains. Test handle accounting with one delivered broadcast receiver, one erased or reset receiver and a separate active environment; cover rejection, expiry, cancellation and safe reuse after the consumer's stream completes. Distinguish native cursor support from the public environment's fully reserved output path. Document the supported entry point, drain counts and ownership boundaries in a runnable consumer example.
4. [ ] Make parity, API contracts, health and performance independently selectable. Split large native mains into named cases using existing fixtures and binaries. Move the long lifecycle soak behind an explicit health selection; keep a small accounting check in ordinary CI. Preserve direct dataset replay commands. Register each case once per normal CI run. Keep leak/race instrumentation as a separate release check on a supported runner.
5. [ ] Add a separate fidelity entry point for the existing measured evaluations. Emit individual results for replay mismatch, source/measurement error and physical-profile qualification. Keep Bristol and Wi-SUN negative results reproducible. Ordinary verification needs no dataset download. Fidelity uses pinned inputs from a cache; release evidence records source digest, expected scope, observed metric, tolerance and verdict. Keep ns-3 and GPU timing runs as distinct reports.
6. [ ] Verify routing and usefulness before merging. From clean and warm builds record test inventory, total time and slow cases per selection. Prove a broken arrival timestamp fails parity, a lost or prematurely recycled handle fails #23, and a changed physical-fit metric appears under fidelity. Missing data, empty selections or unsupported instrumentation must have explicit outcomes. Review redundant assertions by failure mode, retaining separate coverage when native and public entry points can break independently.

## Earlier implementation and qualification evidence

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
