# v0.4.0 packet channel working checklist

Sequence and release evidence for issues #22 and #23. An unchecked item is open even when a narrower fixture passes.

1. [ ] Measurement contract. Bristol ITS-G5 road trial, high-power band, OBU 00 and 01 in both directions. Fit the February 11 morning, hold out February 13 morning in full. Count actual TX requests within joint capture coverage. Compare directional reception across distance bands. Reject a profile if absolute delivery error exceeds 0.10 per supported band with at least 200 offers in each split. Report missing GPS, ambiguous joins, scope and clock precision. Deadline qualification requires independent reception timestamps and bounded clock error.
2. [ ] Resident public environment integration with optional device state, graph/ordinary steps, explicit streams and masked reset.
3. [ ] Timestamped multi-offer timeline, attempt identities, deadlines, CPU oracle and partition invariance.
4. [ ] Bounded queues, output/work continuation, payload and reset lifetime.
5. [ ] Qualify mixed offered loads, access timing and per-sender service beyond the existing aggregate comparison.
6. [ ] Receiver-specific geometry, asymmetric reachability, hidden senders and correlated interference.
7. [ ] Versioned replay and seeded burst/outage profiles with provenance and validity checks.
8. [ ] Receiver-only event tensors, age, deadlines, group completion and evaluator-only diagnostic fixtures.
9. [ ] Independent multi-receiver hardware holdout, uncertainty, packet simulator cross-check.
10. [ ] Qualified-workload GPU cost, useful-delivery scaling, lifecycle and full device sanitizer results.

Validation should exercise the native simulation loop without policy training.

### Native integration checkpoint

Resident shared channel, sparse receiver inbox and full/partial reset tests are implemented. An environment reset clears traffic and backoff while preserving the absolute simulation clock. Per-lane resets discard the lane's sender state and suppress receivers observing packets submitted before their reset. The Python environment can use native capture or ordinary stepping. Verified with 77 Python passes, one skip and five passing CTests on the RTX 5070 Ti. The initial contract permits one offer per sender at each tick boundary.

### Batched timeline checkpoint

Added fixed, arbitrary nanosecond submission offsets within a plant tick. Each offset has one offer slot per sender, a separate admission/completion result, and a reserved receiver event row per subscribed edge. The CUDA block advances the complete schedule in one launch. Frames carry logical and attempt IDs, absolute expiry and opaque payload handle. Receiver observations expose only live deliveries. The evaluator returns completion handles even for in-flight expiry. The caller retains payload storage through completion or episode cancellation. Dynamic output limits, explicit continuation and the independent shared-timeline CPU oracle remain open.
