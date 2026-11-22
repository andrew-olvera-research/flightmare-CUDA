# Resident packet channel

Pass NetworkConfig to the Python Environment or set network_fanin, network_tick_ns and shared_profile in native RuntimeConfig. The simulator arena owns the channel state. Disabled environments allocate zero network bytes. The ordinary step, native CUDA graph and caller-owned graph run the same scheduling and delivery kernels. Use Environment.access(stream) to order writes and reads on a different stream.

```python
from flightlib_cuda import Environment, NetworkConfig

profile = NetworkConfig.from_profile("tools/transport/shared_basic_v1.json",
                                     fanin=1, offsets_ns=(0, 2_000_000))
env = Environment(2, 10, control_dt=.01, network=profile, auto_reset=False)
env.reset_state[2].fill_(5)
env.reset()
with env.access():
    ch = env.network
    ch.sources.fill_(0)                 # every receiver listens to slot zero
    ch.broadcast[:, :, 0].fill_(1)
    ch.size[:, :, 0].fill_(1000)
    ch.packet_id[0, :, 0].fill_(10)
    ch.packet_id[1, :, 0].fill_(11)
env.capture(random=False)
env.step(graph=True, random=False)
with env.access():
    snapshot = ch.inbox.cpu()           # explicit diagnostic transfer
```

Each configured offset is a nanosecond submission opportunity, increasing from zero to below the control period. Inputs are [offset, environment, sender]; one-offset configurations omit the leading dimension. A zero byte length is idle. Size, packet identity, attempt identity, absolute deadline and opaque payload handle are independent. Zero expiry disables the deadline. The current sender can retain one waiting or in-flight frame. A second offer receives Full until that slot is released. Inputs persist until changed or reset.

Subscribe by local source slot in sources[environment, receiver, edge]. Out-of-range indices disable that edge. The per-offset receiver inbox contains validity, logical ID, submitted time, arrival time, attempt ID and handle. It is overwritten on the next plant step. Read terminal packets through the pre-reset callback when using automatic reset. The separate evaluator includes admission status, collision, expiry, completion, handle and fixture label. A duplicate logical ID with a new attempt ID can represent a replay; identity decisions stay in the receiver.

The caller owns payload storage. Rejected submissions acquire no lease. Accepted submissions retain their handle through completion, even when every receiver misses. Queue receiver reads on the consuming stream before recycling underlying storage. Reset cancels outstanding offers for affected senders; the caller tracks those in its ledger. Full-group reset clears pending medium occupancy while keeping the absolute clock monotonic. Receiver reset suppresses packets submitted before its new episode.

The public step reserves all result rows at setup. A lower-level consumer with a smaller output slab can call continueSharedSchedule and drain completed rows between calls. Keep inputs/offsets immutable, preserve the cursor and use stream ordering. Zero budget or zero output rows consumes no simulation time. Reaching the end makes further calls no-ops until a new schedule starts.

Run the native consumer example with `/build/network-ci/shared_test --continuation`. The example in `flightlib-cuda/tests/shared_test.cu` steps one active plant lane once and then drains four offer segments in three calls. With one writable row it advances one segment per call. With two writable rows and a work budget of one it also advances one segment. The final drain packs two rows, including completion of a frame that crossed a segment boundary. Consume the first `written` rows before reusing the slab. Each row contains one reception record per domain and sender. The final cursor records the 1,000,000 ns horizon. The public Python payload example is executable with `python -m pytest -q flightlib-cuda/python/tests/test_network.py::test_broadcast_payload_lifetime` against the built binding.

For receiver-specific sampling, set range_edges_m and populate loss_after_rx and loss_after_miss with directed loss probabilities per receiver edge and distance band. Geometry is sampled at the end of each control step; residual loss is sampled after shared-medium delivery. Calibrate conditional receiver loss separately from contention to avoid charging the same losses twice. Configure equally sized carrier-sense groups with carrier_group_size and assign hidden_sources per subscribed edge. A transmission from a listed sender in another group erases a frame if their on-air intervals overlap. This is a binary overlap rule; power capture and waveform interaction remain outside its validation.

## Running the checks

From the repository root, run `docker compose exec -T dev python3 scripts/local_ci.py PROFILE` or `scripts/check.ps1 PROFILE`. Each run writes its report, step times and logs under `.local-ci/`.

| Profile | Evidence |
| --- | --- |
| `network-parity` | Directed FIFO and shared slot-clock comparisons with independent CPU references. Native and Python JUnit files identify the failing case. |
| `network-api` | Native timelines, output bounds, broadcast and public environment contracts. |
| `network-health` | Saturated packet accounting over 67,108,864 link intervals, including repeated allocation and release. A second check constructs, replays and destroys 24 public network graphs and asserts that live Torch allocations return to baseline each time. Free device memory is diagnostic. |
| `network-sanitize` | Compute Sanitizer checks the two native network binaries for memory errors, leaks, races, uninitialized data and synchronization errors. An unsupported device or debugger yields an explicit nonpassing report. |
| `network-performance` | Three rounds of CUDA event timings across four batch shapes, each with plant-only, shared and sampled steps. The JSON lines are in `resident-step.log`. |
| `network-fidelity` | Offline measured replay, fitted-profile error and saturated throughput. Detailed per-source metrics and thresholds are in `fidelity.json` and its accompanying reports. |

Place the six Bristol files and pinned Wi-SUN `data_5563.csv` in `/build/network-measurements` inside the container, or set `FLIGHTMARE_MEASUREMENT_CACHE` to their directory. The source validators check the file digests. A missing input, replay discrepancy, or loss of a previously qualified measurement range fails the evidence gate. Individual physical fits can remain rejected when the gate passes. Read each scope's verdict before using it. The independent ns-3 comparison is documented in `ns3_comparison.md`.

## Evidence

The Bristol holdout validator verifies the source hashes in four complete vehicle logs plus one 16 MiB roadside transmitter excerpt per session. It joins vehicle receptions by the transmitter MAC from the authors' mapping and packet identifiers. The fit uses February 11 and the untouched-day score uses February 13, 2019. A receiver's nearest TX GPS fix within 250 ms defines positional coverage. For the roadside broadcast subset, both vehicles must have bracketing fixes within 250 ms. Single-direction delivery is evaluated within distance bands. Uncertainty is estimated by resampling whole minutes. The excerpt selection is for descriptive multi-receiver replay. The CSV timestamp definition and available clock evidence do not qualify one-way deadlines. [Authors' dataset](https://doi.org/10.5523/bris.eupowp7h3jl525yxhm3521f57) · [Authors' MAC mapping](https://github.com/v2x-dev/ITS-G5_trials/blob/c6e570575c40e43c2d59d7160580abe9a020ff5a/macAddresses/macAddresses.mat) · [Held-out result](bristol_holdout.json)

```sh
python3 tools/transport/validate_bristol.py /tmp/bristol --download --replay-output /tmp/bristol-masks > /tmp/bristol.json
# Current candidate exits 1 after writing the report because its error gate failed.
/build/packet-public/shared_test --broadcast-replay < /tmp/bristol-masks
```

The GPU replay checks identities and both receiver decisions for all 138,426 measured broadcasts, using synthetic airtime. Prediction on the vehicle-to-vehicle holdout failed 4 of 12 directional distance bands with a 0.10 absolute delivery tolerance. Independent mobility and deadline qualification remains open. The score_deadlines.py tool accepts actual sends and matched receptions plus a measured relative clock-error bound. It reports optimistic and conservative deadline-delivery fractions. [ns-3 comparison](ns3_comparison.md) · [Release checklist](v040_todo.md)
## Optional receiver state and qualification outputs

With a persistent outage profile, `edge_outage_rates=True` exposes `net.outage_rates[environment, receiver, subscription, :]`. Write the entry and recovery probabilities on the environment stream. A valid pair is nonnegative with a positive sum at most one. A zero-initialized or invalid pair falls back to the configured scalar rates. Switching rates retains the current receiver state. On the first eligible arrival after an idle interval, the current pair applies over that entire elapsed interval. The optional storage costs eight bytes per subscribed direction.

For addressed single-recipient frames, `unicast_ack_backoff=True` doubles the next contention window after a collision at the base window. A success or a second consecutive failure returns it to the configured base window. `unicast_success_hold_ns` and `unicast_failed_hold_ns` reserve the sender after data airtime ends according to the contention outcome. An offer while the sender is reserved returns `Full`. Data completion and receiver arrival retain their data-end timestamps. Broadcasts and multicast frames use the base window and release at data completion. These options model sender access for a one-attempt profile. [Measured scope and GPU cost](session_service.md)

Set `NetworkConfig(outage_period_ns=..., outage_enter=..., outage_recover=...)` to enable persistent receiver loss. Enter and recover are transition probabilities per period. Their sum must be positive and at most one. An eligible reception observes the state for its arrival epoch. Good epochs use the configured residual packet-loss sampler. Bad epochs suppress all eligible receptions on that directed edge. Two 64-bit words per subscribed edge hold this state. A zero period selects the existing sampler. The period is in simulation nanoseconds. Mapping a witnessed update opportunity to a physical duration requires timestamp evidence.

Set `dynamic_subscriptions=True` if the source assigned to an inbox slot can change between steps. The kernel observes bindings at the tick boundary, clears link history on a changed binding, and suppresses completions submitted before that boundary. For timing and identity checks, the evaluator exposes `completed_id` and `completed_attempt_id` beside `completed`. Set `record_outcomes=True` to opt into one `receiver_outcomes` byte per offset, lane and subscribed source. Interpret the value with `ReceiverOutcome`. It records one final disposition for the current binding. The sender admission code distinguishes acceptance from rejection while full. On each successful lane reset, the evaluator writes `reset_cancelled`, `cancelled_id`, `cancelled_attempt_id`, `cancelled_handle` and `last_reset_ns`. These five 64-bit fields retain the most recent reset per sender until its next reset. Consume each reset record when maintaining a longer ledger. The six receiver inbox fields retain their existing layout.

For a clock-bounded measurement capture, run `python3 tools/transport/score_deadlines.py CAPTURE --deadline-ns D --clock-bound-ns E`. Include one row per eligible receiver per actual send and leave the receipt empty for misses. Optional `generated_ns` includes pre-submission age. The default generation time is submission and the default quorum is a strict majority. `--quorum K` selects a fixed count. Per-link freshness is sampled immediately before each eligible offer. It reports unknown initial age and uses the greatest generation timestamp already received, including when older packets arrive late. [Qualification results and reproduction](v041_fidelity.md)
