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

For receiver-specific sampling, set range_edges_m and populate loss_after_rx and loss_after_miss with directed loss probabilities per receiver edge and distance band. Geometry is sampled at the end of each control step; residual loss is sampled after shared-medium delivery. Calibrate conditional receiver loss separately from contention to avoid charging the same losses twice. Configure equally sized carrier-sense groups with carrier_group_size and assign hidden_sources per subscribed edge. A transmission from a listed sender in another group erases a frame if their on-air intervals overlap. This is a binary overlap rule; power capture and waveform interaction remain outside its validation.

## Evidence

The Bristol holdout validator verifies the source hashes in four complete vehicle logs plus one 16 MiB roadside transmitter excerpt per session. It joins vehicle receptions by the transmitter MAC from the authors' mapping and packet identifiers. The fit uses February 11 and the untouched-day score uses February 13, 2019. A receiver's nearest TX GPS fix within 250 ms defines positional coverage. For the roadside broadcast subset, both vehicles must have bracketing fixes within 250 ms. Single-direction delivery is evaluated within distance bands. Uncertainty is estimated by resampling whole minutes. The excerpt selection is for descriptive multi-receiver replay. The CSV timestamp definition and available clock evidence do not qualify one-way deadlines. [Authors' dataset](https://doi.org/10.5523/bris.eupowp7h3jl525yxhm3521f57) · [Authors' MAC mapping](https://github.com/v2x-dev/ITS-G5_trials/blob/c6e570575c40e43c2d59d7160580abe9a020ff5a/macAddresses/macAddresses.mat) · [Held-out result](bristol_holdout.json)

```sh
python3 tools/transport/validate_bristol.py /tmp/bristol --download --replay-output /tmp/bristol-masks > /tmp/bristol.json
# Current candidate exits 1 after writing the report because its error gate failed.
/build/packet-public/shared_test --broadcast-replay < /tmp/bristol-masks
```

The GPU replay checks identities and both receiver decisions for all 138,426 measured broadcasts, using synthetic airtime. Prediction on the vehicle-to-vehicle holdout failed 4 of 12 directional distance bands with a 0.10 absolute delivery tolerance. Independent mobility and deadline qualification remains open. The score_deadlines.py tool accepts actual sends and matched receptions plus a measured relative clock-error bound. It reports optimistic and conservative deadline-delivery fractions. [ns-3 comparison](ns3_comparison.md) · [Release checklist](v040_todo.md)
