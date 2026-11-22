# v0.4.1 receiver and contention investigation

The first development checkpoint exercises the public GPU environment with measured reception outcomes withheld. The `network-fidelity` profile writes `resident-wisun.json` next to the existing replay and physical-fit reports. The `network-performance` profile writes graph step timings to `resident-step.log` and per-sender service measurements to `contention.json`.

## Receiver baseline

The pinned Wi-SUN node sent three scheduled 32-byte FSK copies per period. Use the existing witnessed-period cohort. Fit the two resident receiver probabilities by counting successive copy outcomes, including transitions across consecutive witnessed packet counters. Split at day boundaries and missing counters. Each uninterrupted sequence gets an independent two-station environment with a freshly reset receiver. Only sequence lengths, fixed profile probabilities and a seed enter simulation. Compare the receiver inbox with measured mask frequencies, adjacent complete misses and loss-run lengths. Runs are censored at sequence boundaries.

The synthetic schedule has three offers in one millisecond with ample airtime. The evaluator checks admission, completion, collision and expiry. This isolates receiver sampling. There is no inference of physical delay from this schedule. The public receiver initializes with the after-reception probability. The experiment uses seeds 22, 23 and 24.

| Calibration and evaluation | Measured complete misses | Simulated complete misses | Measured adjacent complete-miss pairs | Simulated pairs |
| --- | ---: | ---: | ---: | ---: |
| August 1–14 to September 1–14 | 73 | 18–35 | 19 | 2–4 |
| August 1–14 to October 1–14 | 2,032 | 22–31 | 1,255 | 0–5 |
| October 1–7 to October 8–14, diagnostic | 1,458 | 354–388 | 905 | 108–132 |
| October 1–14 fit and scored on the same sample, capacity diagnostic | 2,032 | 1,534–1,581 | 1,255 | 485–534 |

In the same-sample diagnostic the receiver produced 82.91–83.07% mean delivery against 82.77% measured. Its longest complete-miss run was 6–7 periods against 54 measured. A recent stationary calibration reduces the marginal error, while the last-outcome memory still understates persistent losses on these data. The August fit uses loss probabilities 0.00299 following reception and 0.5 following loss. The physical cause of the change between sessions remains undetermined.

The evidence run enforces the prior September distribution tolerances and reports the complete-miss count criterion separately. October is development data. Reserve the available November dates for confirmation after freezing a candidate, thresholds and coverage rules. The capture has a single receiver and supports opportunity-indexed comparisons. Qualification for elapsed outage duration, shared receiver outages and deadlines requires additional measurements.

## Controlled contention check

The public CUDA diagnostic offers 1,000 bytes every 100 microseconds from each source to a common sink. The receiver materializes all sources. Following 100 ms of warm-up, each run measures one simulated second. For each sender the report records admissions, rejections while full, completions, collisions, successful deliveries, and the median and 95th percentile wait from submission until airtime starts. It enforces conservation across the measurement window including a possible in-flight frame at each boundary. Three seeds give mean aggregate goodput of 24.74, 24.12 and 19.21 Mbps for 1, 4 and 8 sources.

Repeat the pinned ns-3.44 experiment from `ns3_comparison.md` with the same packet interval and size. Set `--ns3::WifiMac::FrameRetryLimit=1 --ns3::WifiMacQueue::MaxSize=1p`, use three trials and set `nMinStas` and `nMaxStas` equal for each count. The ns-3 example otherwise configures a retry limit of 65,535 and a very large queue. Observed mean throughputs were 24.14 and 25.15 Mbps for 1 and 4 sources. At 8 sources the reference aborted its all-stations-traffic check in trial one, including with a five-second measurement window. That row remains unqualified. ns-3 also staggers application startup, models MAC acknowledgements and derives throughput windows from receptions. The CUDA sender admits one outstanding frame and uses fixed recovery overhead. This comparison has no hardware accuracy verdict.

## Step cost

On the RTX 5070 Ti, median sampled step times across three rounds were 36.86, 49.18 and 63.30 microseconds for 1, 4 and 8 simultaneous senders in one 1,000-agent environment. For 100 environments of ten agents the times were 24.58, 26.62 and 30.72 microseconds. Every agent subscribed to every active source, with a 1,000-byte broadcast per sender per plant tick and zero sampled loss. These timings include the increased inbox fan-in. The contention diagnostic above exercises a different, more heavily loaded schedule.

## Remaining qualification

- Compare an optional persistent receiver state against this baseline while holding the input schedule and calibration split fixed. Measure improvement on loss runs as well as average reception. Keep the episode-reset, source-change and stream-ordering contracts.
- Reconcile eight-sender reference coverage and obtain comparable per-sender timing evidence before changing backoff or retries.
- Validate mobile delivery and joint receiver outcomes on an independent session. Establish a target radio, offered traffic, message deadline and eligible peers. Use actual sends and receiver timestamps with a measured clock-error bound for group deadline qualification.
