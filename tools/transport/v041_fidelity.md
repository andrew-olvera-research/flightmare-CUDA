# v0.4.1 receiver and contention investigation

The follow-up tests conditional loss on four additional physical links and qualifies acknowledged unicast against a fresh ns-3 run. [Session loss, sender service and GPU measurements](session_service.md)

## Issue 24 qualification results, November 22, 2026

The public environment takes traffic and topology as inputs. The evaluators score the resulting inbox against measured receivers. A model can pass software parity and still miss the measured distribution. The decisions below apply to the named traffic, dataset and observable.

| Profile or claim | Decision | Evidence |
| --- | --- | --- |
| Measured saturated throughput, 1–8 senders | Qualified for aggregate throughput on the Figure 5 workload | The existing hardware comparison remained within its 10% tolerance. |
| Wi-SUN stationary triplets, previous sampler | Qualified for the September profile, rejected for October transfer | The measured-network suite retained both outcomes. |
| Persistent Wi-SUN receiver, October 1–14 calibration | Qualified on the calibration sample only | Observed 2,032 completely missed updates and 1,255 adjacent pairs of misses. The previous sampler produced 1,534–1,581 misses, 485–534 pairs and a p95 run of 3. Persistent state produced 1,990–2,007 misses, 1,217–1,273 pairs and p95 of 7–8 against 8 observed. |
| Same persistent algorithm, October 1–7 fit to October 8–14 | Rejected | It produced 506–566 complete misses out of 1,458 observed. Mean delivery missed by 16.7–17.4 percentage points. |
| Frozen persistent candidate, November 1–10 and supplemental October 15–31 | Insufficient evidence in each window | Four and six days respectively met the 720-opportunity daily minimum. Seven were required per window. Descriptively, November had 881 misses in 4,457 opportunities against 413–503 predicted. Late October had 231 in 6,321 against 639–747 predicted. Both windows are now used. |
| Bristol vehicle-to-vehicle delivery by direction and distance | Qualified in 8 of 12 bands, rejected in 4 | The held-out directional tolerance was 10 percentage points per band. |
| Bristol Hawthorns broadcast, February 11 to February 13 morning excerpts | Qualified for aggregate receiver counts | Correcting the transmitter identity supersedes the earlier rejection. The high-power channel changed from 5900 to 5200 MHz. Neither vehicle received 94.87% of 71,952 held-out offers. The three GPU seeds predicted 93.37–93.60%. The second vehicle recorded zero receptions in this excerpt. |
| Bristol Hawthorns broadcast, February 13 morning to afternoon on 5200 MHz | Qualified for aggregate receiver counts | All 566,594 afternoon offers with paired geometry were supported by the morning calibration. Neither received 92.284% and both received 0.338% (1,916 offers). GPU predictions across three seeds were 92.024–92.102% and 0.261–0.263%, respectively. The largest total variation distance over the four outcomes was 0.00430. |
| Time until first, quorum or all receivers have a fresh update | Insufficient physical evidence | The scorer handles generation time, actual submissions, missing recipients, late or reordered deliveries and bounded clock error. The available mobile capture lacks a verified relative clock-error bound. Wi-SUN runs are in missed opportunities. |

### Loaded service and runtime cost

All eight ns-3 stations now associate and contribute in each of three trials. Active probing resolved the prior coverage failure. Both simulations offered synchronized 1,000-byte packets every 100 microseconds per sender. The table gives mean reported aggregate useful throughput and the median of per-sender/trial p95 submission-to-airtime waits. The CUDA profile admits one outstanding frame per sender. ns-3 uses a one-packet MAC queue and a retry limit of one. Queue occupancy while transmitting and acknowledgement handling can differ between those implementations.

| Senders | CUDA / ns-3 useful Mbps | CUDA / ns-3 p95 wait (µs) |
| ---: | ---: | ---: |
| 1 | 24.74 / 24.12 | 177 / 124 |
| 4 | 24.12 / 25.04 | 1,635 / 1,735 |
| 8 | 19.21 / 22.45 | 2,656 / 3,358 |

At eight senders, the GPU profile delivers 14.4% fewer useful bytes per second and its tail wait is 20.9% shorter. The public benchmark also exercises periodic, mixed-size bursty and asymmetric loads and checks offer/completion conservation per sender. Hardware queue timing and retry ownership remain open for this profile. [Per-station ns-3 reference across three trials](ns3_service_reference.json)

On an RTX 5070 Ti with one 1,000-agent environment, eight 1,000-byte broadcasters and eight subscriptions per receiver, the following values are medians across three rounds of 128 timed CUDA graph replays. Sampled-loss and persistent modes have 10% marginal loss. Arena bytes include plant, inbox and network storage. The full sweep includes 100 environments of ten agents and single-source fan-in. No step-time or memory budget was specified.

| Profile | Median step (µs) | p95 step (µs) | Arena bytes | Useful receiver deliveries per wall-clock second (millions) |
| --- | ---: | ---: | ---: | ---: |
| Plant only | 16.46 | 24.64 | 559,080 | — |
| Shared airtime | 62.21 | 69.66 | 1,263,680 | 84.1 |
| Sampled, zero residual loss | 62.50 | 69.63 | 1,391,872 | 79.4 |
| Independent 10% loss | 64.10 | 69.73 | 1,391,872 | 72.0 |
| Persistent 10% loss | 61.71 | 69.73 | 1,519,872 | 74.1 |

The pre-change zero-residual sampler measured 63.02 µs median and 69.50 µs p95 across five rounds on this shape. In the preceding three-round sweep, independent and persistent loss measured 63.49 and 64.86 µs median, respectively. Their median ordering changed between runs. The memory increment is reproducible at 128,000 bytes across 8,000 edges.

The 67,108,864-link-interval accounting soak passed. The 24-cycle public create/step/destroy run ended with the same free device memory and zero live Torch bytes. All eleven host UBSan native tests passed. Device leak and race instrumentation remains unverified because Compute Sanitizer could not initialize the WDDM debugger on this runner.

### Reproduce

Run from the repository root with the development container running. Each profile writes a timestamped directory under `.local-ci/`. The fidelity lane reports rejected and underpowered models as scientific results. For a release job, pass `--require-qualified` to the individual outage or broadcast scorer to enforce its verdict.

```sh
for profile in network-parity network-api network-fidelity network-performance network-health; do
  docker compose exec -T dev python3 scripts/local_ci.py "$profile" || exit 1
done
docker compose exec -T dev python3 -m unittest discover -s scripts/tests -p test_deadline_scores.py
# On a runner with a supported CUDA debugger
docker compose exec -T dev python3 scripts/local_ci.py network-sanitize
```

For the service reference, in a disposable checkout of ns-3 revision `43dce6710b8df69685e3479c1d33a571dda714ea`, apply `tools/transport/ns3-service.patch` with `git apply` and build the wifi-bianchi example. The patch synchronizes application startup and traces application offers and PHY starts by packet identity. Run the following in a fresh results directory with the built executable on PATH. The scorer checks all 10,000 offers per sender per trial and matches every counted PHY start in the interval from 10 to 11 seconds.

```sh
for n in 1 4 8; do
  ns3.44-wifi-bianchi --standard=11g --frequency=2.4 --phyMode=ErpOfdmRate54Mbps --infra=1 \
    --nMinStas=$n --nMaxStas=$n --trials=3 --duration=1 --pktInterval=100 --pktSize=1000 \
    --plotBianchiModel=0 --validate=0 --ns3::WifiMac::FrameRetryLimit=1 \
    --ns3::WifiMacQueue::MaxSize=1p --ns3::StaWifiMac::ActiveProbing=true > service-$n.log || exit 1
done
python3 "$FLIGHTMARE_ROOT/tools/transport/score_ns3_service.py" . --output reference.json
```

## Original development record

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

## Proposed v0.4.1 release requirements

The public contract is packet-level communication under a declared workload. Given generated messages, senders, eligible receivers and a topology over time, the runtime produces receiver-visible messages and arrival times. A researcher should be able to estimate which peers received fresh information before their deadline, how often service stopped, and the GPU cost of that estimate. Each versioned profile must state its traffic types, packet sizes and rates, peer count, queue and retry policy, mobility conditions, clock evidence and qualification range. A successful run outside that range is an exploratory result.

| Order | Required behavior | Evidence for the release decision |
| --- | --- | --- |
| 1. Account for every offer | Distinguish creation, admission, waiting, transmission, receipt and expiry. Preserve logical identity across attempts. Record why an offer was rejected and why an eligible peer missed. Keep evaluator details out of receiver observations. | An independent small CPU fixture checks chronology and conservation exactly across tick boundaries, retries if enabled, output continuation and partial resets. Account once per offer at admission, once per attempt on air and once per eligible receiver. |
| 2. Represent persistent loss | Compare the existing last-outcome sampler with an optional persistent good/degraded state on each subscribed direction. Test behavior across idle intervals, source changes and episode boundaries. Measure how often several receivers miss the same broadcast before adding shared loss state. | Freeze the candidate using development sessions. Score delivery, complete misses, consecutive misses and the loss-run distribution on reserved data. Use opportunity counts for Wi-SUN. Elapsed outage duration requires captures with timing coverage. The observed maximum run is a diagnostic; gate on repeatable distributional measures. |
| 3. Model loaded service | Define profile-level capacity in bytes/s, message serialization, finite buffering and behavior on overflow and expiry. Preserve the single-outstanding-frame profile. Add bounded FIFO buffering or retry behavior for profiles whose queueing measurements require it. Exercise idle, bursty, periodic, asymmetric and saturated offers with mixed sizes. | Compare per-sender admissions, useful bytes/s, wait-to-airtime and tail arrival delay across a load sweep. Include at least one overloaded sender alongside a lightly loaded peer. Match queue and retry semantics to an independent reference; resolve the eight-source ns-3 coverage failure or use an independent measured equivalent. |
| 4. Model who receives what and when | Support sparse directed links and changing subscriptions. Keep the cost of a single broadcast transmission distinct from its receiver work. Support a declared base delay and measured delay variation when the sender and receiver timestamps show a component beyond airtime. Specify ordering, duplication, late arrival and cancellation semantics per profile. Evaluate multi-hop traffic by composing links and per-hop queues when that workload is claimed. | Withhold receiver outcomes from simulation. Compare directional reception on a held-out mobile session, the distribution of receivers reached per broadcast, and simultaneous misses. Test stale events after topology changes and resets. |
| 5. Qualify application-visible timing | Measure receiver delay, the age of the freshest received update, and the probability that one, a quorum or all eligible peers receive a message before a declared deadline. Separate generation time, submission time and arrival time where queueing precedes submission. | Use actual offers, the eligible receiver set at send time and bounded relative clock error. Include missing receivers in the denominator and report clock-uncertainty bounds. Score a fresh capture for the advertised traffic envelope. ns-3 is a cross-check on mechanisms; hardware anchors physical accuracy. |
| 6. Bound runtime cost | Keep optional state resident and sized by active senders, queue capacity, domains and subscribed edges. Include offset-by-edge inbox storage in memory budgets. Preserve graph capture, explicit streams, deterministic seeded replay, tiling, and bounded continuation. | Vary environments, agents per medium, simultaneously active senders and receiver fan-in independently. Report median/p95 step cost, arena bytes, useful deliveries/s and drop accounting at sparse and saturated loads. Compare disabled, current and candidate profiles on identical offers and hardware. Exercise a high-parallelism soak and run device leak/race instrumentation on a supported runner. |

### Proposed acceptance thresholds

Retain the existing source-specific limits. Examples are 5 percentage points in mean Wi-SUN delivery, 3 points in complete-triplet loss, a factor-of-two count check where at least 20 complete misses are observed, and 10% relative error for the measured saturated-throughput profile. A pass on the mean alone cannot qualify persistence. Pre-register additional thresholds on development data before opening the confirmation capture.

For a first target profile, propose at most 5 percentage points of error on delivery by deadline, both per peer and for the required group, and at most 20% relative error on supported p95 delay and p95 loss-run length. Check across sender load and direction; report the sample size, uncertainty from whole sessions or time blocks, and between-seed variation. Require adequate tail events and timing resolution; an underpowered or clock-ambiguous metric stays unqualified. Profile owners still need to set the actual deadline and tolerated error for their workload.

Benchmark the current implementation before adding each stateful feature. A proposed regression budget is 5% on the existing enabled profile and 10% incremental p95 step cost for the candidate versus the current sampled profile at the same traffic and fan-in. Check run-to-run variability first and report absolute microseconds as well as percentages. The downstream total step-time and device-memory ceilings are separate inputs to acceptance.

Retain the fast parity and API checks in routine CI. Run measured holdouts, performance sweeps and sanitizer checks in their dedicated lanes. Publish a per-profile table of qualified, rejected and insufficient-evidence results. The generic runtime can be public while downstream workload definitions and private captures stay with their owners.


## Frozen persistent-receiver confirmation

Candidate calibration uses October 1–14 exclusively. A completely missed witnessed triplet labels a degraded period. Estimate entry and recovery probabilities from adjacent counters in the same day, and fit residual loss on uninterrupted good periods. The optional GPU sampler uses a stationary initial state and advances a two-state Markov chain lazily over fixed simulation periods. Within an epoch, all eligible frames on an edge share the outage. Across idle epochs the transition uses the corresponding power of the transition matrix. Each edge owns two uint64 words and uses a seeded draw independent of residual packet loss. Receiver reset, sender reset and observed changes in source subscription invalidate prior state. The one-millisecond period in this fixture represents one opportunity and has no measured physical duration.

Before reading November outcomes, select November 1–10, 2019, include every day with at least 720 witnessed periods, and require at least seven eligible days. Use seeds 22, 23 and 24 and the identical public-step schedule for both models. Require every seed to meet the established distribution and complete-miss count limits, absolute repeat-miss probability error at most 0.10 (at least 100 pairs starting with a miss in each sample), and relative p95 run-length error at most 0.20 (at least 200 uncensored loss runs in each sample). The tail excludes runs touching either end of an uninterrupted sequence. An unsupported metric stays unqualified. Report whole-day bootstrap intervals and all seed outcomes. Report GPU costs separately; there is no user-specified time or memory budget.

In October's same-data capacity check the observed p95 was 8 missed periods. The baseline produced 3 at every seed and the persistent sampler produced 7–8. Observed adjacent miss pairs numbered 1,255; the persistent sampler produced 1,217–1,273. Calibration on October 1–7 still failed transfer to October 8–14. November is a test of transfer. A failed confirmation must remain in the evidence history.


November yielded four eligible days (1, 6, 7 and 8); the frozen minimum was seven. Record insufficient evidence. For a supplemental chronological holdout, use all previously unused October 15–31 dates, with the identical October 1–14 calibration, model, seeds, daily eligibility, minimum day count and acceptance thresholds. These dates precede the November coverage audit but follow all calibration dates. Preserve both verdicts regardless of outcome.
