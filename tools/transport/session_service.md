# Session loss and loaded service

## Results

The acknowledged-unicast candidate passes the independent ns-3 service comparison at 1, 4 and 8 synchronized sources. Receiver loss still needs local measurements to reproduce changes between sessions. The kernel now accepts separate outage transition rates for each subscribed direction. The comparisons below specify when those rates were available to the simulation.

### What changed between receiver sessions

On Wi-SUN link 5563, the probability that a received update was followed by a completely missed update rose from 2.24% in October week one to 5.87% in week two. The probability of recovering on the next update fell from 36.59% to 32.06%. Median signal strength among received FSK packets fell from −102 to −105 dBm. Late August and late September measured −96 and −98 dBm. Failed packets have no received signal reading. These observations establish a shift in link quality and outage entry rates. The capture cannot distinguish a physical obstruction, interference or a change in equipment as the cause.

| Candidate and scoring window | Result across three GPU seeds |
| --- | --- |
| Prior observations from a separate radio mode on link 5563 | Rejected in the October week split and both reserved historical windows. Even in the best probe class, 8.9% of October week-two updates were fully missed. This class lost less than 0.3% in each historical window. The mapping from probe quality to FSK delivery changes between sessions. |
| First 240 local observations, score the remainder | Rejected on both new links. On 5565 the prefix predicted 6,141–6,235 complete misses against 4,097. On 5599 it predicted 1,130–1,172 against 685. The beginning of the day overestimated loss later that day. |
| Periodic local calibration, blind intervening blocks on 55ad | Measured delivery was 68.46%. Pooled calibration predicted 83.4–84.5%. Local pilots predicted 70.0–70.3% and 3,760–3,816 complete misses against 3,870. Rejected for persistence. The p95 run was 4 misses against 3 observed and repeat-miss probability error was 11.6–12.0 percentage points against a 10-point limit. |
| Same pilot protocol on 55b3 | Measured delivery was 98.88%. Local pilots predicted 98.9–99.2%. Only 17 complete misses occurred in 14,756 scored periods. Insufficient evidence for outage counts and tails. |

Pilot periods are withheld from all reported scores. Periodic calibration assumes the caller can supply representative local measurements throughout the scenario. The experiment used 25% of each full witnessed block for that purpose. A simulation driven solely by topology and a traffic schedule has no such observations. [Pinned Wi-SUN measurements](https://github.com/wine-uoc/wisun_traces/tree/66c350d88d28fa39516fbc90722bdd3d9930d991)

### Why eight senders waited longer in ns-3

The original GPU sender chose from 16 backoff slots after every attempt. In ns-3 run 11 the station contention window switched between 16 and 32 choices after unacknowledged unicast frames, even with one on-air attempt per application packet. Holding this window fixed in ns-3 reduced the eight-sender p95 wait from 3,344 to 2,318 µs. The corrected CUDA window alone moved p95 from 2,656 to 3,727 µs. A second discrepancy concerned sender admission. CUDA freed the sender at data completion (180 µs after start). The ns-3 station updated its contention window at 226 or 221 µs after data start. The candidate reserves the sender for the corresponding additional 46 or 41 µs.

Fresh ns-3 random run 12 used three trials at each load. Rates are mean useful throughput over trials. Tail waits are medians of the per-sender/trial p95 intervals from application submission to PHY start. Each sender offered 10,000 synchronized 1,000-byte packets per measured second.

| Senders | Previous / candidate / ns-3 p95 wait (µs) | Previous / candidate / ns-3 useful throughput (Mbps) |
| ---: | ---: | ---: |
| 1 | 177 / 135 / 124 | 24.74 / 23.13 / 24.12 |
| 4 | 1,635 / 1,734 / 1,707 | 24.12 / 24.89 / 25.24 |
| 8 | 2,656 / 3,491 / 3,381 | 19.21 / 21.78 / 22.51 |

All three loads meet the specified 20% wait and 10% throughput bounds for this reference. The timing constants apply to this packet size and MAC profile. The sender infers a failed acknowledgement from a collision on its contention medium. Receiver-specific residual loss or hidden interference can still cause unacknowledged deliveries outside this workload. Hardware timing qualification remains open. [Per-trial reference](ns3_service_run12.json)

### GPU step cost

RTX 5070 Ti, five interleaved rounds, 128 timed public graph steps per round. Values are medians of the per-round median and p95 step times. In the loaded runs each of eight sources generated ten 1,000-byte offers in each 1 ms step. All receivers subscribed to all eight sources and there was one destination per offer.

| Shape and traffic | Fixed or pooled median / p95 (µs) | Candidate median / p95 (µs) | Device arena increment |
| --- | ---: | ---: | ---: |
| 1 environment × 1,000 agents, loaded unicast | 141.31 / 147.49 | 141.33 / 149.44 | 0 bytes |
| 100 environments × 10 agents, loaded unicast | 40.96 / 45.22 | 40.99 / 47.10 | 0 bytes |
| 1 environment × 1,000 agents, persistent broadcast, eight senders | 63.47 / 69.66 | 63.50 / 69.73 | 64,000 bytes |

The broadcast comparison keeps probabilities and offers equal and varies how the kernel reads the two transition probabilities. Supplying or estimating new external link measurements is outside this timer.

### Reproduction

Inside the dev container, set `PYTHONPATH=/build/network-ci/python` for the CUDA Python interpreter. With the base Wi-SUN capture and other existing fidelity inputs cached, run the two local-capture commands once with `--download` to populate the four checksum-pinned captures. The dedicated fidelity and performance lanes then run offline.

```sh
export PYTHONPATH=/build/network-ci/python
py=/build/flightlib-cuda-python/bin/python
csv=/build/network-measurements/data_5563.csv
"$py" tools/transport/qualify_probe.py "$csv" --prefix --download --output /tmp/prefix.json
"$py" tools/transport/qualify_probe.py "$csv" --pilots --download --output /tmp/pilots.json
python3 scripts/local_ci.py network-fidelity
python3 scripts/local_ci.py network-performance
"$py" tools/transport/bench_resident.py --session-service --senders 1 4 8 --rounds 5
```

Rebuild the pinned ns-3.44 wifi-bianchi example after applying `ns3-service.patch` followed by `ns3-ack-window.patch`. Use the synchronized 1/4/8-source commands in the original qualification report with `--serviceRun=11` for the window experiment. Repeat in a separate directory with `--serviceRun=11 --fixedWindow=1` for its control. Use `--serviceRun=12` for confirmation. Score each directory with `score_ns3_service.py DIRECTORY --output reference.json`. For CUDA run `compare_contention.py` with its default settings, with `--unicast-ack-backoff`, and with both `--unicast-ack-backoff --ack-hold`, supplying a distinct `--output` each time. Pass the resulting file to the reference scorer with `--cuda FILE`; `--require-qualified` enforces both gates. [Original reference commands](v041_fidelity.md)

## Frozen comparison

Fit Wi-SUN on October 1–14. Reserve August 15–31 and September 15–30 as two independent calendar windows for this candidate. They precede calibration, so these are historical transfer checks. Score every day with at least 720 independently witnessed periods, requiring seven such days per window. Preserve the seeds and full distribution, miss-count, repeat-miss and p95 gates in `qualify_outages.py`. Also use October week one to week two for diagnosis.

Compare a pooled link with a link conditioned on the preceding twelve consecutive opportunities of the separate O-QPSK probe. Classify the fraction of its 36 received replicas into below 0.70, below 0.94, and the remainder. Use pooled calibration before twelve observations or after a counter gap. No target FSK outcomes or current/future probe observations are supplied to the simulator. The conditional workload requires an independently supplied probe schedule. Estimate transition rates and residual loss per class on the calibration days. Preserve reception state across class changes.

For sender service, compare per-sender p95 submission-to-first-airtime and delivered Mbps with ns-3 at 1, 4 and 8 senders. Use trial-level results and a fresh ns-3 random run as confirmation for a changed profile. Report relative errors against 20% for tail wait and 10% for useful throughput at each load, including any regression at lighter loads. Keep the existing basic-access profile and the same single-outstanding-frame contract. Record GPU step cost separately.

After the preceding-probe candidate failed both windows, freeze a second, conditional use case. On the next two links in repository name order (5565 and 5599), use each eligible day's first 240 witnessed periods for calibration. Score only the remaining periods on October 1–14. Use one pseudo-observation per transition origin with the pooled 5563 October calibration as prior. Require 720 total witnessed periods per day and seven eligible days per link. Compare with the pooled model on the identical suffixes with the same seeds and gates. No suffix outcomes enter calibration. This tests adaptation with a local calibration capture.

The prefix overshot complete losses on both links. Freeze a distributed pilot comparison on both of the next unused links, 55ad and 55b3, for the same dates. In each block of 240 witnessed periods, use the first 60 as pilots and score the remainder. At each scored block estimate the local profile from up to four most recent pilot blocks of that day. Keep transitions contiguous within individual pilot or score runs and restart at counter gaps. Compare local and pooled profiles on identical scored blocks and retain the same gates. Publish both link verdicts.

Run 11 of the pinned ns-3 service fixture confirms the CW change and supplies a second mechanism check. For the 1,000-byte, 54 Mbps fixture, advance a candidate sender's admission clock through acknowledgement resolution. The observed start-to-release times are 226 µs on success and 221 µs on failure. Keep the existing 180 µs CUDA data occupancy in the ablation and use residual holds of 46 and 41 µs. Score the candidate as specified at all three loads on previously unused ns-3 run 12. Report the CW-only comparison alongside it.
