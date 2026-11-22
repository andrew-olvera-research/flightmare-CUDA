# Cross-session packet loss in v0.4.1

The optional CUDA receiver samples each eligible packet at its original submission offset using recent simulated outcomes. This fixes the October 55e4 packet-pattern and outage-tail misses. Unrestricted cross-session prediction remains unqualified.

## Measurement and calibration

The prior comes from October 1–7 on link 5563. On the scoring link, each block contributes eight witnessed pilot updates followed by up to 24 blind updates. The estimator uses up to eight recent pilot blocks from the same day. It counts losses by replica position and by up to five previous packet outcomes. Sparse counts borrow one observation from a shorter history and one from the prior. Once a copy of the current update has landed, the second observation comes from local pilots with the same within-update history. Counter gaps and pilot boundaries break counted transitions. Each scored uninterrupted suffix starts with empty simulated history. Only the suffix lengths and pilot-derived probabilities reach the GPU. [Pinned receiver logs](https://github.com/wine-uoc/wisun_traces/tree/66c350d88d28fa39516fbc90722bdd3d9930d991)

The previous model counted every fully missed pilot triplet toward its outage estimate. Its residual packet sampler independently generated another 151–161 fully missed triplets on the original 55e4 scoring schedule. Its common probabilities across replica positions also missed the observed ordering of partial deliveries. Two of three seeds predicted a p95 run of three against four measured. The new profile models the observable packet sequence directly.

## GPU results

All comparisons use seeds 22–24 and the existing thresholds in `qualify_outages.py`. A profile has to pass every supported gate at every seed and meet the minimum event and day counts to qualify. The two October 55e4 windows were used during development. After freezing the local within-update calibration, we scored the previously unused August, September and final 55e4 windows, and the later windows on the three additional links.

| Link and scoring dates in 2019 | Measured updates | Result |
| --- | ---: | --- |
| 5563, October 8–14 | 6,697 | Qualified. Measured p95 was six and simulated p95 was six or seven. |
| 55e4, October 8–14 | 7,145 | Qualified. All seeds matched p95 of four. Mean delivery error was 0.5–1.8 percentage points. |
| 55e4, October 15–28 | 11,329 | Qualified. All seeds matched p95 of four. Mean delivery error was 1.7–2.2 points. |
| 55e4, August | 29,365 | Rejected. All seeds matched p95 of six. One missed the 3-point complete-loss limit with 3.014 points of error. |
| 55e4, September | 31,155 | Rejected. The capture had 655 complete misses. One seed exceeded the factor-of-two count limit. The 152 uncensored loss runs fell below the 200 needed to score p95. |
| 55e4, October 29–November 10 | 12,334 | Rejected. Observed 178 complete misses against 463–520 simulated. The pilots recorded 68 misses in 4,173 updates. The measured repeat-miss fraction was 10.8% across scored opportunities and 21.4% in pilots. |
| 560b, October 8–14 | 7,393 | Insufficient evidence. Zero complete misses. |
| 5632, October 8–14 and 15–28 | 7,377 and 13,554 | Insufficient evidence in the first window. One seed exceeded the miss-count limit in the second. |
| 5653, October 8–14 and 15–28 | 7,044 and 11,226 | Rejected on each. Two seeds gave p95 four against three in the first window. All matched four in the second but mean delivery error reached 7.8–8.4 points. |
| 630a, October 8–14 and 15–28 | 7,373 and 13,656 | First window rejected on complete-miss counts. Second window had zero complete misses and insufficient evidence. |

After scoring the final window, we repeated the three October comparisons using local pilots alone. The schedule, kernel, seeds, history length and thresholds were identical. On the final window this reduced predicted complete misses to 203–208 against 178 observed. All supported limits passed, with too few uncensored runs to qualify p95. In the October 8–14 window, two seeds returned to a p95 of three against four observed. October 15–28 continued to qualify. This control isolates sensitivity to borrowing the 5563 calibration. Choosing its weight for a new session remains open.

Five interleaved rounds on an RTX 5070 Ti used eight broadcasters, eight subscriptions per receiver, three 1,000-byte offers per sender per step and 10% loss in both tables. At 1 × 1,000 agents, the history sampler took 147.66 µs median and 161.76 µs p95, versus 147.60 and 161.79 µs for the ordinary sampler. Arena storage rose from 2,479,872 to 8,527,872 bytes. At 100 × 10, both medians were 51.20 µs. These timings exclude calibration and host reads.

## Reproduce

Run inside the development container. The fidelity lane enforces the three qualified regressions and retains the rejected and underpowered results in its JSON. Add `--require-qualified` to the scorer to enforce every reported window. That broader check currently fails.

```sh
PYTHONPATH=/build/network-ci/python /build/flightlib-cuda-python/bin/python tools/transport/qualify_history.py /build/network-measurements --download --output /tmp/history-transfer.json
python3 scripts/local_ci.py network-fidelity
python3 scripts/local_ci.py network-parity
python3 scripts/local_ci.py network-performance
```
