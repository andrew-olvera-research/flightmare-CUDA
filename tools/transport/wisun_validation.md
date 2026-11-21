# Measured packet reception qualification

The native CUDA test supports replay of receiver losses. A lost attempt still consumes sender airtime and increments the accepted and modeled-loss counters. Receiver output contains delivered packet IDs only. Each input bit selects a measured receiver outcome. This mode operates after queue and deadline admission.

## Reproduce

Inside the development container, build the packet test following [the native benchmark instructions](packet_benchmark.md). Python 3.10 or later suffices for the analysis.

```sh
curl -fL https://raw.githubusercontent.com/wine-uoc/wisun_traces/66c350d88d28fa39516fbc90722bdd3d9930d991/data_5563.csv -o /tmp/data_5563.csv
python3 tools/transport/validate_wisun.py /tmp/data_5563.csv /build/packet-tests/packet_test > /tmp/wisun_results.json
```

The validator verifies the source SHA-256, rejects duplicate repetitions and sparse days, fits the August profile, compares two baselines, and replays September and October outcomes through CUDA. A replay mismatch aborts. A completed evaluation writes JSON and exits 1 when the empirical profile fails qualification. This pinned dataset currently produces that rejection. The default offline packet CTest exercises all eight reception patterns without downloading data.

## Measurement boundary

Node 55-63 is a fixed industrial sensor sending three repetitions per modulation in each scheduled period. We evaluate FSK, with a 32-byte PSDU. Each period enters the cohort when the gateway records at least one of the other two modulations for that same packet counter. For that period the denominator is three scheduled FSK repetitions. An absent FSK reading counts as a missed reception. Entirely unwitnessed periods are excluded, so this estimate can underrepresent complete outages. Transmission opportunities are inferred from firmware scheduling. The dataset has no independent transmitter log for these rows. [Dataset and collection protocol](https://www.mdpi.com/2306-5729/5/3/64) · [Authors’ data release](https://github.com/wine-uoc/wisun_traces/tree/66c350d88d28fa39516fbc90722bdd3d9930d991)

Each period is encoded by a mask from 0 to 7. Bit zero is the first FSK repetition. The empirical profile is the August distribution over the eight masks. The independent baseline fits mean reception. The two-state baseline fits initial reception and a transition probability for each preceding outcome, pooling the two transitions within a triplet. Statistical scores use exact model probabilities. GPU replay uses observed masks and verifies every ID, output count, modeled loss and airtime. Replay uses synthetic timing to exercise queued delivery. Physical delay, other packet sizes, shared contention, cross-minute outages and mobile links remain outside this qualification.

## Held-out results

We fit on August 1–14, 2019 and scored September 1–14 from the same deployment. The preset absolute-error limits were 5 percentage points on delivery, 3 points on complete loss, and 0.10 on total variation across the eight patterns. All three models passed September. The independent baseline nevertheless predicted about 0.004 complete misses across 19,153 periods, versus 73 observed.

Before scoring October 1–14, we added a complete-loss count gate requiring 20 observed events and a predicted count within a factor of two. We kept the August parameters and all September thresholds. The empirical profile predicted 48 complete misses across 18,587 October periods. The gateway recorded 2,032. All three models failed. This invalidates a stationary August calibration across these dates. The measurements alone do not identify the cause of the shift.

The report includes per-day sample sizes, fitted probabilities, errors and uncertainty intervals from resampling whole days within each holdout. These intervals describe variability within the observed deployment. [Machine-readable result](wisun_5563_results.json)
