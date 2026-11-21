# Secondary packet simulator comparison

ns-3.44's wifi-bianchi example provides an independent saturated WLAN simulation. Build the wifi-bianchi CMake target with examples enabled and the wifi, internet-apps and applications modules. Tested at upstream revision 43dce6710b8df69685e3479c1d33a571dda714ea. Invoke the resulting executable in a fresh results directory.

```sh
ns3.44-wifi-bianchi --standard=11g --frequency=2.4 --phyMode=ErpOfdmRate54Mbps --infra=1 --nMinStas=1 --nMaxStas=8 --nStepSize=1 --trials=3 --duration=1 --pktInterval=100 --pktSize=1000 --plotBianchiModel=0 --validate=0
```

Match the number of saturated sources, common sink, nominal 54 Mbps rate and 1,000-byte packet size. The ns-3 experiment uses infrastructure mode and ns-3 MAC/retry defaults. The CUDA profile uses fixed-window single attempts and approximated overhead. These implementation differences limit interpretation of their divergence. The hardware comparison is a separate basic-access experiment. Throughputs below are aggregate Mbps; CUDA values are normalized results from the saturation fixture multiplied by 54.

| Sources | ns-3 | CUDA |
|---:|---:|---:|
| 1 | 24.85 | 25.02 |
| 2 | 25.52 | 26.10 |
| 3 | 25.57 | 25.34 |
| 4 | 25.36 | 24.44 |
| 5 | 25.18 | 23.06 |
| 6 | 24.95 | 21.65 |
| 7 | 25.21 | 20.47 |
| 8 | 25.06 | 19.13 |

At eight senders the two simulations differ by about 24% relative to ns-3. ns-3 remains nearly flat as contention increases; the CUDA profile's delivered throughput decreases. Reconcile MAC settings, retransmission accounting and per-station delay before transferring a profile across these scenarios. [ns-3 experiment source](https://github.com/nsnam/ns-3-dev-git/blob/43dce6710b8df69685e3479c1d33a571dda714ea/src/wifi/examples/wifi-bianchi.cc) · [Primary hardware comparison](shared_channel.md)
