"""Score independent, distance-conditioned GPU broadcast reception on a withheld Bristol day."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

import torch
from flightlib_cuda import Environment, NetworkConfig
from validate_bristol import DAYS, EDGES


def masks_for_offers(offers, loss, seed):
    batch = min(len(offers), 2048)
    config = NetworkConfig(window_slots=1, range_edges_m=tuple(EDGES), seed=seed)
    bounds = [v for v in (-500000,500000,-10,10,0,20) for _ in range(3*batch)]
    env = Environment(batch, 3, network=config, bounds=bounds, auto_reset=False)
    env.reset_state[2].fill_(5)
    net = env.network
    probs = torch.tensor(loss, device=env.device)
    result = []
    for offset in range(0,len(offers),batch):
        chunk = offers[offset:offset+batch]
        n = len(chunk)
        distances = torch.tensor([dist for _,dist,_ in chunk], device=env.device)
        env.reset_state[0].view(batch,3)[:n,1:] = distances
        env.reset()
        with env.access():
            net.sources.zero_()
            net.size[:n,0] = 32  # Synthetic airtime. Only receiver decisions are scored.
            net.broadcast[:n,0] = 1
            net.packet_id[:n,0] = torch.arange(offset,offset+n,device=env.device)
            net.loss_after_rx[:,1:,0,:] = probs
            net.loss_after_miss.copy_(net.loss_after_rx)
        env.step(random=False)
        with env.access():
            if not net.evaluator["completed"][:n,0].all().item():
                raise RuntimeError("incomplete roadside transmissions")
            bits = net.inbox[:n,1:,0,0].to(torch.uint8)
            result.extend((bits[:,0]+2*bits[:,1]).tolist())
    return result


def rates(counts):
    return [counts[m]/sum(counts.values()) for m in range(4)]


def evaluate(path):
    counts = [[[0,0] for _ in range(len(EDGES)+1)] for _ in range(2)]
    held = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            dist = [float(row[f"distance_{i}"]) for i in range(2)]
            mask = int(row["mask"])
            if row["day"] == DAYS[0]:
                for i,d in enumerate(dist):
                    cell = counts[i][bisect_right(EDGES,d)]
                    cell[0] += 1
                    cell[1] += (mask >> i) & 1
            elif row["day"] == DAYS[1]:
                held.append((int(row["time_ms"]),dist,mask))
    supported = [row for row in held if all(counts[i][bisect_right(EDGES,d)][0] >= 200
                                            for i,d in enumerate(row[1]))]
    if not supported:
        raise ValueError("no held-out offers supported by calibration")
    loss = [[1-hits/n if n else 0 for n,hits in peer] for peer in counts]
    measured = rates(Counter(mask for _,_,mask in supported))
    # Diagnostic only. Re-estimate the peers' marginal rates on the scoring day
    # within the same joint distance cells to isolate residual co-reception.
    cells = defaultdict(Counter)
    for _, distances, mask in supported:
        cells[tuple(bisect_right(EDGES,d) for d in distances)][mask] += 1
    independent_both = 0
    independent_neither = 0
    for counts_in_cell in cells.values():
        p0, p1, p2, p3 = rates(counts_in_cell)
        weight = sum(counts_in_cell.values()) / len(supported)
        independent_both += weight*(p1+p3)*(p2+p3)
        independent_neither += weight*(p0+p1)*(p0+p2)
    models = []
    for seed in (22,23,24):
        predicted = masks_for_offers(supported,loss,seed)
        p = rates(Counter(predicted))
        errors = dict(mask_tv=sum(abs(a-b) for a,b in zip(p,measured))/2,
                      both_error=p[3]-measured[3], neither_error=p[0]-measured[0])
        # Block bootstrap the paired errors; both trajectories share the time blocks.
        blocks = defaultdict(list)
        for (t,_,mask),sim in zip(supported,predicted):
            blocks[t//60000].append(int(sim==3)-int(mask==3))
        sums = [(sum(b),len(b)) for b in blocks.values()]
        rng = random.Random(24)
        samples = []
        for _ in range(1000):
            draw = rng.choices(sums,k=len(sums))
            samples.append(sum(a for a,_ in draw)/sum(b for _,b in draw))
        samples.sort()
        models.append(dict(seed=seed, mask_probabilities=p, errors=errors,
                           both_error_minute_bootstrap_95=[samples[24],samples[974]],
                           passes=errors["mask_tv"] <= .10 and abs(errors["both_error"]) <= .05
                           and abs(errors["neither_error"]) <= .05))
    return dict(schema=1, fit_day=DAYS[0], held_day=DAYS[1], range_edges_m=EDGES,
                calibration_counts=counts, eligible_geometry_rows=len(held), supported=len(supported),
                measured_mask_probabilities=measured, seeds=models,
                held_day_marginal_diagnostic=dict(independent_both=independent_both,
                                                  independent_neither=independent_neither,
                                                  observed_minus_independent_both=measured[3]-independent_both), limits=dict(mask_tv=.10, both=.05, neither=.05),
                scope="distance-supported roadside offers with geometry for both receivers, one held-out session",
                verdict="qualified" if all(m["passes"] for m in models) else "rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-qualified", action="store_true",
                        help="exit unsuccessfully when the candidate misses a qualification threshold")
    args = parser.parse_args()
    result = evaluate(args.geometry_csv)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["verdict"], "for", result["supported"], "supported two-receiver offers")
    raise SystemExit(int(args.require_qualified and result["verdict"] != "qualified"))
