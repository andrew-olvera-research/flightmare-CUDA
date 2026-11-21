"""Compare saturated GPU unicast with the basic-access means in UCT Figure 5."""
import argparse
import csv
import json
import statistics
import subprocess

# Manually read from Figure 5, upper panel; rounding uncertainty about 0.002.
MEASURED = [.432, .470, .462, .444, .422, .400, .384, .367]
URL = "https://pubs.cs.uct.ac.za/id/eprint/534/1/Final_Version.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_test")
    args = parser.parse_args()
    output = subprocess.check_output([args.shared_test, "--saturation"], text=True)
    samples = {}
    for row in csv.reader(output.splitlines()):
        stations, replicate, frames = map(int, row)
        key = stations, replicate
        if key in samples or frames < 0:
            raise ValueError("invalid replicate")
        samples[key] = frames * 8000 / (.108 * 54_000_000)
    rows = []
    for n, reference in enumerate(MEASURED, 1):
        values = [samples[n, r] for r in range(32)]
        simulated = statistics.mean(values)
        rows.append(dict(senders=n, measured=reference, simulated=simulated,
                         simulated_sd=statistics.stdev(values),
                         relative_error=simulated/reference-1))
    passed = all(abs(row["relative_error"]) <= .10 for row in rows)
    print(json.dumps(dict(source=URL, approximate_means=True, replicates=32,
                          simulated_seconds_per_replicate=.108, rows=rows,
                          relative_error_limit=.10, passed=passed), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
