"""Reproduce measured network evidence using only the pinned local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def evaluate(script, arguments, output):
    result = subprocess.run([sys.executable, HERE / script, *arguments],
                            text=True, capture_output=True)
    if result.returncode not in (0, 1) or not result.stdout.strip():
        raise ValueError(result.stderr.strip() or f"{script} exited {result.returncode}")
    report = json.loads(result.stdout)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def replay(binary, flag, trace, count):
    masks = trace.read_bytes()
    if not count or len(masks) != count:
        raise ValueError(f"expected {count} measured masks, extracted {len(masks)}")
    result = subprocess.run([binary, flag], input=masks, capture_output=True)
    if result.returncode or result.stdout.strip() != str(count).encode():
        raise RuntimeError(f"expected {count} verified masks, stdout={result.stdout!r}, "
                           f"exit={result.returncode}, stderr={result.stderr[-1500:]!r}")
    return dict(masks=count, sha256=hashlib.sha256(masks).hexdigest(), tolerance=0)


def physical(scope, passed, required, metrics, limits):
    return dict(stage="physical_fit", scope=scope, status="qualified" if passed else "rejected",
                required=required, observed=metrics, tolerance=limits)


def measure(name, cache, build, output):
    results = []
    stage = "measurement"
    try:
        if name == "saturation":
            stage = "engine_run"
            report = evaluate("validate_shared.py", [build / "shared_test"], output / "saturation.json")
            stage = "measurement"
            baseline = read_json(HERE / "shared_validation_results.json")
            refs = [r["measured"] for r in report["rows"]]
            if refs != [r["measured"] for r in baseline["rows"]]:
                raise ValueError("Figure 5 reference means changed")
            results.append(dict(stage=stage, status="verified", source=report["source"],
                                means_sha256=hashlib.sha256(json.dumps(refs).encode()).hexdigest()))
            results.append(physical("1-8 saturated senders, Figure 5", report["passed"], True,
                                    report["rows"], report["relative_error_limit"]))
            return results

        trace = output / f"{name}.masks"
        if name == "bristol":
            report = evaluate("validate_bristol.py", [cache, "--replay-output", trace],
                              output / "bristol.json")
            baseline = read_json(HERE / "bristol_holdout.json")
            if report["replay_sha256"] != baseline["replay_sha256"]:
                raise ValueError("extracted broadcast masks changed on pinned inputs")
            results.append(dict(stage=stage, status="verified", sources=report["sources"]))
            expected = {(r["direction"], r["band"]): r["passed"] for r in baseline["scores"]}
            observed = {(r["direction"], r["band"]): r for r in report["scores"]}
            if expected.keys() != observed.keys():
                raise ValueError("supported directional distance bands changed")
            for key, required in expected.items():
                row = observed[key]
                results.append(physical(f"Feb 13 direction {key[0]} band {key[1]}", row["passed"],
                                        required, row["absolute_error"], .10))
            count = sum(r["denominator"] for r in baseline["broadcast"])
            binary, flag = build / "shared_test", "--broadcast-replay"
        else:
            report = evaluate("validate_wisun.py", [cache / "data_5563.csv", "--replay-output", trace],
                              output / "wisun.json")
            results.append(dict(stage=stage, status="verified", source=report["source"],
                                sha256=report["sha256"]))
            for month, models, limits, passed, required in (
                ("September", "models", report["limits"], "passed", True),
                ("October", "october_models", {**report["limits"], **report["october_limits"]},
                 "qualified", False),
            ):
                model = report[models]["empirical_triplet"]
                results.append(physical(f"{month} stationary FSK triplets", model[passed],
                                        required, model, limits))
            baseline = read_json(HERE / "wisun_5563_results.json")
            count = (baseline["cuda_replayed_attempts"] + baseline["october_cuda_replayed_attempts"]) // 3
            binary, flag = build / "packet_test", "--replay"
        stage = "engine_replay"
        results.append(dict(stage=stage, status="matched", **replay(binary, flag, trace, count)))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        results.append(dict(stage=stage, status="error", detail=str(exc)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {name: measure(name, args.cache, args.build, args.output)
                for name in ("bristol", "wisun", "saturation")}
    failures = [r for rows in datasets.values() for r in rows
                if r["status"] == "error" or (r.get("required") and r["status"] == "rejected")]
    (args.output / "fidelity.json").write_text(json.dumps(
        dict(schema=1, evidence_gate_passed=not failures, datasets=datasets), indent=2) + "\n")
    for name, rows in datasets.items():
        print(name, ", ".join(r.get("scope", r["stage"]) + "=" + r["status"] for r in rows))
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
