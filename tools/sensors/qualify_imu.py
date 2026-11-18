"""Fit stationary six-axis noise on a time split and compare held-out Allan deviations to CUDA."""

import argparse
import json
import math
from pathlib import Path
from itertools import combinations
import numpy as np
import torch
from flightlib_cuda import Environment, ImuConfig


def allan_variance(data, widths):
    """Non-overlapping block means, then adjacent differences; preserve the lane axis."""
    rows = []
    for width in widths:
        n = len(data) // width
        blocks = (
            data[: n * width]
            .reshape(n, width, *data.shape[1:])
            .mean(axis=1, dtype=np.float64)
        )
        rows.append(0.5 * np.mean(np.diff(blocks, axis=0) ** 2, axis=0))
    return np.asarray(rows)


def fit_noise(variance, widths, rate, readout):
    # Allan variance = N²/tau + K²*(tau/3 + dt²/(6*tau)) for discrete bias increments.
    tau = widths / rate
    design = np.stack((1 / tau, tau / 3 + 1 / (6 * rate**2 * tau)), axis=1)
    if readout:
        design = np.column_stack((design, 1 / (4 * tau**2)))
    fits = []
    for target in variance.T:
        if not np.all(target > 0):
            raise ValueError("each axis needs nonzero stationary noise")
        a = design / target[:, None]
        y = np.ones(len(target))
        candidates = []
        for count in range(1, a.shape[1] + 1):
            for axes in combinations(range(a.shape[1]), count):
                fit = np.zeros(a.shape[1])
                fit[list(axes)] = np.linalg.lstsq(a[:, axes], y, rcond=None)[0]
                if np.all(fit >= 0):
                    candidates.append(fit)
        fits.append(min(candidates, key=lambda f: np.linalg.norm(a @ f - y)))
    return np.sqrt(fits).T


def cuda_residuals(noise, walk, width, rate, samples, lanes):
    # Zero thrust, zero rotation and unconstrained free fall give zero ideal output.
    period_ns = round(1e9 / rate)
    dt = np.float32(1 / rate)
    # Sensor periods are integer nanoseconds. Control periods are float32.
    # Keep the maximum physics substep admissible even when rounding differs.
    if rate > 400 and math.floor(float(dt) * 1e9 + 0.5) > period_ns:
        dt = np.nextafter(dt, np.float32(0))
    step_ns = math.floor(float(dt) * 1e9 + 0.5)
    if min(step_ns, period_ns) < 1:
        raise ValueError("sensor and simulation periods must be at least one nanosecond")
    duration = samples / rate
    height = 10 + 5 * duration**2
    env = Environment(
        1,
        lanes,
        control_dt=float(dt),
        models=None,
        bounds=[
            v
            for v in (-height, height, -height, height, -height, height * 2)
            for _ in range(lanes)
        ],
        imu=ImuConfig(
            rate_hz=rate,
            gyro_noise_density=noise[:3],
            accel_noise_density=noise[3:],
            gyro_random_walk=walk[:3],
            accel_random_walk=walk[3:],
            seed=509,
            gyro_increment_width=width[:3],
            accel_increment_width=width[3:],
        ),
    )
    env.actions.fill_(-1)
    env.reset_state[2] = height
    env.reset()
    env.capture(random=False)
    data = torch.empty((samples, lanes, 6), device=env.device)
    valid = torch.empty((samples, lanes), device=env.device, dtype=torch.uint8)
    written = 0
    time_ns = 0
    while written < samples:
        env.step(graph=True, random=False)
        time_ns += step_ns
        stop = min(samples, time_ns // period_ns)
        count = stop - written
        if count:
            data[written:stop].copy_(env.imu.values[:count])
            valid[written:stop].copy_(env.imu.valid[:count])
            written = stop
    if not valid.all() or env.imu.sequence[count - 1, 0].item() != samples:
        raise ValueError("CUDA run produced invalid or out-of-order IMU samples")
    return data.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording",
        type=Path,
        help="stationary uniformly sampled .npy [time,6] or CSV with gx,gy,gz,ax,ay,az in SI",
    )
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument(
        "--samples", type=int, default=16384, help="CUDA samples per independent lane"
    )
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument(
        "--increment-readout",
        action="store_true",
        help="also fit differenced cumulative readout error",
    )
    args = parser.parse_args()
    if args.recording.suffix == ".npy":
        data = np.load(args.recording, mmap_mode="r", allow_pickle=False)
    else:
        csv = np.genfromtxt(args.recording, delimiter=",", names=True)
        data = np.column_stack([csv[k] for k in ("gx", "gy", "gz", "ax", "ay", "az")])
    if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(data).all():
        raise ValueError("expected finite six-axis SI samples")
    if not math.isfinite(args.rate) or args.rate <= 0 or args.lanes < 1:
        raise ValueError("rate and lanes must be positive")
    split = len(data) // 2
    widths = np.array(
        [w for w in (1, 4, 16, 64, 256, 1024) if w * 16 <= min(args.samples, split)]
    )
    if len(widths) < (3 if args.increment_readout else 2):
        raise ValueError("insufficient samples for the requested number of noise terms")
    train = allan_variance(data[:split], widths)
    fits = fit_noise(train, widths, args.rate, args.increment_readout)
    noise, walk = fits[:2]
    width = fits[2] if args.increment_readout else np.zeros(6)
    measured = np.sqrt(allan_variance(data[split:], widths))
    sim = cuda_residuals(noise, walk, width, args.rate, args.samples, args.lanes)
    generated = np.sqrt(allan_variance(sim, widths).mean(axis=1))
    print(
        json.dumps(
            dict(
                training_samples=split,
                held_out_samples=len(data) - split,
                rate_hz=args.rate,
                axis_order=["gx", "gy", "gz", "ax", "ay", "az"],
                noise_density=noise.tolist(),
                bias_random_walk=walk.tolist(),
                increment_width=width.tolist(),
                tau_s=(widths / args.rate).tolist(),
                held_out_adev=measured.tolist(),
                cuda_adev=generated.tolist(),
                cuda_to_held_out=(generated / measured).tolist(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
