"""Replay the Cube Orange hardware fixture against the CUDA pressure model.

Input is PX4/pyulog test/sample_log_small.ulg at revision
0b392f8444f843b796405b9766cc15a6dddedf62. Requires pyulog, numpy and Flightlib.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pyulog import ULog
from flightlib_cuda import Environment, BarometerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    path = parser.parse_args().log
    if hashlib.sha256(path.read_bytes()).hexdigest() != "68d1020f688109f027dba08d2950247cc0ae34aaefdcf37e4a7d60838f3e1aa3":
        raise ValueError("unexpected fixture digest")
    log = ULog(str(path))
    assert log.msg_info_dict["ver_hw"] == "CUBEPILOT_CUBEORANGE"
    gps = log.get_dataset("vehicle_gps_position").data
    air = log.get_dataset("vehicle_air_data").data
    gt = gps["timestamp"][gps["fix_type"] >= 3]
    alt = gps["alt"][gps["fix_type"] >= 3] * 0.001  # MSL millimetres
    t = air["timestamp_sample"]
    overlap = (t >= gt[0]) & (t <= gt[-1])
    measured = air["baro_pressure_pa"][overlap].astype(float)
    height = np.interp(t[overlap], gt, alt)
    split = len(height) // 2
    qnh = np.median(measured[:split] / (1 - height[:split] / 44330.77) ** 5.25588)
    # Map each held-out GNSS height to an independent hover lane in one graph.
    h = height[split:]
    origin = float(np.floor(min(h)) - 1)
    env = Environment(len(h), control_dt=0.01, auto_reset=False,
                      barometer=BarometerConfig(rate_hz=100, origin_altitude_m=origin, sea_level_pa=qnh))
    env.reset_state[2].copy_(torch.as_tensor(h - origin))
    env.reset()
    env.capture(random=False)
    env.step(graph=True, random=False)
    assert env.barometer.valid[0].all()
    predicted = env.barometer.values[0, :, 0].cpu().numpy()
    residual = predicted - measured[split:]
    # First-difference statistic excludes constant offsets, but can contain motion.
    def variation(x):
        return float(np.std(np.diff(x), ddof=1) / np.sqrt(2))

    # Repeat a fixed-height acquisition using the calibration partition's
    # readout scale. Compare exactly the same statistic on the held-out signal.
    scale = variation(measured[:split])
    noise_env = Environment(1, control_dt=0.01, auto_reset=False,
                            barometer=BarometerConfig(rate_hz=100, origin_altitude_m=origin,
                                                      sea_level_pa=qnh, noise_std_pa=scale, seed=17))
    noise_env.reset_state[2].fill_(2)
    noise_env.reset()
    noise_env.capture(random=False)
    samples = []
    for _ in h:
        noise_env.step(graph=True, random=False)
        assert noise_env.barometer.valid[0, 0]
        samples.append(noise_env.barometer.values[0, 0, 0].item())
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    observed, simulated = variation(measured[split:]), variation(samples)
    print(json.dumps(dict(calibration_samples=split, held_out_samples=len(h), qnh_pa=qnh,
                          pressure_rmse_pa=rmse, pressure_bias_pa=float(residual.mean()),
                          gps_epv_median_m=float(np.median(gps["epv"])), calibration_noise_pa=scale,
                          held_out_variation_pa=observed, simulated_variation_pa=simulated), indent=2))
    # Coarse GNSS consistency envelope (local pressure gradient ~12 Pa/m).
    assert rmse < 12 * np.median(gps["epv"])
    assert 0.5 < simulated / observed < 2.0


if __name__ == "__main__":
    main()
