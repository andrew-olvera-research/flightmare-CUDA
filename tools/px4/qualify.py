"""Score the PX4 ULog against plant truth using EKF sample time on the SITL clock."""

import math
import numpy as np
from pyulog import ULog
from flightlib_cuda.px4_sitl import ALT, DT_US, FRD, LAT, LON


def evaluate(directory, steps):
    truth = np.load(directory / "truth.npy")
    if (
        len(truth) != steps
        or not np.all(np.diff(truth[:, 0]) == DT_US)
        or not np.all(truth[:, 0] - truth[:, 20] == DT_US)
    ):
        raise ValueError("missing tick or incorrect actuator application interval")
    air = truth[truth[:, 15].astype(bool)]
    if len(air) < 500 or not air[:, 14].all() or air[-1, 3] - truth[0, 3] < 1.5:
        raise ValueError("no sustained armed closed-loop climb")
    logs = list(directory.glob("log/**/*.ulg"))
    if len(logs) != 1:
        raise ValueError(f"expected one SITL ULog, found {len(logs)}")
    u = ULog(
        str(logs[0]),
        message_name_filter_list=["vehicle_local_position", "vehicle_attitude"],
    )

    def aligned(name):
        d = u.get_dataset(name).data
        t = d["timestamp_sample"].astype(np.int64)
        mask = (t >= air[0, 0]) & (t <= air[-1, 0])
        t = t[mask]
        if (
            len(t) < 3
            or t[0] - air[0, 0] > 250000
            or air[-1, 0] - t[-1] > 250000
            or np.any(np.diff(t) <= 0)
            or np.max(np.diff(t)) > 250000
        ):
            raise ValueError(f"insufficient time coverage of {name}")
        state = np.stack(
            [np.interp(t, truth[:, 0], v) for v in truth[:, 1:14].T], axis=1
        )
        return {k: v[mask] for k, v in d.items()}, state

    d, state = aligned("vehicle_local_position")
    if not all(
        np.all(d[k]) for k in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid")
    ):
        raise ValueError("EKF position or velocity invalid during flight")
    # PX4 may select a local origin slightly displaced from the simulated GNSS datum.
    offset = np.column_stack(
        (
            np.radians(LAT - d["ref_lat"]) * 6371000,
            np.radians(LON - d["ref_lon"]) * 6371000 * math.cos(math.radians(LAT)),
            d["ref_alt"] - ALT,
        )
    )

    def rms(v):
        return float(np.sqrt(np.mean(np.sum(v * v, axis=1))))

    pos = rms(
        np.column_stack([d[k] for k in ("x", "y", "z")]) - (state[:, :3] * FRD + offset)
    )
    vel = rms(
        np.column_stack([d[k] for k in ("vx", "vy", "vz")]) - state[:, 7:10] * FRD
    )
    a, state = aligned("vehicle_attitude")
    q = state[:, 3:7] * [1, 1, -1, -1]
    est = np.column_stack([a[f"q[{i}]"] for i in range(4)])
    dot = np.sum(q * est, axis=1) / (
        np.linalg.norm(q, axis=1) * np.linalg.norm(est, axis=1)
    )
    att = float(
        np.sqrt(np.mean(np.degrees(2 * np.arccos(np.clip(abs(dot), 0, 1))) ** 2))
    )
    if not (pos < 0.2 and vel < 0.2 and att < 3.0):
        raise ValueError(
            f"EKF airborne RMS outside qualification limits: {pos}, {vel}, {att}"
        )
    return dict(
        matched_ticks=steps,
        airborne_ticks=len(air),
        climb_m=float(air[-1, 3] - truth[0, 3]),
        position_samples=len(d["x"]),
        attitude_samples=len(est),
        position_rmse_m=pos,
        velocity_rmse_mps=vel,
        attitude_rmse_deg=att,
    )
