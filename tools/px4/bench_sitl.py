"""Run one closed-loop Flightlib / ROS 2 / PX4 SITL vertical takeoff qualification."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
import rclpy
from flightlib_cuda import BarometerConfig, ImuConfig
from flightlib_cuda.ros2 import SingleAgentBridge
from flightlib_cuda.px4_sitl import ALT, SingleAgentSITL
from qualify import evaluate


def run(args, bridge, log):
    binary = args.build.resolve() / "bin/px4"
    env = dict(
        os.environ,
        PX4_SYS_AUTOSTART="10016",
        PX4_PARAM_MPC_THR_HOVER=".5",
        PX4_PARAM_THR_MDL_FAC="0",
        PX4_PARAM_MIS_TAKEOFF_ALT="2.5",
        PX4_PARAM_EKF2_MULTI_IMU="1",
        PX4_PARAM_SENS_IMU_MODE="1",
        PX4_PARAM_CA_ROTOR0_KM=".016",
        PX4_PARAM_CA_ROTOR1_KM=".016",
        PX4_PARAM_CA_ROTOR2_KM="-.016",
        PX4_PARAM_CA_ROTOR3_KM="-.016",
        PX4_PARAM_CA_ROTOR0_PX=".177",
        PX4_PARAM_CA_ROTOR1_PX="-.177",
        PX4_PARAM_CA_ROTOR2_PX=".177",
        PX4_PARAM_CA_ROTOR3_PX="-.177",
        PX4_PARAM_CA_ROTOR0_PY=".177",
        PX4_PARAM_CA_ROTOR1_PY="-.177",
        PX4_PARAM_CA_ROTOR2_PY="-.177",
        PX4_PARAM_CA_ROTOR3_PY=".177",
    )
    libs = args.build.resolve() / "lib"
    if libs.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(libs), env.get("LD_LIBRARY_PATH", ""))
        )
    rows = []
    latencies = []
    takeoff = None
    with socket.create_server(("127.0.0.1", 4560)) as server:
        server.settimeout(20)
        proc = subprocess.Popen(
            [binary, "-d", args.build.resolve() / "etc", "-w", args.output.resolve()],
            start_new_session=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            sock, _ = server.accept()
            with sock:
                loop = SingleAgentSITL(bridge, sock)
                started = time.perf_counter()
                for tick in range(1, args.steps + 1):
                    tick_start = time.perf_counter()
                    rows.append(loop.step())
                    latencies.append(time.perf_counter() - tick_start)
                    if tick == 3500:
                        # Normal SITL commander takeoff with preflight checks enabled.
                        takeoff = subprocess.Popen(
                            [binary.with_name("px4-commander"), "takeoff"],
                            env=env,
                            stdout=log,
                            stderr=log,
                        )
                bridge.drain()
                elapsed = time.perf_counter() - started
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if takeoff is not None:
                if takeoff.poll() is None:
                    takeoff.terminate()
                takeoff.wait()
            np.save(args.output / "truth.npy", np.array(rows))
    return elapsed, latencies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--camera-every", type=int)
    args = p.parse_args()
    if args.steps <= 3500:
        p.error("qualification requires a flight interval after the 3500-tick takeoff request")
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node("flightmare_sitl")
    bridge = SingleAgentBridge(
        node,
        control_dt=0.004,
        imu=ImuConfig(rate_hz=250),
        barometer=BarometerConfig(origin_altitude_m=ALT, noise_std_pa=1.2, seed=17),
        camera_every=args.camera_every,
    )
    try:
        with (args.output / "px4.log").open("w") as log:
            elapsed, latencies = run(args, bridge, log)
        result = evaluate(args.output, args.steps)
        result.update(
            elapsed_s=elapsed,
            ticks_per_s=args.steps / elapsed,
            realtime_factor=args.steps * 0.004 / elapsed,
            camera_every=args.camera_every,
            tick_wall_ms_p50_p95_p99=(
                np.percentile(latencies, [50, 95, 99]) * 1000
            ).tolist(),
        )
        report = json.dumps(result, indent=2)
        (args.output / "report.json").write_text(report)
        print(report)
    finally:
        bridge.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
