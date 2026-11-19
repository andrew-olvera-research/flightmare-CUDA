"""Single-vehicle PX4 v1.17 simulator TCP link. Body FLU, local NWU metres."""

import math
import socket
import time
from collections import deque

import numpy as np
import torch
from pymavlink.dialects.v20 import common as mav

DT_US = 4000
LAT, LON, ALT = 47.397742, 8.545594, 488.0
FRD = np.array([1.0, -1.0, -1.0])


def rotor_actions(controls):
    """Default 1 kg Flightlib model and PX4 none_iris motor functions 1..4."""
    u = np.asarray(controls, dtype=float)[[2, 0, 3, 1]]
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError("invalid PX4 non-reversible rotor output")
    # Linear thrust curve, 50% hover. Flightlib default mean=mg/4, scale=mg/2.
    return u - 0.5


def body_field(q):
    w = q[0]
    xyz = q[1:]
    # Fixed local magnetic field in NWU (gauss); inverse body-to-world rotation.
    field = np.array([0.215, -0.012, -0.43])
    return FRD * (field + 2 * np.cross(xyz, np.cross(xyz, field) - w * field))


class Link:
    def __init__(self, sock):
        self.sock = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.codec = mav.MAVLink(self, srcSystem=1, srcComponent=51)
        self.buffer = deque()
        self.noise = np.random.default_rng(17)

    def write(self, data):
        self.sock.sendall(data)

    def send(self, us, imu, state, tick):
        x, y, z = state[:3]
        vn, ve, vd = state[7:10] * FRD
        if tick % 25 == 0:
            lat = LAT + math.degrees(x / 6371000)
            lon = LON - math.degrees(y / (6371000 * math.cos(math.radians(LAT))))
            self.codec.hil_gps_send(
                us,
                3,
                round(lat * 1e7),
                round(lon * 1e7),
                round((ALT + z) * 1000),
                50,
                80,
                round(math.hypot(vn, ve) * 100),
                round(vn * 100),
                round(ve * 100),
                round(vd * 100),
                65535,
                12,
            )
        field = body_field(state[3:7])
        hpa = 1013.25 * (1 - (ALT + z) / 44330.0) ** 5.255
        flags = 63  # XYZ accelerometer and gyro
        if tick % 5 == 0:
            flags |= (
                448 | 512 | 2048 | 4096
            )  # XYZ magnetic field, pressure, altitude, temperature
            # A stationary physical sensor still fluctuates. PX4 detects long
            # runs of identical readings as stuck measurements.
            field += self.noise.normal(0.0, 0.0005, 3)
            hpa += self.noise.normal(0.0, 0.012)
        self.codec.hil_sensor_send(
            us,
            *(imu[3:] * FRD),
            *(imu[:3] * FRD),
            *field,
            hpa,
            0.0,
            ALT + z,
            15.0,
            flags,
        )

    def actuator(self, expected, timeout=2.0, *, startup=False):
        deadline = time.monotonic() + timeout
        while True:
            while self.buffer:
                msg = self.buffer.popleft()
                if msg.get_type() != "HIL_ACTUATOR_CONTROLS":
                    continue
                if msg.time_usec < expected and startup:
                    continue
                if msg.time_usec != expected or not (msg.flags & 1):
                    raise RuntimeError(
                        f"PX4 lockstep violation at {expected}: received {msg.time_usec}, flags={msg.flags}"
                    )
                return msg
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no actuator frame for {expected}")
            self.sock.settimeout(remaining)
            data = self.sock.recv(8192)
            if not data:
                raise ConnectionError("PX4 simulator socket closed")
            self.buffer.extend(self.codec.parse_buffer(data) or ())


class SingleAgentSITL:
    """Drive a fresh 250 Hz ROS bridge from a connected PX4 simulator socket.

    This fixture uses the default 1 kg rotor-thrust plant and a stationary
    level cradle at the reset pose until the controller commands liftoff.
    The caller owns the bridge, socket and PX4 process.
    """

    def __init__(self, bridge, sock):
        config = bridge.imu_config
        if (
            bridge.tick
            or bridge.tick_ns != DT_US * 1000
            or config.rate_hz != 250
            or config.latency_s
            or config.dropout
            or config.phase_s
            or tuple(config.mount_wxyz) != (1.0, 0.0, 0.0, 0.0)
        ):
            raise ValueError(
                "requires a fresh 4 ms bridge with a synchronous aligned 250 Hz IMU"
            )
        self.bridge = bridge
        self.host_action = torch.empty(4, dtype=torch.float32, pin_memory=True)
        self.host_state = torch.empty(13, dtype=torch.float32, pin_memory=True)
        self.link = Link(sock)
        stamp = 0
        # The external clock must run while PX4 starts its output driver.
        state = self.bridge.env.reset_state[:13, 0].cpu().numpy()
        for tick in range(1, 3001):
            stamp += DT_US
            self.link.send(stamp, np.array([0, 0, 0, 0, 0, 9.81]), state, tick)
            try:
                ack = self.link.actuator(stamp, 0.006, startup=True)
            except TimeoutError:
                continue
            break
        else:
            raise RuntimeError("PX4 never started lockstep output")
        self.epoch = stamp
        self.released = False
        self.ack = ack
        self.tick = 0

    def step(self):
        """Apply feedback for t, integrate to t+4 ms and wait for the matching output."""
        if self.bridge.tick != self.tick:
            raise RuntimeError("simulation advanced outside the lockstep loop")
        tick = self.tick + 1
        sim = self.bridge.env
        action_us = self.ack.time_usec
        armed = bool(self.ack.mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)
        actions = rotor_actions(self.ack.controls[:4]) if armed else np.full(4, -0.5)
        # Level launch cradle carries the vehicle until commanded thrust
        # exceeds weight. Flightlib has swept queries but no ground reaction.
        if not self.released:
            self.released = armed and actions.mean() > 0.005
        if not self.released:
            actions = np.zeros(4)
            sim.state[:13, 0].copy_(sim.reset_state[:13, 0])
        self.host_action.numpy()[:] = actions
        sim.actions[0, 0].copy_(self.host_action, non_blocking=True)
        _, samples = self.bridge.step_with_imu()
        if len(samples) != 1 or samples[0][0] != tick * DT_US * 1000:
            raise RuntimeError("missing current Flightlib IMU")
        imu = np.asarray(samples[0][1])
        self.host_state.copy_(sim.state[:13, 0])
        state = self.host_state.numpy()
        stamp = self.epoch + tick * DT_US
        self.link.send(stamp, imu, state, tick)
        self.ack = self.link.actuator(stamp)

        self.tick = tick
        return [stamp, *state, armed, self.released, *actions, action_us]
