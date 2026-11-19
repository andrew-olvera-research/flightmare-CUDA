"""Matched benchmark producers for single-aircraft course RL (CPU OpenMP / CUDA)."""

import ctypes as ct
from importlib.resources import files
from pathlib import Path
import tempfile

import numpy as np
import torch

LIBRARY = "/build/course-profile/libcourse_profile.so"
CONFIG = Path(__file__).resolve().parents[2] / "flightlib/configs/quadrotor_env.yaml"


class CpuWorld:
    def __init__(self, n, workers, queries=True, terminate=True, poses=True):
        import yaml
        from flightlib_cuda.scene import _compile

        if workers < 1:
            raise ValueError("invalid CPU worker count")
        lib = self.lib = ct.CDLL(LIBRARY)
        ptr = ct.c_void_p
        lib.course_create.argtypes = [ct.c_int, ct.c_int, ct.c_char_p, ptr, ct.c_int]
        lib.course_create.restype = ptr
        lib.course_step.argtypes = [
            ptr,
            ptr,
            ct.c_bool,
            ct.c_bool,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
        ]
        lib.course_destroy.argtypes = [ptr]
        lib.course_destroy.restype = None
        cfg = yaml.safe_load(CONFIG.read_text())
        cfg["quadrotor_dynamics"].update(
            mass=1.0, arm_l=0.25, motor_omega_max=2000.0, motor_tau=0.05
        )
        nodes = np.array(
            _compile(
                files("flightlib_cuda")
                .joinpath("assets", "split_s.flightcourse")
                .read_bytes()
            ),
            dtype=np.float32,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp.flush()
            self.ptr = lib.course_create(
                n, workers, tmp.name.encode(), nodes.ctypes.data, len(nodes)
            )
        if not self.ptr:
            raise RuntimeError("could not create CPU baseline")
        self.n = n
        self.queries, self.terminate = queries, terminate
        self.obs = np.empty((n, 12), np.float32)
        self.poses = np.empty((n, 7), np.float32) if poses else None
        self.rew = np.empty(n, np.float32)
        self.done = np.empty(n, np.uint8)
        self.hits = np.empty(n, np.uint8)
        self.episodes = np.ones(n, np.int64)
        self.lanes = list(range(n))
        self.tick = 0
        self.counts = 0

    def advance(self, actions):
        a = np.ascontiguousarray(actions, dtype=np.float32)
        if a.shape != (self.n, 4):
            raise ValueError("invalid CPU action batch")
        if self.lib.course_step(
            self.ptr,
            a.ctypes.data,
            self.queries,
            self.terminate,
            self.obs.ctypes.data,
            self.poses.ctypes.data if self.poses is not None else None,
            self.rew.ctypes.data,
            self.done.ctypes.data,
            self.hits.ctypes.data,
        ):
            raise ValueError("invalid CPU transition")
        self.counts += int(self.hits.sum())
        self.tick += 1
        if self.poses is not None:
            self.capture_episodes = self.episodes.copy()
            self.episodes += self.done

    def packet(self):
        return dict(
            sequence=self.tick,
            requested_ns=self.tick * 20_000_000,
            acquired_ns=self.tick * 20_000_000,
            lanes=self.lanes,
            episodes=self.capture_episodes.tolist(),
            valid=[1] * self.n,
            poses=self.poses.tolist(),
        )

    def close(self):
        self.lib.course_destroy(self.ptr)


class CudaWorld:
    def __init__(self, n, queries=True, terminate=True, poses=True):
        from flightlib_cuda import Environment, Subscription

        self.env = e = Environment(
            n,
            1,
            control_dt=0.02,
            scene="split_s" if queries else None,
            terminate_on_collision=queries and terminate,
            collision_radius=0.125,
            subscriptions=(Subscription("pose", destination="host", required=True),)
            if poses
            else (),
        )
        lane = torch.arange(n, device=e.device)
        reset = e.reset_state
        reset[0] = -0.9 + (lane % 5 - 2) * 0.1
        reset[1] = -4.27 - (lane % 3) * 0.2
        reset[2] = 3.48
        reset[3] = reset[6] = 0.5**0.5
        e.reset()
        e.state[:3, 0] = torch.tensor([-0.1, -1.60, 3.48], device=e.device)
        e.state[8, 0] = 70
        e.capture(random=False)
        self.counts = torch.zeros((), device=e.device, dtype=torch.int64)
        self.channel = e.subscriptions[0] if poses else None

    def advance(self, action):
        e = self.env
        e.actions[:, 0].copy_(torch.as_tensor(action, device=e.device))
        o = e.step(graph=True, random=False)
        self.obs = o["observation"].reshape(-1, 12)
        self.rew = o["reward"].flatten()
        self.done = o["done"].flatten()
        self.valid = o["valid"].flatten()
        self.hits = e.collision_hit
        if self.hits is not None:
            self.counts.add_(self.hits.sum())

    def packet(self):
        with self.channel.acquire(wait=True) as f:
            return dict(
                sequence=f.sequence,
                requested_ns=f.requested_ns,
                acquired_ns=f.acquired_ns,
                lanes=f.lane.tolist(),
                episodes=f.episode.tolist(),
                valid=f.valid.tolist(),
                poses=f.values.tolist(),
            )

    def close(self):
        self.env.close_subscriptions()


def prepare(world, n, queries):
    world.advance(np.zeros((n, 4), np.float32))
    if queries and (int(world.hits[0]) != 1 or int(world.hits[1:].sum()) != 0):
        raise RuntimeError("expected only the swept gate crossing")
    if isinstance(world, CudaWorld) and world.channel:
        world.packet()
