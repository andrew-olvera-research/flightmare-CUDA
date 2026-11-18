"""Bounded subscriptions to completed world poses and delivered sensor packets."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
import math
import struct

import torch
from . import _native


def step_nanoseconds(dt):
    # Same float32 control period and positive llround as native dynamics.
    (value,) = struct.unpack("f", struct.pack("f", dt))
    return math.floor(value * 1e9 + 0.5)


@dataclass(frozen=True)
class Subscription:
    """actors is an iterable of (environment, slot) pairs, or None for all."""

    kind: str
    actors: tuple | None = None
    destination: str = "device"
    depth: int = 3
    required: bool = False
    period_ns: int | None = None  # pose deadline interval, defaults to each tick
    staging_depth: int | None = (
        None  # optional smaller device ring for a host subscriber
    )


class Backpressure(RuntimeError):
    """A required queue was full before the proposed step began. Drain and retry."""


class _Slot:
    def __init__(self, env, kind, ids, capacity, width):
        self.rows = torch.empty((capacity, width), dtype=torch.uint8, device=env.device)
        self.count = torch.empty(1, dtype=torch.uint32, device=env.device)
        self.pack = _native.NativeExport(env._native, kind, ids, self.rows, self.count)
        self.ready = self.pack.ready
        self.complete = torch.cuda.Event()
        # Torch events create their CUDA handles on the first record.
        self.complete.record()
        self.state = "free"


class _HostSlot:
    def __init__(self, capacity, width):
        self.host_rows = torch.empty(
            (capacity, width), dtype=torch.uint8, pin_memory=True
        )
        self.host_count = torch.empty(1, dtype=torch.uint32, pin_memory=True)
        self.complete = torch.cuda.Event()
        self.complete.record()
        self.source = None
        self.state = "free"


class Frame:
    """Lease on a single capture. Device consumers enqueue all reads on stream."""

    def __init__(self, channel, slot, stream):
        self._channel, self._slot, self.stream = channel, slot, stream
        self.sequence, self.requested_ns, self.acquired_ns = slot.identity
        if channel.host:
            rows = slot.host_rows[: slot.length]
            self.count = slot.length
        else:
            rows = slot.rows[:]
            self.count = slot.count[:]  # device scalar, first count rows are populated
        self.records = rows
        if channel.kind == "pose":
            self.episode = rows[:, :8].view(torch.int64)[:, 0]
            offset = 8
        else:
            self.packet_metadata = rows[:, :48].view(torch.int64)
            (
                self.packet_sequence,
                self.episode,
                self.scheduled_ns,
                self.packet_acquired_ns,
                self.available_ns,
                self.delivered_ns,
            ) = self.packet_metadata.unbind(1)
            offset = 48
        self.lane, self.valid = (
            rows[:, offset : offset + 8].view(torch.uint32).unbind(1)
        )
        self.values = rows[:, offset + 8 : offset + 8 + 4 * channel.channels].view(
            torch.float32
        )

    def release(self):
        self._channel._release(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class Channel:
    def __init__(self, env, spec, dt_ns):
        if not isinstance(spec, Subscription) or spec.kind not in (
            "pose",
            "imu",
            "odometry",
        ):
            raise ValueError("expected a pose, imu or odometry Subscription")
        if (
            spec.destination not in ("device", "host")
            or type(spec.depth) is not int
            or spec.depth < 1
        ):
            raise ValueError("invalid subscription destination or depth")
        if type(spec.required) is not bool:
            raise ValueError("required must be a boolean")
        self.kind, self.host = spec.kind, spec.destination == "host"
        staging = spec.depth if spec.staging_depth is None else spec.staging_depth
        if spec.staging_depth is not None and (
            not self.host or type(staging) is not int or not 1 <= staging <= spec.depth
        ):
            raise ValueError("host staging_depth must be between one and depth")
        if self.kind != "pose" and spec.period_ns is not None:
            raise ValueError(
                "sensor subscriptions export all delivered packets at the sensor's rate"
            )
        self.period = dt_ns if spec.period_ns is None else spec.period_ns
        if dt_ns < 1 or type(self.period) is not int or self.period < dt_ns:
            raise ValueError("pose period must be at least one positive control tick")
        n = env.slots * env.environments
        ids = range(n)
        if spec.actors is not None:
            ids = []
            for world, actor in spec.actors:
                if (
                    type(world) is not int
                    or type(actor) is not int
                    or not (0 <= world < env.environments and 0 <= actor < env.slots)
                ):
                    raise ValueError("invalid (environment, slot) selection")
                ids.append(world * env.slots + actor)
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("actor selection must be nonempty and unique")
        self.actors = len(ids)
        capacity = self.actors
        if self.kind != "pose":
            packets = getattr(env, self.kind)
            if packets is None:
                raise ValueError(f"enable {self.kind} before subscribing")
            capacity *= len(packets.values)
        self.channels, self.width = {
            "pose": (7, 48),
            "imu": (6, 80),
            "odometry": (10, 96),
        }[self.kind]
        self.required, self.device = spec.required, env.device
        self.closed = False
        self._lock = Lock()
        self._pending = deque()
        self._stats = dict(
            published=0,
            dropped=0,
            backpressure=0,
            acquired=0,
            high_water=0,
            host_bytes=0,
        )
        self.deadline = self.period
        with torch.cuda.device(self.device):
            selection = torch.tensor(ids, device=self.device, dtype=torch.uint32)
            self._initialized = torch.cuda.Event()
            self._initialized.record()
            self._transfer = (
                torch.cuda.Stream(device=self.device) if self.host else None
            )
            kind = getattr(_native.ExportKind, self.kind.upper())
            self._sources = [
                _Slot(env, kind, selection, capacity, self.width)
                for _ in range(staging)
            ]
            self._slots = self._sources
            if self.host:
                self._slots = [
                    _HostSlot(capacity, self.width) for _ in range(spec.depth)
                ]
                for slot in self._sources:
                    slot.rows.record_stream(self._transfer)
                    slot.count.record_stream(self._transfer)

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    @staticmethod
    def _free_slot(slots):
        for slot in slots:
            if slot.state == "retiring" and slot.complete.query():
                slot.state = "free"
            if slot.state == "free":
                return slot

    def _reserve(self):
        with self._lock:
            if self.closed:
                return None
            slot = self._free_slot(self._slots)
            if slot is not None:
                source = self._free_slot(self._sources) if self.host else slot
                if source is not None:
                    if self.host:
                        slot.source = source
                        source.state = "reserved"
                    slot.state = "reserved"
                    return slot
            if self.required:
                self._stats["backpressure"] += 1
                raise Backpressure(f"{self.kind} queue full")
            return None

    def _cancel(self, slot):
        with self._lock:
            if slot is not None and slot.state == "reserved":
                if self.host:
                    slot.source.state = "free"
                    slot.source = None
                slot.state = "free"

    def _copy_rows(self, slot, length):
        if length:
            slot.host_rows[:length].copy_(slot.source.rows[:length], non_blocking=True)
        slot.source.complete.record(self._transfer)
        slot.source.state = "retiring"
        slot.source = None
        slot.length = length
        slot.complete.record(self._transfer)
        slot.state = "copy"
        self._stats["host_bytes"] += length * self.width

    def _publish(self, slot, time_ns, stream):
        with self._lock:
            deadline = self.deadline
            self.deadline += self.period
            if self.closed:
                return
            if slot is None:
                self._stats["dropped"] += 1
                return
            stream.wait_event(self._initialized)
            source = slot.source if self.host else slot
            source.pack.pack(stream.cuda_stream)
            slot.identity = deadline // self.period, deadline, time_ns
            slot.state = "ready"
            if self.host:
                with torch.cuda.stream(self._transfer):
                    self._transfer.wait_event(source.ready)
                    if self.kind == "pose":
                        self._copy_rows(slot, self.actors)
                    else:
                        # A host reader advances the second transfer once the exact
                        # device count arrives. Empty/padded rows never cross PCIe.
                        slot.host_count.copy_(source.count, non_blocking=True)
                        slot.complete.record(self._transfer)
                        slot.state = "header"
                        self._stats["host_bytes"] += 4
            self._pending.append(slot)
            self._stats["published"] += 1
            occupied = sum(s.state != "free" for s in self._slots)
            self._stats["high_water"] = max(self._stats["high_water"], occupied)

    def acquire(self, stream=None, *, wait=False):
        """Pop the oldest capture. Host wait=True waits only for this capture's events."""
        while True:
            with self._lock:
                if self.closed or not self._pending:
                    return None
                slot = self._pending[0]
                if self.host:
                    if stream is not None:
                        raise ValueError(
                            "host frames do not take a CUDA consumer stream"
                        )
                    if slot.complete.query():
                        if slot.state == "header":
                            with torch.cuda.stream(self._transfer):
                                self._copy_rows(slot, int(slot.host_count.item()))
                        else:
                            return self._lease(slot, None)
                    if not wait:
                        return None
                    event = slot.complete
                else:
                    stream = (
                        torch.cuda.current_stream(self.device)
                        if stream is None
                        else stream
                    )
                    if stream.device != self.device:
                        raise ValueError("consumer stream device mismatch")
                    stream.wait_event(slot.ready)
                    slot.rows.record_stream(stream)
                    slot.count.record_stream(stream)
                    return self._lease(slot, stream)
            # A blocking host reader must not hold the producer's lock.
            event.synchronize()

    def _lease(self, slot, stream):
        frame = Frame(self, slot, stream)
        self._pending.popleft()
        slot.state = "leased"
        self._stats["acquired"] += 1
        return frame

    def _release(self, frame):
        with self._lock:
            slot = frame._slot
            if slot is None:
                return
            if frame.stream is None:
                slot.state = "free"
            else:
                slot.complete.record(frame.stream)
                slot.state = "retiring"
            frame._slot = None

    def close(self):
        """Disconnect, discard queued frames and drain submitted staging operations."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self._pending.clear()
            buffers = (self._slots, self._sources)
            events = [
                s.complete if self.host or s.state == "retiring" else s.ready
                for s in self._slots
                if s.state not in ("free", "reserved")
            ]
            # An outstanding lease keeps its own buffer alive across disconnect.
            self._slots = []
            self._sources = []
        for event in events:
            event.synchronize()
        for source in buffers[1]:
            source.pack = None
        del buffers


class Transport:
    def __init__(self, env, specs, dt):
        self.step_ns = step_nanoseconds(dt)
        self.time_ns = 0
        self.failed = False
        self.channels = tuple(Channel(env, spec, self.step_ns) for spec in specs)

    def step(self, native, stream, graph, random):
        if self.failed:
            raise RuntimeError(
                "subscription publication failed after step submission; close subscriptions"
            )
        end = self.time_ns + self.step_ns
        reserved = []
        try:
            for channel in self.channels:
                if not channel.closed and channel.deadline <= end:
                    reserved.append((channel, channel._reserve()))
            poses = any(c.kind == "pose" and slot is not None for c, slot in reserved)
            native.step(stream.cuda_stream, graph, random, poses)
            self.time_ns = end
            self.failed = True  # advancing the clock commits this step
            try:
                for channel, slot in reserved:
                    channel._publish(slot, end, stream)
            finally:
                if poses:
                    native.finish_transition(stream.cuda_stream, random)
            self.failed = False
        except BaseException:
            for channel, slot in reserved:
                channel._cancel(slot)
            raise
