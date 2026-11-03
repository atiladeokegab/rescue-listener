"""The only module that touches `cl`. Tomorrow, on the CL1, only this file should change.

Real cl-sdk 1.0 if installed (`uv add cl-sdk`), else cl_mock (same surface). Seeded from cl-notes/adapter_sketch.py;
names and signatures per cl-notes/api-map.md. Lines marked `# CHECK:` are not confirmed by api-map.md.
What it hides from bci.py / rhythm.py:
  - tick.timestamp is deprecated: ours is tick.analysis.start_timestamp (the first frame of the tick's data)
  - tick.frames: raw int16 (25, 64) -> float32 uV (x 0.195); read()'s from_timestamp is positional-only
  - plan.run() returns when the pulse should land (iteration_timestamp + lead); the caller locks its windows to
    the real echo in tick.analysis.stims (stim_time())
  - safety: <= 3 uA, <= 3 nC per phase, biphasic alternating, <= 200 Hz per channel, never 0, 7, 56, 63 or 4,
    checked HERE as well, because the SDK labels its validators "Simulator only"
  - a loop overrun (TimeoutError on the device) is recovered with loop.recover_from_jitter() and counted
"""
from contextlib import contextmanager

import numpy as np

try:
    import cl as _cl  # real CL1 / cl-sdk
except ImportError:
    import cl_mock as _cl

MOCK = _cl.__name__ == "cl_mock"
FPS = 25_000
UV_PER_UNIT = 0.195  # CHECK: notebook CL-05; confirm on the device (recording attribute uV_per_sample_unit)
NO_STIM = frozenset((0, 7, 56, 63, 4))  # grounded (the API drops them silently) + reference 4 (the API allows it: we don't)
MAX_UA, MAX_NC, MAX_HZ, LEAD_US = 3.0, 3.0, 200, 80
MAX_JITTER_RECOVERIES = 20
ChannelSet = _cl.ChannelSet
ChannelQueueFull, TransactionRejected = _cl.ChannelQueueFull, _cl.TransactionRejected


def is_simulator():
    return _cl.is_simulator()


class StimDesign:
    """Checked StimDesign(duration_us, current_uA, ...): biphasic, alternating sign, <= 3 uA, <= 3 nC per phase."""

    def __init__(self, *phases):
        amps = phases[1::2]
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in phases):  # NaN passes every comparison
            raise ValueError(f"unsafe stim {phases}: finite numbers only")
        if any(us < 20 or us % 20 for us in phases[::2]):
            raise ValueError(f"unsafe stim {phases}: durations >= 20 us and multiples of 20 us")
        if len(phases) not in (4, 6) or any(a * b >= 0 for a, b in zip(amps, amps[1:])):
            raise ValueError(f"unsafe stim {phases}: biphasic with alternating sign only")
        for us, ua in zip(phases[::2], amps):
            if abs(ua) > MAX_UA or abs(us * ua) / 1000 > MAX_NC:
                raise ValueError(f"unsafe stim {phases}: over {MAX_UA} uA or {MAX_NC} nC per phase")
        self.phases, self.real = phases, _cl.StimDesign(*phases)


class BurstDesign:
    def __init__(self, count, hz):
        if not 0 < hz <= MAX_HZ:
            raise ValueError(f"burst over {MAX_HZ} Hz")
        self.count, self.hz, self.real = int(count), hz, _cl.BurstDesign(int(count), float(hz))


class _Tick:
    """The real tick is only valid inside the loop body; so is this."""
    __slots__ = ("raw", "timestamp", "analysis")

    def __init__(self, tick):
        self.raw, self.analysis = tick, tick.analysis
        self.timestamp = tick.analysis.start_timestamp  # first frame of this tick's data

    @property
    def frames(self):  # (frames_per_tick, 64) float32 uV, same span as analysis
        return self.raw.frames.astype(np.float32) * UV_PER_UNIT


def stim_time(tick, channels):
    """Timestamp of a stim on any of `channels` that began in this tick's window, or None."""
    for s in tick.analysis.stims:
        if s.channel in channels:
            return int(s.timestamp)
    return None


class _Plan:
    def __init__(self, nr):
        self.nr, self.items = nr, []

    def stim(self, channels, design, burst=None):
        chs = [int(c) for c in channels]
        if not chs or set(chs) & NO_STIM:
            raise ValueError(f"refusing non-stimmable channels {sorted(set(chs) & NO_STIM)}")
        if not isinstance(design, StimDesign) or not (burst is None or isinstance(burst, BurstDesign)):
            raise TypeError("use adapter.StimDesign / adapter.BurstDesign: they carry the safety checks")
        self.items.append((channels, chs, design, burst))

    def run(self):
        """Run now; returns the expected pulse timestamp (lock windows to stim_time() when the echo arrives)."""
        nr = self.nr
        # stims issued in a tick body land at that tick's iteration_timestamp + lead time (api-map)
        t = (nr.iteration_ts if nr.iteration_ts is not None else nr.n.timestamp()) + LEAD_US * FPS // 1_000_000
        for _, chs, _, b in self.items:
            last = t + (int((b.count - 1) * FPS / b.hz) if b else 0)
            for c in chs:
                if c in nr.last_pulse and t - nr.last_pulse[c] < FPS / MAX_HZ:
                    raise ValueError(f"channel {c}: over {MAX_HZ} Hz")
            nr.last_pulse.update(dict.fromkeys(chs, last))
        p = nr.n.create_stim_plan()
        for channels, _, d, b in self.items:
            p.stim(channels, d.real, b.real if b else None)  # positional: (channel_set, design, /, burst_design)
        p.run()  # frozen after run: one plan per trial
        return t


class Neurons:
    def __init__(self, n):
        self.n, self.iteration_ts, self.last_pulse, self.jitter_recoveries = n, None, {}, 0

    def timestamp(self):
        return self.n.timestamp()  # 25 kHz frame counter; resets if the device restarts

    def create_stim_plan(self):
        return _Plan(self)

    def read(self, frame_count, from_timestamp=None):  # not used in the loop (tick.frames is); float32 uV
        return self.n.read(frame_count, from_timestamp).astype(np.float32) * UV_PER_UNIT

    def loop(self, ticks_per_second, stop_after_seconds=None):
        lp = self.n.loop(ticks_per_second, stop_after_seconds=stop_after_seconds)
        while True:
            try:
                for tick in lp:
                    self.iteration_ts = tick.iteration_timestamp
                    yield _Tick(tick)
                return
            except TimeoutError:
                # the device raises when a body overruns the tick (the simulator only logs it). Recover and carry on;
                # the frames skipped are stale in our ring, so a trial straddling the gap is noisier.
                # CHECK: that iterating the same Loop again after recover_from_jitter() resumes it on the device
                self.jitter_recoveries += 1
                if self.jitter_recoveries > MAX_JITTER_RECOVERIES:
                    raise
                lp.recover_from_jitter()


@contextmanager
def open():  # shadows builtin on purpose, like cl.open
    with _cl.open() as n:  # take_control=True: ControlRequestError if another process holds the CL1
        yield Neurons(n)
