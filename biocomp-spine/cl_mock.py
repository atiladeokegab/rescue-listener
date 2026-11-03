"""Mock of the Cortical Labs `cl` module (cl-sdk 1.0.0 surface). Only adapter.py imports it.

Real-API shape: tick.frames int16 (25, 64) raw (x 0.195 = uV), tick.analysis.start_timestamp / .spikes / .stims,
read(frame_count, from_timestamp, /), loop() returns a Loop with stop() and recover_from_jitter(). Stims issued in a
tick body land at the tick's iteration_timestamp + lead time and echo in a later tick's analysis.stims.
Limits ENFORCED (raise, so tests catch what the CL1 might refuse or, worse, accept): <= 3 uA and <= 3 nC per phase,
biphasic with alternating sign, durations multiples of 20 us, <= 200 Hz per channel, lead time >= 80 us (multiple
of 40), never channels 0, 7, 56, 63 (grounded) or 4 (reference).

Toy culture, continuous in virtual time:
  - network bursts every 3.5-6 s; population rate decays back over ~1 s afterwards
  - the evoked response depends on WHICH channels were stimulated (neighbours on the 8x8 grid fire most, plus a few
    randomly wired partners), and is gain-modulated by the burst
    cycle (weak just after a burst, full strength in the quiet between bursts), plus noise and slow drift
  - short-term memory: paired-pulse facilitation and depression over 50-500 ms gaps (Tsodyks-Markram u, R)
  - NO_CELLS=1: dead dish. Gaussian noise, stim artefact and a trickle of noise spikes only.
Nothing sleeps unless MOCK_SPEED=<x> is set (then loop() runs at x times real time, for the live view).
"""
import bisect
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

FPS = 25_000
N_CH = 64
GROUNDED = (0, 7, 56, 63)
REFERENCE = (4,)
TAU = 1.0            # s, burst tail
BASE_HZ = 3.0        # spontaneous rate per live channel
BURST_X = 6.0        # rate multiplier at burst peak
# short-term memory (Tsodyks-Markram): facilitation u and resources R carry over 50-500 ms between pulses
STP_U, TAU_F, TAU_REC = 0.6, 0.05, 0.4   # mostly depressing: a pulse 60 ms after another gives ~40%, 500 ms ~70%
EVOKED_HZ, SPIKE_TAU = 200.0, 0.05   # evoked spike rate at r = 1, decay (s): ~4 spikes 10-50 ms at the peak
AMP = 2.0            # evoked oscillation amplitude (noise sigma = 1)
ARTEFACT_TAU = 0.002  # s: the stim artefact (40 sigma on the stimulated electrodes) decays with this
NOISE_UV, UV_PER_UNIT = 5.0, 0.195   # raw int16 units: noise sigma 5 uV = 25.6 units
NO_STIM = GROUNDED + REFERENCE
MAX_UA, MAX_NC, MAX_HZ = 3.0, 3.0, 200


class ChannelQueueFull(Exception):
    pass


class TransactionRejected(Exception):
    pass


def is_simulator():
    return True


class ChannelSet:
    def __init__(self, *channels):  # ChannelSet(*channels: int | Iterable[int])
        chs = [int(x) for c in channels for x in (c if hasattr(c, "__iter__") else (c,))]
        if not all(0 <= c < N_CH for c in chs):
            raise ValueError(f"channels must be 0-63: {chs}")
        self.channels = tuple(sorted(set(chs)))

    def __iter__(self):
        return iter(self.channels)


class StimDesign:
    """StimDesign(duration_us_1, current_uA_1[, d2, c2[, d3, c3]], /)."""

    def __init__(self, *phases):
        if len(phases) not in (2, 4, 6):
            raise ValueError("StimDesign takes 2, 4 or 6 values")
        if not all(np.isfinite(v) for v in phases):  # NaN slips past every comparison below
            raise ValueError("StimDesign values must be finite numbers")
        amps = phases[1::2]
        for dur, amp in zip(phases[::2], amps):
            if dur < 20 or dur % 20:
                raise ValueError("duration must be >= 20 us and a multiple of 20 us")
            if abs(amp) > MAX_UA:
                raise ValueError(f"{MAX_UA} uA current limit")
            if abs(dur * amp) / 1000 > MAX_NC:  # us * uA = pC; /1000 -> nC
                raise ValueError(f"{MAX_NC} nC per-phase charge limit")
        if len(amps) < 2 or any(a * b >= 0 for a, b in zip(amps, amps[1:])):
            raise ValueError("biphasic only: adjacent phases must alternate sign")
        self.phases = phases


class BurstDesign:
    def __init__(self, burst_count, burst_hz, /):
        if type(burst_count) is not int or burst_count < 1:
            raise ValueError("burst_count must be a Python int >= 1")
        if not 0 < burst_hz <= MAX_HZ:
            raise ValueError(f"burst_hz must be in (0, {MAX_HZ}]")
        self.burst_count, self.burst_hz = burst_count, burst_hz


@dataclass
class Spike:
    channel: int
    timestamp: int
    samples: np.ndarray = field(default=None, repr=False)


@dataclass
class Stim:
    channel: int
    timestamp: int


class _Analysis:
    def __init__(self, lo, hi, spikes, stims):
        self.start_timestamp, self.stop_timestamp, self.spikes, self.stims = lo, hi, spikes, stims


class _Tick:  # no deprecated .timestamp: the adapter must not lean on it
    def __init__(self, loop, i, lo, hi, spikes, stims, frames):
        self.loop, self.iteration, self.iteration_timestamp = loop, i, hi
        self.analysis, self.frames = _Analysis(lo, hi, spikes, stims), frames


class _Recording:
    # shortcut: real record() writes HDF5 and has more args; mock does nothing
    def stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()


class _Plan:
    def __init__(self, n):
        self.n, self.items, self.ran = n, [], False

    def stim(self, channels, design, /, burst_design=None, lead_time_us=80):
        if self.ran:
            raise TransactionRejected("a stim plan is frozen after run()")
        self.items.append((channels, design, burst_design, lead_time_us))

    def run(self, at_timestamp=None):  # shortcut: at_timestamp ignored (runs now), we never pass it
        self.ran = True
        for c, d, b, lead in self.items:
            self.n.stim(c, d, b, lead)


class _Loop:
    def __init__(self, n, tps, n_ticks):
        self.n, self.tps, self.n_ticks, self.stopped, self.recoveries = n, tps, n_ticks, False, 0

    def stop(self):
        self.stopped = True

    def recover_from_jitter(self, callback=None, timeout_seconds=5.0):
        self.recoveries += 1

    def __iter__(self):
        return self.n._ticks(self)


class Neurons:
    def __init__(self, no_cells):
        self.no_cells = no_cells
        seed = int(os.environ.get("CL_SDK_RANDOM_SEED", 0))
        crng = np.random.default_rng(seed)  # the "culture": fixed wiring and per-channel preferred frequency
        self.C = (crng.random((N_CH, N_CH)) < 0.15) * crng.random((N_CH, N_CH))
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
        self.f = np.array([crng.uniform(*bands[i % 5]) for i in range(N_CH)])
        self.live = np.array([c not in GROUNDED + REFERENCE for c in range(N_CH)])
        rc = np.array([(c // 8, c % 8) for c in range(N_CH)])
        dist = np.hypot(*(rc[:, None, :] - rc[None, :, :]).transpose(2, 0, 1))
        self.near = np.exp(-dist / 1.2) * self.live  # 8x8 grid: evoked strength falls off with distance
        self.rng = np.random.default_rng(seed + 1)
        self.off = np.zeros(N_CH)  # slow random-walk drift
        self.u, self.R, self._prev = STP_U, 1.0, None  # short-term plasticity state
        self.zg = 0.0  # log gain: excitability drifts slowly from pulse to pulse (AR(1), sd 0.3)
        self._t = 0
        self._trial = 0
        self._bursts = [int(1.5 * FPS)]
        self._last = None  # dict describing the latest stim
        self._stims, self._last_pulse = [], {}  # Stim echoes not yet delivered; last pulse time per channel
        self._evoked = []
        self.speed = float(os.environ.get("MOCK_SPEED", 0))

    # --- burst cycle ---------------------------------------------------------
    def tail(self, ts):
        """1.0 at a burst onset, decaying with TAU; 0 for a dead dish."""
        if self.no_cells:
            return 0.0
        while self._bursts[-1] < ts + 6 * FPS:
            self._bursts.append(self._bursts[-1] + int(self.rng.uniform(3.5, 6.0) * FPS))
        i = bisect.bisect_right(self._bursts, ts) - 1
        return float(np.exp(-(ts - self._bursts[i]) / FPS / TAU)) if i >= 0 else 0.0

    def _gain(self):
        self.zg = 0.9 * self.zg + np.sqrt(1 - 0.81) * self.rng.normal(0, 0.3)
        return float(np.exp(self.zg))

    def _stp(self, ts):
        """Paired-pulse facilitation and depression: 1.0 for an isolated pulse, more or less after recent ones."""
        if self._prev is not None:
            gap = (ts - self._prev) / FPS
            self.R = 1 - (1 - self.R) * np.exp(-gap / TAU_REC)
            self.u = STP_U + (self.u - STP_U) * np.exp(-gap / TAU_F)
        self._prev = ts
        out = self.u * self.R / STP_U
        self.R -= self.u * self.R
        self.u += STP_U * (1 - self.u)
        return out

    # --- real-API surface ----------------------------------------------------
    def timestamp(self):
        return self._t

    def get_channel_count(self):
        return N_CH

    def get_frames_per_second(self):
        return FPS

    def record(self, *a, **k):
        return _Recording()

    def create_stim_plan(self):
        return _Plan(self)

    def stim(self, channels, design, /, burst_design=None, lead_time_us=80):
        chs = list(channels if isinstance(channels, ChannelSet) else ChannelSet(channels))
        if isinstance(design, (int, float)):  # a float means StimDesign(160, -x, 160, x)
            design = StimDesign(160, -float(design), 160, float(design))
        if not isinstance(design, StimDesign) or (burst_design is not None and not isinstance(burst_design, BurstDesign)):
            raise TypeError("stim(channel_set, stim_design, /, burst_design=None, lead_time_us=80)")
        if not chs or set(chs) & set(NO_STIM):
            raise ValueError(f"refusing to stimulate {sorted(set(chs) & set(NO_STIM)) or 'no channels'}: grounded or reference")
        if lead_time_us < 80 or lead_time_us % 40:
            raise ValueError("lead_time_us must be >= 80 and a multiple of 40")
        burst = burst_design
        first = self._t + int(lead_time_us * FPS // 1_000_000)  # body runs at iteration_timestamp; stim lands after the lead
        last = first + (int((burst.burst_count - 1) * FPS / burst.burst_hz) if burst else 0)
        for c in chs:
            if c in self._last_pulse and first - self._last_pulse[c] < FPS / MAX_HZ:
                raise ChannelQueueFull(f"channel {c}: over {MAX_HZ} Hz")
            self._last_pulse[c] = last
        self._stims += [Stim(c, first) for c in chs]
        rng = self.rng
        r = np.zeros(N_CH)
        if not self.no_cells:
            self.off = 0.97 * self.off + rng.normal(0, 0.15, N_CH)  # mean-reverting so long runs stay bounded
            g = max(0.0, 1 - 1.2 * self.tail(last))  # weak just after a burst
            w = self.C[chs].sum(0)
            r = self.near[chs].sum(0) + 0.5 * w / (w.max() + 1e-9)  # neighbours fire, plus a few wired partners
            r = r / (r.max() + 1e-9) * g * self._gain() * (1 + 0.2 * np.sin(self._trial * 0.03)) * self._stp(first)
        self._last = dict(first=first, last=last, chs=chs, amp=AMP * r, ph=rng.uniform(0, 6.28, N_CH))
        # evoked spikes after the 10 ms artefact: rate (3 + EVOKED_HZ * r) decaying with SPIKE_TAU, up to 240 ms
        e0, e1 = np.exp(-0.01 / SPIKE_TAU), np.exp(-0.24 / SPIKE_TAU)
        n = rng.poisson((3 + EVOKED_HZ * r) * SPIKE_TAU * (e0 - e1) * self.live)
        ch = np.repeat(np.arange(N_CH), n)
        t = -SPIKE_TAU * np.log(e0 - rng.random(len(ch)) * (e0 - e1))  # truncated exponential spike times
        self._evoked = sorted((Spike(int(c), int(last + v * FPS)) for c, v in zip(ch, t)), key=lambda s: s.timestamp)
        self._trial += 1

    def read(self, frame_count, from_timestamp=None, /):
        """-> int16 (frame_count, 64) raw units (x 0.195 = uV). shortcut: renders with the latest stim only."""
        raw = self._sigma(frame_count, self._t if from_timestamp is None else from_timestamp)
        return np.clip(np.round(raw * (NOISE_UV / UV_PER_UNIT)), -32768, 32767).astype(np.int16)

    def _sigma(self, frame_count, start):  # the signal in noise-sigma units
        t = (np.arange(frame_count) + start) / FPS
        raw = self.rng.standard_normal((frame_count, N_CH), dtype=np.float32)
        if not self.no_cells:
            raw += self.off * np.sin(2 * np.pi * 0.7 * t)[:, None]
        L = self._last
        if L:
            tr = t - L["last"] / FPS  # seconds after the last pulse
            post = (tr >= 0) & (tr < 0.9)  # beyond 0.9 s the response is < 0.3% of its peak: skip the work
            if post.any():
                tp = tr[post][:, None]
                if tp[0, 0] < 0.05:  # stim artefact, first ~10 ms, present with or without cells
                    raw[np.ix_(post, L["chs"])] += 40.0 * np.exp(-tp / ARTEFACT_TAU)
                if not self.no_cells:
                    raw[post] += L["amp"] * np.sin(2 * np.pi * self.f * tp + L["ph"]) * np.exp(-tp / 0.15)
        raw[:, GROUNDED] = 0
        return raw

    def loop(self, ticks_per_second, stop_after_seconds=None, stop_after_ticks=None, ignore_jitter=False,
             jitter_tolerance_frames=0):
        n = stop_after_ticks if stop_after_seconds is None else int(stop_after_seconds * ticks_per_second)
        return _Loop(self, ticks_per_second, n)

    def _ticks(self, loop):
        dt = FPS // loop.tps
        n_ticks = loop.n_ticks
        i = k = 0
        ev, wall0 = self._evoked, time.perf_counter()
        while (n_ticks is None or k < n_ticks) and not loop.stopped:
            lo, hi = self._t, self._t + dt
            lam = np.where(self.live, BASE_HZ * (1 + BURST_X * self.tail(lo)) if not self.no_cells else 0.3, 0) * dt / FPS
            sp = [Spike(int(c), lo) for c in np.nonzero(self.rng.poisson(lam))[0]]
            if ev is not self._evoked:  # a new stim replaced the evoked list
                ev, i = self._evoked, 0
            while i < len(self._evoked) and self._evoked[i].timestamp < hi:
                if self._evoked[i].timestamp >= lo:
                    sp.append(self._evoked[i])
                i += 1
            frames = self.read(dt, lo)
            stims = [s for s in self._stims if s.timestamp < hi]
            self._stims = [s for s in self._stims if s.timestamp >= hi]
            self._t = hi  # the body runs once the tick's data is in: neurons.timestamp() == iteration_timestamp
            yield _Tick(loop, k, lo, hi, sp, stims, frames)
            k += 1
            if self.speed and k % 50 == 0:  # pace to wall clock: k ticks of virtual time take k*dt/FPS/speed seconds
                time.sleep(max(0.0, k * dt / FPS / self.speed - (time.perf_counter() - wall0)))


@contextmanager
def open():  # shadows builtin on purpose: the real API is cl.open()
    yield Neurons(no_cells=os.environ.get("NO_CELLS") == "1")
