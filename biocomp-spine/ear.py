"""Rescue Listener, ear mode: the dish hears the sound itself, not the knock times.

Audio in 50 ms frames. Each frame's power in 16 log-spaced bands (100 Hz - 7 kHz); a band "sounds" when it is
ON_DB over its median for the clip, and the (at most) 3 loudest sounding bands each pulse one electrode. Bands sit
tonotopically along a strip of 16 electrodes (rows 3-4 of the 8x8 grid, a snake, low to high), so neighbouring
pitches drive neighbouring electrodes. The reader, the per-pulse windows and the controls are knock mode's
(rhythm.py): shuffled labels, a no-cells dish, and software on the same pulse timings.

Input: the WAV trials in ../rescue-data beside the spine (RESCUE_DATA overrides) (sequences.json: survivor patterns SOS, 3-knock motif, 2-2 pairs and
five_tap_reply, each matched by debris on knock count and span; both classes use the same knocks, so only timing
differs), mixed over a synthetic bed named per trial (`bed`, e.g. drone rotor on or off) at --snr 10, 0 or -10 dB.
Calibration takes N_CAL of each class drawn at random over the whole file (seeded), so every pattern and bed shows.

    uv run python ear.py [--snr 10] [--seed 0]   # calibrate on the mock (or real CL1) and print the tally
    NO_CELLS=1 uv run python ear.py             # dead dish: chance
Live: uv run python live.py --rhythm, then the Ear switch.
"""
import json
import os
import wave

import numpy as np

import bci
import rhythm
from bci import cl

DATA = os.path.realpath(os.path.expanduser(os.environ.get(  # the kit puts the data beside the spine
    "RESCUE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rescue-data"))))
FRAME_S, N_BANDS, MAX_PULSES, ON_DB = 0.05, 16, 3, 6.0
KNOCK_BANDS, KNOCK_DB = 10, 9.0  # a frame sounds when bands 0-9 (100-1400 Hz, where a knock's energy is) average 9 dB over median
EDGES = np.geomspace(100, 7000, N_BANDS + 1)
# rows 3 and 4 as a snake: 24..31 left to right, then 39..32, so band b and b+1 are always grid neighbours
BAND_CH = list(range(24, 32)) + list(range(39, 31, -1))
READ = [c for c in bci.USABLE if c not in BAND_CH and c // 8 in (1, 2, 5, 6)]  # the rows either side of the strip
READ_SET = frozenset(READ)
assert not set(BAND_CH) & set(bci.EXCLUDED) and len(set(BAND_CH)) == N_BANDS
N_CAL = 40


def trials(snr="10"):
    """[(id, label 1=survivor, wav path, bed)] in file order. The data repo is read, never written."""
    meta = json.load(open(os.path.join(DATA, "sequences.json")))
    key = f"snr{snr}" if str(snr).startswith("-") or str(snr) == "0" else f"snr+{snr}"
    return [(t["id"], int(t["class"] == "survivor"), os.path.join(DATA, t["files"][key]), t.get("bed", "?"))
            for t in meta["trials"]]


def index():
    """{id: the trial's sequences.json entry} (pattern, bed, knock_times_ms, ...)."""
    return {t["id"]: t for t in json.load(open(os.path.join(DATA, "sequences.json")))["trials"]}


def load_wav(path):
    with wave.open(path) as w:  # untrusted file: 16-bit mono only, read with numpy
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: want 16-bit mono")
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768, w.getframerate()


def band_db(x, sr):
    """(frames, 16) band power in dB, 50 ms frames, Hann window."""
    n = int(FRAME_S * sr)
    fr = x[: len(x) // n * n].reshape(-1, n) * np.hanning(n)
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    return 10 * np.log10(np.stack([p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], 1) + 1e-12)


def events(x, sr):
    """[(time s, ChannelSet)]: in each frame that sounds, its <= 3 loudest bands that are ON_DB over their clip median.

    Without the frame gate every bed impact (debris, grit, rotor) pulses too: software on those pulse times scored
    49-60% against 99% on the true knock times. With it: 67% at +10 dB, chance at 0 and -10 dB (400 trials, 5-fold).
    """
    db = band_db(x, sr)
    over = db - np.median(db, axis=0)
    knock = over[:, :KNOCK_BANDS].mean(1)
    sound = set(np.nonzero(knock >= KNOCK_DB)[0])
    if len(sound) < 2:  # a near-silent clip still plays its 2 loudest frames
        sound = set(np.argsort(-knock)[:2])
    ev = []
    for i, row in enumerate(over):
        top = [int(b) for b in np.argsort(-row)[:MAX_PULSES] if row[b] >= ON_DB] or [int(row.argmax())]
        if i in sound:
            ev.append((i * FRAME_S, bci.ChannelSet(*[BAND_CH[b] for b in top])))
    return ev


def calibration(snr="10", n=N_CAL):
    """n trials of each class, a fixed random draw over the whole file (later trials hold other patterns and beds)."""
    tr = trials(snr)
    picked, k = [], {0: 0, 1: 0}
    for i in np.random.default_rng(0).permutation(len(tr)):
        if k[tr[i][1]] < n:
            picked.append(tr[i])
            k[tr[i][1]] += 1
    return picked


def calibrate(neurons, snr="10", seed=0, n=N_CAL):
    cal = calibration(snr, n)
    seqs = [events(*load_wav(t[2])) for t in cal]
    y = np.array([t[1] for t in cal])
    T = np.array([rhythm.timing_features(np.array([t for t, _ in e])) for e in seqs])
    F, TR, body, recs = rhythm.run_rhythm(neurons, seqs, read_set=READ_SET)
    keep = [r["meta"]["i"] for r in recs]  # dropped clips (skipped ticks, late pulses) leave the tally
    r = rhythm.evaluate(F, T[keep], y[keep], seed, trap=TR)
    r.update(tick_s=body[:, 0], tick_cpu_s=body[:, 1], pulses=float(np.mean([sum(len(list(c)) for _, c in e) for e in seqs])))
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snr", default="10", choices=["10", "0", "-10"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with cl.open() as neurons:
        r = calibrate(neurons, a.snr, a.seed)
    print(f"source: {bci.source()}{' (simulator check: cells off)' if os.environ.get('NO_CELLS') == '1' else ''}  ear mode, snr {a.snr} dB, "
          f"{r['pulses']:.0f} pulses per clip")
    print(rhythm.fmt(r).replace("timing-only software", "software on the same pulse timings"))
