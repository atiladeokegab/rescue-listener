"""Wetware scaling law: information per stimulus (bits) as the stimulation alphabet grows 2, 4, 8, 16.

Each stage stimulates one electrode per trial from a nested alphabet (ALPHABET[:N]), REPS times each, in a
random order. A cross-validated reader guesses which electrode was stimulated from 5-band power 10-410 ms
after the pulse. Information = plug-in mutual information of the held-out confusion matrix minus the mean of
20 label shuffles (small-sample bias). Controls: one fresh shuffle (the grey line) and the same reader on a
400 ms window BEFORE the pulse (must sit at ~0). Artefact trap: the same reader on a window that starts AT the
pulse, so it includes the stimulation. Bigger alphabets change the stimulation, so the artefact alone can draw a
rising curve (a saline dish scored 44% on MNIST in Loeffler); the trap shows how much and is never counted.

    uv run python bci.py                    # mock (or real CL1) culture, prints the four points
    uv run python bci.py --reps 6           # quicker, noisier
    NO_CELLS=1 uv run python bci.py         # dead-dish control: every point on the floor
    uv run python bci.py --replay           # Wagenaar 2006 rat cortex recordings on disk
Live view: uv run python live.py
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time
import warnings

import adapter as cl  # every `cl` call goes through adapter.py (real cl-sdk if installed, else cl_mock)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", message="The number of unique classes")  # early live estimates: few trials per class
BurstDesign, ChannelSet, StimDesign = cl.BurstDesign, cl.ChannelSet, cl.StimDesign

# ---- the three knobs: stim pattern, window, decoder ----------------------------
# 16 single electrodes spread over the 8x8 array (none of the excluded 0, 4, 7, 56, 63); stage N uses the first N
ALPHABET = [18, 45, 21, 42, 10, 53, 29, 34, 2, 61, 24, 39, 13, 50, 31, 32]
STIM = StimDesign(160, -2.0, 160, 2.0)   # 160 us per phase, 2 uA, negative-leading biphasic: 0.32 nC (limit 3 nC)
PULSES, PULSE_HZ = 1, 10                 # one pulse per trial (limit 200 Hz per channel)
SKIP_MS, WINDOW_MS = 10, 400             # window: decode 10-410 ms after the last pulse (Loeffler-style; CLI --skip-ms, try 50)
TREES = 100                              # decoder: random forest
STAGES = (2, 4, 8, 16)
REPS = 20                                # trials per pattern per stage (CLI --reps): 320 trials at 16, 600 in all
TRIAL_S = 1.0                            # one stimulus per second: window + decay; throughput = bits / TRIAL_S
WARMUP_S = 2
N_SHUFFLES = 20

FPS = 25_000
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
EXCLUDED = (0, 4, 7, 56, 63)            # grounded 0,7,56,63 + reference 4
USABLE = [c for c in range(64) if c not in EXCLUDED]
assert not set(ALPHABET) & set(EXCLUDED) and len(set(ALPHABET)) == 16
PATTERNS = [ChannelSet(c) for c in ALPHABET]
BURST = BurstDesign(PULSES, PULSE_HZ)
BIN_MS = 50


def source():
    if cl.MOCK:
        return "mock"
    return "cl-sdk simulator" if cl.is_simulator() else "CL1"


def features(raw):
    """5-band log power per live channel over the whole window. shortcut: one FFT per window,
    not Loeffler's 10 ms bins (100 Hz resolution makes the low bands meaningless), revisit if raw is clean."""
    raw = np.asarray(raw, dtype=float)  # (frames, 64)
    spec = np.abs(np.fft.rfft(raw[:, USABLE] - raw[:, USABLE].mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(raw.shape[0], 1 / FPS)
    return np.concatenate([np.log10(spec[(f >= lo) & (f < hi)].sum(0) + 1e-6) for lo, hi in BANDS])


def schedule(reps, seed):
    """[(stage N, pattern index)]: nested alphabets, REPS each, order randomised within each stage."""
    rng = np.random.default_rng(seed)
    return [(N, int(k)) for N in STAGES for k in rng.permutation(np.repeat(np.arange(N), reps))]


RING = 40_000                   # frames kept (1.6 s at 25 kHz; a multiple of the 25-frame tick)
WIN = WINDOW_MS * FPS // 1000


def write_tick(ring, tick):
    """tick.frames (uV, the same span as tick.analysis) into our ring buffer."""
    f = tick.frames
    ring[(tick.timestamp + np.arange(len(f))) % RING] = f


def relock(busy, tick, channels, keys):
    """Move a trial's window timestamps onto the real pulse time once its echo shows in tick.analysis.stims."""
    if busy.get("locked"):
        return
    t = cl.stim_time(tick, channels)
    if t is not None:
        d = t - busy[keys[0]]
        for k in keys:
            busy[k] += d
        busy["locked"] = True


def ring_window(ring, start, n, now):
    assert now - start <= RING, "ring too short"
    return ring[(start + np.arange(n)) % RING]


def _new_out():
    return {"X": [], "pre": [], "trap": [], "y": [], "stage": []}


def run_session(neurons, reps=REPS, seed=0, on_bin=None, on_fire=None, on_trial=None):
    """One continuous loop through all four stages. The tick body only reads, buffers and stims;
    window cutting, FFT and the decode run in a worker thread fed by a queue."""
    plan_ = schedule(reps, seed)
    pos = [sum(1 for n, _ in plan_[:i + 1] if n == N) for i, (N, _) in enumerate(plan_)]  # trial number in its stage
    ring = np.zeros((RING, 64), dtype=np.float32)   # our own buffer: nothing relies on read() reaching into the past
    st = dict(chbin=np.zeros(64), nt=0, i=0, busy=None, submitted=0,
              next_fire=neurons.timestamp() + WARMUP_S * FPS)
    out = _new_out()
    q = queue.Queue()

    def worker():
        while (job := q.get()) is not None:
            b = job["busy"]
            for key, v in dict(X=features(job["post"]), pre=features(job["pre"]), trap=features(job["trap"]),
                                 y=b["k"], stage=b["N"]).items():
                out[key].append(v)
            if on_trial:
                on_trial(len(out["y"]), out, b)

    def window(start, n, now):
        return ring_window(ring, start, n, now)

    def step(tick):
        ts = tick.timestamp
        write_tick(ring, tick)
        for sp in tick.analysis.spikes:
            st["chbin"][sp.channel] += 1
        st["nt"] += 1
        if st["nt"] % BIN_MS == 0:  # 1000 ticks/s
            if on_bin:
                on_bin(st["chbin"].copy())
            st["chbin"] = np.zeros(64)
        busy = st["busy"]
        if busy:
            relock(busy, tick, (busy["electrode"],), ("t0", "t_win", "t_end"))
            if ts >= busy["t_end"]:
                q.put(dict(busy=busy, post=window(busy["t_win"], WIN, ts).copy(),
                           pre=window(busy["t0"] - WIN, WIN, ts).copy(), trap=window(busy["t0"], WIN, ts).copy()))
                st["busy"], st["submitted"] = None, st["submitted"] + 1
                return st["submitted"] >= len(plan_)
            return False
        if ts < st["next_fire"]:
            return False
        N, k = plan_[st["i"]]
        plan = neurons.create_stim_plan()  # one stim plan per trial
        plan.stim(PATTERNS[k], STIM, BURST)
        t0 = plan.run()
        t_win = t0 + int((PULSES - 1) * FPS / PULSE_HZ) + SKIP_MS * FPS // 1000
        last = st["i"] + 1 == len(plan_) or plan_[st["i"] + 1][0] != N
        st["busy"] = dict(N=N, k=k, electrode=ALPHABET[k], t0=t0, t_win=t_win, t_end=t_win + WIN + FPS // 50,
                          stage_trial=pos[st["i"]], stage_total=N * reps,
                          last_in_stage=last)
        st["i"] += 1
        st["next_fire"] = t0 + int(TRIAL_S * FPS)
        if on_fire:
            on_fire(ALPHABET[k])
        return False

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    body = []
    for tick in neurons.loop(ticks_per_second=1000):
        t, c = time.perf_counter(), time.thread_time()
        done = step(tick)
        body.append((time.perf_counter() - t, time.thread_time() - c))  # wall, and CPU (blind to preemption)
        if done:
            break
    q.put(None)
    th.join()
    res = {k: np.array(v) for k, v in out.items()}
    res["tick_s"], res["tick_cpu_s"] = np.array(body).reshape(-1, 2).T
    return res


# ---- replay: Wagenaar, Pine, Potter 2006 (rat cortex, dense, stimulus-evoked) -----------------
REPLAY_DIR = os.environ.get("REPLAY_DIR", "data/replay")
REPLAY_TRIAL_S = 0.301               # their fixed inter-stimulus interval
REPLAY_POST_MS = (10, 290)           # the next stimulus arrives at 301 ms, so 10-410 does not fit
REPLAY_PRE_MS = (-280, 0)
REPLAY_TRAP_MS = (0, 290)            # starts at the pulse: includes the stimulation


def _spike_counts(t_spk, ch_spk, T, lo, hi):
    X = np.zeros((len(T), 60), np.float32)
    a, b = np.searchsorted(t_spk, T + lo / 1e3), np.searchsorted(t_spk, T + hi / 1e3)
    for i, (x, y) in enumerate(zip(a, b)):
        np.add.at(X[i], ch_spk[x:y], 1)
    return X


def replay_session(name=None, reps=REPS, speed=0.0, on_trial=None):
    """Same output as run_session, from one recording. Files are untrusted data: read with numpy only.
    shortcut: features are spike counts per channel (the files hold spikes, not raw voltage)."""
    names = sorted(os.path.basename(f).split(".")[0] for f in glob.glob(os.path.join(REPLAY_DIR, "*.tri.txt.bz2")))
    name = name or names[0]
    if name not in names:
        raise SystemExit(f"no recording {name!r} in {REPLAY_DIR}; have {', '.join(names[:8])}...")
    tri = np.loadtxt(os.path.join(REPLAY_DIR, name + ".tri.txt.bz2"), ndmin=2)
    spk = np.loadtxt(os.path.join(REPLAY_DIR, name + ".spk.txt.bz2"), usecols=(0, 1), ndmin=2)
    tri = tri[tri[:, 1] >= 0]
    spk = spk[(spk[:, 1] >= 0) & (spk[:, 1] < 60)]
    spk = spk[np.argsort(spk[:, 0], kind="stable")]
    t_spk, ch_spk = spk[:, 0], spk[:, 1].astype(int)
    T, cls = tri[:, 0], tri[:, 1].astype(int)
    present = np.unique(cls)
    alphabet = present[np.linspace(0, len(present) - 1, 16).round().astype(int)]  # 16 electrodes spread over theirs
    plan, out = [], _new_out()
    reps = min(reps, min(int((cls == e).sum()) for e in alphabet) // len(STAGES))  # about 50 per electrode: 12 per stage
    for s, N in enumerate(STAGES):  # each stage takes its own block of repetitions, so stages share no trials
        idx = np.concatenate([np.nonzero(cls == e)[0][s * reps:(s + 1) * reps] for e in alphabet[:N]])
        plan += [(N, int(i)) for i in np.sort(idx)]  # their order: fixed cycle, not randomised
    post = _spike_counts(t_spk, ch_spk, T, max(SKIP_MS, REPLAY_POST_MS[0]), REPLAY_POST_MS[1])
    trap = _spike_counts(t_spk, ch_spk, T, *REPLAY_TRAP_MS)
    pre = _spike_counts(t_spk, ch_spk, T, *REPLAY_PRE_MS)
    k_of = {int(e): j for j, e in enumerate(alphabet)}
    for j, (N, i) in enumerate(plan):
        out["X"].append(post[i]); out["pre"].append(pre[i]); out["trap"].append(trap[i]); out["y"].append(k_of[cls[i]]); out["stage"].append(N)
        if on_trial:
            on_trial(len(out["y"]), out, dict(N=N, k=k_of[cls[i]], electrode=int(cls[i]), counts=post[i], alphabet=alphabet,
                                              stage_trial=sum(1 for n, _ in plan[:j + 1] if n == N),
                                              stage_total=sum(1 for n, _ in plan if n == N),
                                              last_in_stage=j + 1 == len(plan) or plan[j + 1][0] != N))
        if speed:
            time.sleep(REPLAY_TRIAL_S / speed)
    res = {k: np.array(v) for k, v in out.items()}
    res["alphabet"], res["name"] = alphabet, name
    return res


# ---- the reader ---------------------------------------------------------------------------
def mi_bits(y, p, n_cls):
    """Plug-in mutual information (bits) of a confusion matrix."""
    C = np.zeros((n_cls, n_cls))
    np.add.at(C, (y, p), 1)
    P = C / C.sum()
    pt, pp = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pt @ pp)[nz])).sum())


def info(X, y, n_cls, trees=TREES, seed=0, n_shuffles=N_SHUFFLES):
    """Held-out information in bits, bias-corrected, plus one fresh-shuffle control. None if too few trials.
    Folds are stratified WITHOUT shuffling: each class's trials are split into contiguous blocks in time,
    so a test trial is never scored by a reader trained on its own time-neighbours."""
    counts = np.bincount(y, minlength=n_cls)
    if counts.min() < 2:
        return None
    cv = StratifiedKFold(int(min(4, counts.min())))
    p = cross_val_predict(RandomForestClassifier(trees, random_state=seed, n_jobs=1), X, y, cv=cv)
    rng = np.random.default_rng(seed)
    bias = float(np.mean([mi_bits(rng.permutation(y), p, n_cls) for _ in range(n_shuffles)]))
    return dict(live=mi_bits(y, p, n_cls) - bias, ctrl=mi_bits(rng.permutation(y), p, n_cls) - bias,
                plugin=mi_bits(y, p, n_cls), bias=bias, acc=float((p == y).mean()), n=int(len(y)))


def evaluate_stage(d, N, trees=TREES, seed=0):
    m = d["stage"] == N
    if not m.any():
        return None
    r = info(d["X"][m], d["y"][m], N, trees, seed)
    if r:
        pre = info(d["pre"][m], d["y"][m], N, trees, seed)
        r["pre"] = pre["live"] if pre else None
        trap = info(d["trap"][m], d["y"][m], N, trees, seed)
        r["trap"] = trap["live"] if trap else None
    return r


def scaling(d, trees=TREES, seed=0):
    return {N: evaluate_stage(d, N, trees, seed) for N in STAGES}


def fmt(res, trial_s=TRIAL_S):
    rows = ["  N   bits  ceiling  shuffled  pre-stim  acc    bits/s  trials   trap (includes stimulation: not counted)"]
    for N, r in res.items():
        if r:
            rows.append(f"{N:3d}  {r['live']:5.2f}  {np.log2(N):7.2f}  {r['ctrl']:8.2f}  {r['pre']:8.2f}  "
                        f"{r['acc']:4.0%}  {r['live'] / trial_s:7.2f}  {r['n']:6d}   {r['trap']:5.2f}")
    return "\n".join(rows)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=REPS, help="trials per pattern per stage (default 12)")
    ap.add_argument("--replay", nargs="?", const="", default=None, metavar="NAME",
                    help="replay a Wagenaar 2006 recording (default: the first, e.g. 1-1-20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ms", type=int, default=SKIP_MS, help="readout starts this long after the pulse (>= 10)")
    ap.add_argument("--rhythm", action="store_true", help="Rescue Listener: knock-rhythm mode (rhythm.py)")
    ap.add_argument("--short", action="store_true", help="probe experiment: 20 blocks instead of 30 (about 4 min instead of 6)")
    ap.add_argument("--cal", type=int, default=40, help="knock calibration sequences per class (default 40, minimum 20)")
    ap.add_argument("--port", type=int, default=8777, help="live.py's screen port (default 8777)")
    a = ap.parse_args(argv)
    if a.cal < 20:
        ap.error("--cal below 20 per class leaves the reader too few sequences to train and score")
    if a.skip_ms < 10:
        ap.error("--skip-ms below 10 reads the stimulation artefact")
    if a.rhythm and a.replay is not None:
        ap.error("--rhythm needs a dish that can be stimulated: no --replay")
    globals()["SKIP_MS"] = a.skip_ms
    return a


if __name__ == "__main__":
    a = parse_args()
    if a.rhythm:
        import rhythm
        rhythm.bci.SKIP_MS = SKIP_MS  # this file runs as __main__; rhythm imports its own copy of bci
        rhythm.main(a)
        sys.exit()
    if a.replay is not None:
        d = replay_session(a.replay or None, a.reps)
        print(f"REPLAY rat cortex, Wagenaar 2006, recording {d['name']} (fixed stimulus order: order can leak)")
        print(fmt(scaling(d, seed=a.seed), REPLAY_TRIAL_S))
        sys.exit()
    with cl.open() as neurons:
        d = run_session(neurons, a.reps, a.seed)
    print(f"source: {source()}{' (simulator check: cells off)' if os.environ.get('NO_CELLS') == '1' else ''}")
    print(fmt(scaling(d, seed=a.seed)))
