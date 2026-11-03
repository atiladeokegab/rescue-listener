"""Rescue Listener: the dish tells a survivor's knock rhythm from falling debris, using its short-term memory.

Each knock becomes one pulse (the same safe pulse as the scaling law, on one input electrode) at the knocked
timing. Survivor = a human rhythm (SOS: 3 fast, 3 slow, 3 fast; any repeated 3-knock motif; or
five_tap_reply, 5 even taps 0.4-0.7 s apart answering "tap five times"). Debris = the
same number of knocks at random (Poisson-like) times over the same span. Calibration plays 40 of each.

Reader: per-pulse response (spikes 10-50 ms after each pulse on the electrodes within 2 of the input) summarised
over the sequence (stats, lag 1-3 autocorrelation, first 12 in order),
plus 5-band power 10-410 ms after the last pulse; random forest, folds by sequence in time order.
Windows are locked to the pulse's echo in tick.analysis.stims, not to when the stim was asked for.
Controls: shuffled labels, a no-cells dish (NO_CELLS=1), and the timing-only baseline: the same reader on the
knock timings alone. Software reads rhythm from timing easily; the baseline is shown so that question is asked.
Artefact trap (not counted): the knock times recovered from the input electrode's own trace, which includes the
stimulation: it reads the knocks themselves, so it scores like software.

    uv run python rhythm.py [--seed 0] [--skip-ms 10]   # calibrate on the mock (or real CL1), print the tally
    NO_CELLS=1 uv run python rhythm.py                  # dead dish: chance
    (same as: uv run python bci.py --rhythm)
Live: uv run python live.py --rhythm
"""
import math
import queue
import threading
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import bci
from bci import FPS, cl

N_READ = 24


def set_input(ch, read=None):
    """Input electrode and readout electrodes (default: the 24 within 2 of the input; scan() picks by response)."""
    global IN_CH, INPUT, READ, READ_SET
    IN_CH, INPUT = int(ch), bci.ChannelSet(int(ch))
    READ = [int(c) for c in read] if read is not None else [
        c for c in bci.USABLE if max(abs(c // 8 - ch // 8), abs(c % 8 - ch % 8)) in (1, 2)]
    READ_SET = frozenset(READ)
    assert IN_CH in bci.USABLE and not READ_SET & set(bci.EXCLUDED) and IN_CH not in READ_SET


set_input(27)  # near the array centre until a scan picks one
DRIVE_ORDER = None  # usable electrodes, strongest driver first, from the latest scan (noisefilter.py takes the top two)
BURST = bci.BurstDesign(1, 10)        # one pulse per knock
PER_PULSE_MS = 40                     # per-pulse window: SKIP_MS .. SKIP_MS + 40 ms after each pulse
MAX_KNOCKS, MAX_SPAN_S = 20, 4.0      # judge input is clipped to this
REST_S = 1.0                          # quiet between sequences so the short-term memory resets
N_CAL = 40                            # calibration sequences per class
TREES = 100
LABELS = ("debris", "survivor")




def min_gap_s():
    """Shortest knock gap: each pulse's window (SKIP_MS + 40 ms) closes before the next pulse, plus 2 ms for tick
    scheduling, so every pulse gets the same window. 60 ms at the default 10 ms skip; 92 ms at --skip-ms 50."""
    return max(0.06, (bci.SKIP_MS + PER_PULSE_MS + 2) / 1000)


# ---- knock sequences (offsets in seconds from the first knock) -----------------------------
FIVE_TAP = True  # the reply to a rescuer's "tap five times"; False gives the old SOS/motif-only draws exactly


def five_tap(rng):
    """five_tap_reply: 5 roughly even taps, 0.4-0.7 s apart, 8% human jitter on each gap."""
    g = rng.uniform(0.4, 0.7) * rng.normal(1, 0.08, 4)
    return np.concatenate([[0], np.cumsum(np.maximum(g, min_gap_s()))])


def survivor(rng):
    tempo = rng.uniform(0.85, 1.2)
    r = rng.random()
    if FIVE_TAP and r < 1 / 3:
        return five_tap(rng)
    if r < (2 / 3 if FIVE_TAP else 0.5):  # SOS: 3 fast, 3 slow, 3 fast
        gaps = [0.12, 0.12, 0.36, 0.3, 0.3, 0.36, 0.12, 0.12]
    else:                    # any repeated motif: 3 knocks, played 3 times
        a, b, pause = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), rng.uniform(0.3, 0.45)
        gaps = [a, b, pause, a, b, pause, a, b]
    g = np.maximum(np.array(gaps) * tempo * rng.normal(1, 0.06, len(gaps)), min_gap_s())
    return np.concatenate([[0], np.cumsum(g)])


def debris(rng, n, span):
    while True:  # uniform times over the span = Poisson conditioned on the count
        t = np.sort(np.concatenate([[0, span], rng.uniform(0, span, n - 2)]))
        if np.diff(t).min() >= min_gap_s():
            return t


def calibration(seed=0, n=N_CAL):
    rng = np.random.default_rng(seed)
    seqs, y = [], []
    for c in rng.permutation(np.repeat([0, 1], n)):  # interleaved, so drift can't separate the classes
        s = survivor(rng)
        seqs.append(s if c else debris(rng, len(s), s[-1]))
        y.append(int(c))
    return seqs, np.array(y)


def cal_duration_s(seqs):
    """Planned chip time for a calibration, as run_rhythm paces it: warm-up, then per sequence its span, the last
    pulse's read window (+20 ms) and the rest. Excludes dropped-sequence restarts and worker lag."""
    return bci.WARMUP_S + sum(float(s[-1]) + (bci.SKIP_MS + bci.WINDOW_MS + 20) / 1000 + REST_S for s in seqs)


def clean(knocks_ms):
    """Judge input (untrusted): numbers in ms -> sorted offsets in s, too-close knocks merged, clipped."""
    t = np.sort(np.asarray([float(k) for k in knocks_ms if math.isfinite(float(k))]) / 1000.0)
    if len(t) == 0:
        return t
    t = t - t[0]
    keep = [0.0]
    for v in t[1:]:
        if v - keep[-1] >= min_gap_s() and v <= MAX_SPAN_S:
            keep.append(float(v))
    return np.array(keep[:MAX_KNOCKS])


# ---- features --------------------------------------------------------------------------------
ORDERED = 12


def _stats(v):
    """Summary stats, autocorrelation at lags 1-3 (rhythm = periodicity), and the first 12 values in order,
    centred. shortcut: the ordered part assumes about 9-12 knocks like calibration; longer input is cut."""
    v = np.asarray(v, float)
    d = np.diff(v) if len(v) > 1 else np.zeros(1)
    c = v - v.mean()
    den = (c * c).sum() + 1e-12
    ac = [(c[:-k] * c[k:]).sum() / den if len(v) > k else 0.0 for k in (1, 2, 3)]
    o = np.zeros(ORDERED)
    o[:min(ORDERED, len(v))] = c[:ORDERED]
    return [v.mean(), v.std(), v.min(), v.max(), *np.percentile(v, [25, 50, 75]), np.abs(d).mean(), d.std(), *ac, *o]


def dish_features(counts, post, read_set=None):
    """Summary of the per-pulse responses (spikes 10-50 ms after each pulse on the read set), plus the response
    after the last pulse (5-band power, mean over the read set: never a stimulated electrode, whose artefact and
    polarisation could carry the timing without any neuron)."""
    read_set = READ_SET if read_set is None else read_set
    a = np.log1p(np.asarray(counts, float))
    cols = [i for i, c in enumerate(bci.USABLE) if c in read_set]
    return np.array(_stats(a) + list(bci.features(post).reshape(len(bci.BANDS), -1)[:, cols].mean(1)))


def band_power(raw, read_set=None):
    """5-band log power per read electrode (bands x electrodes, flattened) of a raw window: bci.features' one FFT."""
    read_set = READ_SET if read_set is None else read_set
    return bci.features(raw).reshape(len(bci.BANDS), -1)[:, [i for i, c in enumerate(bci.USABLE) if c in read_set]].ravel()


def trap_features(env):
    """The trap: knock times read back from the input electrode's trace (1 value per ms, includes the
    stimulation), then the timing reader. Shown, never counted."""
    env = np.asarray(env)
    hit = np.nonzero(env > 0.5 * env.max() + 1e-9)[0]  # the pulse artefact towers over the neurons
    t = hit[np.concatenate([[True], np.diff(hit) > 5])] / 1000.0 if len(hit) else np.zeros(1)
    return timing_features(t if len(t) > 1 else np.zeros(2))


def timing_features(t):
    """The trap/baseline: the knock timings alone, no dish."""
    return np.array(_stats(np.diff(t)) + [len(t), t[-1]])


# ---- the closed loop ---------------------------------------------------------------------------
def run_rhythm(neurons, seqs, on_seq=None, more=None, stop=None, read_set=None, on_drop=None, win_ms=None):
    """Play each sequence into the dish at its timing. A sequence is knock times in s (each a pulse on INPUT) or
    a list of (time s, channels) events. The tick body only buffers, stims and counts spikes in each pulse's 40 ms
    window; features run in a worker thread. After `seqs`, keep pulling from more() (judge knocks) until stop
    is set; without more(), return when the list is done.

    A sequence is dropped, never fired late, when the loop skips ticks under it (jitter recovery leaves stale ring
    data), a pulse comes due over 2 ms late (catching up would bunch pulses past 200 Hz), or a pulse would land
    before the previous pulse's window closed (a truncated window). Dropped sequences go to on_drop(meta, reason).
    Returns (F, TR, body, recs): features and trap rows of the kept sequences, tick body times (wall, CPU), and per
    kept sequence dict(meta, probe, pre, base_hz, probe_bp, pre_bp) where probe = spikes per read electrode in the
    last pulse's window, pre = the same-length window just before the first pulse, base_hz = rate in the 500 ms
    before it, probe_bp / pre_bp = band_power of the raw window SKIP_MS to SKIP_MS + WINDOW_MS after the last pulse
    and of the WINDOW_MS just before the first."""
    ring = np.zeros((bci.RING, 64), dtype=np.float32)
    ms = FPS // 1000
    per, skip, win = PER_PULSE_MS * ms, bci.SKIP_MS * ms, (win_ms * ms if win_ms else bci.WIN)  # win: the post window
    read_set = READ_SET if read_set is None else read_set
    read = sorted(read_set)
    spk = np.zeros((1024, 64), dtype=np.int32)  # spikes per ms per electrode, the last ~1 s: the pre-trial windows
    todo = [(s, dict(kind="cal", i=i)) for i, s in enumerate(seqs)]
    st = dict(seq=None, next_start=neurons.timestamp() + bci.WARMUP_S * FPS, done=0, prev=None)
    out = []
    q = queue.Queue()

    def worker():
        while (job := q.get()) is not None:
            C = np.array(job["counts"])[:, read]
            f, trap = dish_features(C.sum(1), job["post"], read_set), trap_features(job["env"])
            out.append((f, trap, dict(meta=job["meta"], probe=C[-1], pre=job["pre"], base_hz=job["base_hz"],
                                      probe_bp=band_power(job["post"], read_set), pre_bp=band_power(job["pre_raw"], read_set))))
            job["meta"]["pulse_counts"] = C.sum(1).tolist()  # response size per pulse: the screen's history chart
            job["meta"]["gaps_ms"] = (np.diff(job["t"]) * 1000).tolist()
            if on_seq:
                on_seq(f, timing_features(job["t"]), trap, job["meta"])

    def drop(ts, why):
        s = st["seq"]
        st["seq"], st["next_start"] = None, max(st["next_start"], ts + int(REST_S * FPS))
        if s is not None and on_drop:
            on_drop(s["meta"], why)

    def window(r1, n):  # spikes per read electrode in the n ms before row r1
        return spk[np.arange(r1 - n, r1) % len(spk)][:, read].sum(0)

    def step(tick):
        ts = tick.timestamp
        bci.write_tick(ring, tick)
        row = ts // ms
        spk[row % len(spk)] = 0
        for sp in tick.analysis.spikes:
            spk[(sp.timestamp // ms) % len(spk), sp.channel] += 1
        gap = st["prev"] is not None and ts - st["prev"] > 2 * ms
        st["prev"] = ts
        if gap:  # ticks were skipped (jitter recovery): what is in flight is incomplete; rest before the next
            st["next_start"] = max(st["next_start"], ts + int(REST_S * FPS))
            if st["seq"] is not None:
                drop(ts, "ticks skipped")
                return False
        s = st["seq"]
        if s is None:
            if ts < st["next_start"]:
                return False
            nxt = todo.pop(0) if todo else (more() if more else None)
            if nxt is None:
                return not more or (stop is not None and stop.is_set())
            t, meta = nxt
            ev = [(v, INPUT) for v in t] if isinstance(t, np.ndarray) else list(t)
            chs = sorted({c for _, cs in ev for c in cs})
            st["seq"] = s = dict(t=np.array([v for v, _ in ev]), meta=meta, ev=ev, due=[ts + int(v * FPS) for v, _ in ev],
                                 k=0, counts=[], env=[], chs=chs, cur=None,
                                 pre=window(row, PER_PULSE_MS), base_hz=window(row, 500) / 0.5,
                                 pre_raw=bci.ring_window(ring, ts - win, win, ts).copy())  # stim-free: REST_S since the last window
        if s["k"] < len(s["due"]) and ts >= s["due"][s["k"]]:
            if ts - s["due"][s["k"]] > 2 * ms:
                drop(ts, "pulse overdue")
                return False
            if s["cur"] and ts + ms < s["cur"]["p"] + skip + per:  # 1 ms slack: the echo can trail the request
                drop(ts, "pulse inside the previous window")
                return False
            cs = s["ev"][s["k"]][1]
            plan = neurons.create_stim_plan()
            plan.stim(cs, bci.STIM, BURST)
            s["cur"] = dict(p=plan.run(), chs=tuple(cs))
            s["counts"].append(np.zeros(64, dtype=np.int32))
            s["k"] += 1
        cur = s["cur"]
        if cur:
            bci.relock(cur, tick, cur["chs"], ("p",))  # the window starts from the real pulse, not the request
            p = cur["p"]  # spikes belong to the latest pulse: its window closes before the next pulse (min_gap_s)
            for sp in tick.analysis.spikes:
                if sp.channel in read_set and skip <= sp.timestamp - p < skip + per:
                    s["counts"][-1][sp.channel] += 1
            i0 = ts % bci.RING  # the trap's trace: the stimulated electrodes, stimulation included
            s["env"].append(float(np.abs(ring[i0:i0 + ms][:, s["chs"]]).max()))
            if s["k"] == len(s["due"]) and ts >= p + skip + win + FPS // 50:
                q.put(dict(counts=s["counts"], env=s["env"], t=s["t"], meta=s["meta"], pre=s["pre"], base_hz=s["base_hz"], pre_raw=s["pre_raw"],
                           post=bci.ring_window(ring, p + skip, win, ts).copy()))
                st["seq"], st["next_start"], st["done"] = None, ts + int(REST_S * FPS), st["done"] + 1
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
    return (np.array([f for f, _, _ in out]), np.array([t for _, t, _ in out]), np.array(body).reshape(-1, 2),
            [r for _, _, r in out])


def scan(neurons, seconds=30.0, on_progress=None, win_ms=None):
    """Response scan: one pulse on every usable electrode in turn (twice, ~0.25 s apart), spikes counted on every
    electrode 10-50 ms after, minus the median (spontaneous firing). Input = the electrode that drives the most; readout = its
    24 strongest responders. Returns dict(input, read, drive)."""
    order = bci.USABLE * 2
    period = int(seconds / len(order) * FPS)
    w0, w1 = win_ms or (bci.SKIP_MS, bci.SKIP_MS + PER_PULSE_MS)  # the probe experiment selects on 40-250 ms
    skip, per = w0 * FPS // 1000, (w1 - w0) * FPS // 1000
    assert skip + per <= period, "the response window runs into the next scan pulse"
    resp = np.zeros((64, 64))
    t0 = neurons.timestamp() + FPS // 2
    fired = []  # dict(p=pulse timestamp, c=electrode)
    for tick in neurons.loop(ticks_per_second=1000):
        ts = tick.timestamp
        if len(fired) < len(order) and ts >= t0 + len(fired) * period:
            c = order[len(fired)]
            plan = neurons.create_stim_plan()
            plan.stim(bci.ChannelSet(c), bci.STIM, BURST)
            fired.append(dict(p=plan.run(), c=c))
            if on_progress:
                on_progress(len(fired), len(order))
        if fired:
            f = fired[-1]
            bci.relock(f, tick, (f["c"],), ("p",))
            for sp in tick.analysis.spikes:
                if skip <= sp.timestamp - f["p"] < skip + per:
                    resp[f["c"], sp.channel] += 1
        if len(fired) == len(order) and ts >= fired[-1]["p"] + skip + per:
            break
    np.fill_diagonal(resp, 0)  # the input's own trace holds its artefact
    resp[:, list(bci.EXCLUDED)] = 0
    resp -= np.median(resp[bci.USABLE][:, bci.USABLE])  # spontaneous firing sets the floor
    drive = resp.clip(0).sum(1)
    ch = int(np.argmax(drive))
    read = [int(c) for c in np.argsort(-resp[ch]) if c != ch and c not in bci.EXCLUDED][:N_READ]
    global DRIVE_ORDER
    DRIVE_ORDER = [int(c) for c in np.argsort(-drive, kind="stable") if c in bci.USABLE]
    return dict(input=ch, read=sorted(read), ranked=read, drive=float(drive[ch]))  # ranked: strongest first


# ---- the reader --------------------------------------------------------------------------------
def p_chance(correct, n):
    """One-sided binomial p of scoring >= correct out of n by coin flips."""
    return sum(math.comb(n, k) for k in range(correct, n + 1)) / 2 ** n


def evaluate(F, T, y, seed=0, n_shuffles=20, trap=None):
    """Held-out accuracy of the dish, shuffled-label control, timing-only baseline, the artefact trap (not
    counted); plus models fit on all."""
    cv = StratifiedKFold(5)  # no shuffle: folds are blocks of sequences in time order
    rf = lambda n=TREES: RandomForestClassifier(n, random_state=seed, n_jobs=1)
    acc = lambda X: float((cross_val_predict(rf(), X, y, cv=cv) == y).mean())
    prob = lambda X: cross_val_predict(rf(), X, y, cv=cv, method="predict_proba")[:, 1]
    pr, pt = prob(F), prob(T)
    p = (pr > 0.5).astype(int)  # = the forest's predict: argmax, a tie goes to class 0
    rng = np.random.default_rng(seed)
    shuf = [float((cross_val_predict(rf(30), F, ys, cv=cv) == ys).mean()) for ys in (rng.permutation(y) for _ in range(n_shuffles))]
    correct = int((p == y).sum())
    return dict(acc=correct / len(y), correct=correct, n=int(len(y)), p=p_chance(correct, len(y)),
                shuffled=float(np.mean(shuf)), shuffled_sd=float(np.std(shuf)), timing_only=float(((pt > 0.5) == y).mean()),
                trap=acc(trap) if trap is not None else None,
                # pile flag thresholds: the 95th percentile of held-out P(target) on the irregular sequences
                threshold=float(np.percentile(pr[y == 0], 95)), software_threshold=float(np.percentile(pt[y == 0], 95)),
                model=rf().fit(F, y), timing_model=rf().fit(T, y))


def verdict(model, f):
    pr = model.predict_proba(np.asarray(f)[None])[0]
    k = int(pr.argmax())
    return dict(verdict=LABELS[k], confidence=float(pr[k]))


def fmt(r):
    return (f"calibration CV, a diagnostic: dish {r['correct']}/{r['n']} correct ({r['acc']:.0%}, p={r['p']:.2g} vs 50%)  "
            f"shuffled {r['shuffled']:.0%}  "
            f"timing-only software {r['timing_only']:.0%}  trap (includes stimulation: not counted) {r['trap']:.0%}")


def calibrate(neurons, seed=0, n=N_CAL):
    seqs, y = calibration(seed, n)
    T = np.array([timing_features(s) for s in seqs])
    F, TR, body, recs = run_rhythm(neurons, seqs)
    keep = [r["meta"]["i"] for r in recs]  # dropped sequences (skipped ticks, late pulses) leave the tally
    r = evaluate(F, T[keep], y[keep], seed, trap=TR)
    r["dropped"] = len(seqs) - len(keep)
    r["tick_s"], r["tick_cpu_s"] = body.T
    return r


def main(a):
    with cl.open() as neurons:
        r = calibrate(neurons, seed=a.seed)
    print(f"source: {bci.source()}{' (simulator check: cells off)' if bci.os.environ.get('NO_CELLS') == '1' else ''}")
    print(fmt(r))
    print(f"tick body p99 {np.percentile(r['tick_s'], 99) * 1e3:.2f} ms wall, "
          f"{np.percentile(r['tick_cpu_s'], 99) * 1e3:.2f} ms CPU, jitter recoveries {neurons.jitter_recoveries}")


if __name__ == "__main__":
    main(bci.parse_args())
