"""Pre-registered probe-omission experiment (prep.md, revised 2 Nov 18:25 UTC and amended after the hostile
review, before any real data): does the culture's response to an identical pulse depend on the earlier stimulation
history, beyond the activity that history leaves behind?

Histories, one fixed pattern each: A = pulses at 0, 200, 400, 600 ms; B = 0, 100, 500, 600 ms (same count, span and
last-history-to-probe gap). The probe pulse comes at 800 ms, or is omitted (a silent scheduled slot). Four
conditions, A+ A- B+ B-, one of each per block in random order, trials at fixed 3 s onsets: 30 blocks (120 trials,
6 min), --short 20. Before any A/B trial, all label-blind and frozen to frozen.json: a quiet recording sets the
network-burst threshold; the scan picks the input and the readout electrodes by their response 40-250 ms after a
pulse; a jitter check on the input moves the Y window start from 20 to 40 ms if a readout electrode shows
sub-millisecond-jitter events 20-40 ms after the pulse (electrical, not neural). Spikes are the device's own
detections: nothing here adapts a threshold per trial. Every trial counts; a pulse that would go out over 2 ms late
is never fired, and that trial is flagged, never dropped or re-run.

Primary: Y = raw pooled readout spike count 20 (or 40) to 250 ms after the scheduled probe time. Per block
D = (Y[A+] - Y[A-]) - (Y[B+] - Y[B-]): the omission trials subtract what the history leaves behind, so D is the
history x probe interaction (a purely additive response y = h + p gives D = 0 exactly; tests/test_spine.py).
Reported: mean D, its 95% interval from the same sign flips, the two-sided sign-flip p over blocks (the rule:
p < 0.05), median D, the share of blocks with D > 0, the minimum detectable effect, the four condition means,
bursts, baseline rate, flagged trials, first half against second. Sensitivity: without trials that had a burst in
the 1 s before onset; and Y minus a matched pre-history window. Descriptive: Y[A-] against Y[B-] (the leftover
activity alone), and D in the 20-60 and 60-250 ms sub-windows. Secondary, labelled: a logistic decoder of A vs B on
the probe trials' per-electrode counts, trained on the first half of the session and scored once on the second.

The design's numbers live in probe_design.json and are copied into frozen.json.

    uv run python probe.py [--short] [--seed 0]
    NO_CELLS=1 uv run python probe.py          # simulator check: cells off, D must sit near 0
"""
import datetime
import hashlib
import json
import math
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bci
import rhythm

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN, DESIGN = os.path.join(HERE, "frozen.json"), os.path.join(HERE, "probe_design.json")
CONDS = ("A+", "A-", "B+", "B-")
MS = bci.FPS // 1000


def load_design(path=DESIGN):
    with open(path) as f:
        d = json.load(f)
    a, b = (np.asarray(d["histories_ms"][k]) for k in "AB")
    for h in (a, b):  # same count and span; the probe after the last history pulse; 100 ms minimum spacing
        assert h[0] == 0 and len(h) == len(a) and h[-1] == a[-1] and np.diff(np.append(h, d["probe_ms"])).min() >= 100
    assert not np.array_equal(a, b)
    y0, y1 = d["y_window_ms"]
    assert bci.SKIP_MS <= y0 < d["y_window_start_if_electrical_ms"] < y1
    assert all(y0 <= w0 < w1 <= y1 for w0, w1 in d["sub_windows_ms"])
    assert d["burst_lookback_ms"] >= y1 - y0, "the matched baseline window must fit in the lookback"
    end = d["burst_lookback_ms"] + d["probe_ms"] + y1 + 50
    assert end < d["onset_s"] * 1000 and end < RING_MS, "trials would overlap"
    return d


RING_MS = 4096
D = load_design()


def schedule(rng, blocks):
    """[(block, condition)]: every block holds A+ A- B+ B- once, in random order."""
    return [(b, CONDS[i]) for b in range(blocks) for i in rng.permutation(4)]


def pulses_ms(cond):
    h = list(D["histories_ms"][cond[0]])
    return h + [D["probe_ms"]] if cond[1] == "+" else h


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---- frozen.json -------------------------------------------------------------------------------
def freeze(section, config, path=FROZEN):
    """Write one section of frozen.json: its config, the config's sha256 and the UTC time. Returns the entry."""
    entry = dict(config=config, sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                 frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    allf = load(path, verify=False)
    allf[section] = entry
    with open(path, "w") as f:
        json.dump(allf, f, indent=1)
    return entry


def load(path=FROZEN, verify=True):
    """frozen.json -> dict; with verify, every entry's hash must match its config."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        allf = json.load(f)
    for k, e in allf.items():
        if verify and hashlib.sha256(json.dumps(e["config"], sort_keys=True).encode()).hexdigest() != e["sha256"]:
            raise ValueError(f"frozen.json: {k} config does not match its hash")
    return allf


# ---- one trial ---------------------------------------------------------------------------------
def trial_record(raster, read, win, burst_thr):
    """raster: spikes per ms per electrode, from burst_lookback_ms before history onset to the end of the Y window
    (64 columns). win: the frozen Y window (ms after the probe slot). Returns the trial's numbers."""
    pre, (y0, y1), t = D["burst_lookback_ms"], win, D["burst_lookback_ms"] + D["probe_ms"]
    r = raster[:, read]
    y_ch = r[t + y0: t + y1].sum(0)
    base = int(r[pre - (y1 - y0): pre].sum())  # matched length, just before history onset
    return dict(y=int(y_ch.sum()), base=base, y_bs=int(y_ch.sum()) - base, y_ch=y_ch.tolist(), win=list(win),
                subs=[int(r[t + max(w0, y0): t + w1].sum()) for w0, w1 in D["sub_windows_ms"]],
                base_hz=base / ((y1 - y0) / 1000) / len(read), burst_pre=bool(bursts(raster[:pre], burst_thr).any()),
                bursts_in=int(bursts(raster[pre:], burst_thr).sum()))


def bursts(raster, thr):
    """Network bursts, label-blind: bins of burst_bin_ms in which at least thr usable electrodes spike."""
    n = D["burst_bin_ms"]
    k = len(raster) // n
    if not k:
        return np.zeros(0, bool)
    return (raster[:k * n, bci.USABLE].reshape(k, n, -1).sum(1) > 0).sum(1) >= thr


def record(neurons, seconds):
    """Spikes per ms per electrode over `seconds`, nothing stimulated."""
    out = np.zeros((int(seconds * 1000), 64), dtype=np.int16)
    t0 = None
    for tick in neurons.loop(ticks_per_second=1000):
        t0 = tick.timestamp // MS if t0 is None else t0
        for sp in tick.analysis.spikes:
            if 0 <= sp.timestamp // MS - t0 < len(out):
                out[sp.timestamp // MS - t0, sp.channel] += 1
        if tick.timestamp // MS - t0 >= len(out):
            return out


def burst_threshold(quiet):
    """Frozen from the quiet recording: mean + burst_sd sd of the active-electrode count per bin (at least 2)."""
    n = D["burst_bin_ms"]
    k = len(quiet) // n
    a = (quiet[:k * n, bci.USABLE].reshape(k, n, -1).sum(1) > 0).sum(1)
    return max(2, int(math.ceil(a.mean() + D["burst_sd"] * a.std())))


def jitter_check(neurons, ch, read):
    """Pulse the input; a readout electrode whose events 20-40 ms after the pulse come on most pulses with under
    1 ms of jitter is electrical, not neural. Returns (electrical electrodes, the Y window start)."""
    j = D["jitter_check"]
    w0, w1 = (v * MS for v in j["window_ms"])
    lat = {c: [] for c in read}
    fired, t0 = [], neurons.timestamp() + bci.FPS // 2
    for tick in neurons.loop(ticks_per_second=1000):
        ts = tick.timestamp
        if len(fired) < j["pulses"] and ts >= t0 + len(fired) * j["spacing_ms"] * MS:
            p = neurons.create_stim_plan()
            p.stim(bci.ChannelSet(ch), bci.STIM, rhythm.BURST)
            fired.append(dict(p=p.run(), seen=set()))
        if fired:
            f = fired[-1]
            bci.relock(f, tick, (ch,), ("p",))
            for sp in tick.analysis.spikes:
                if sp.channel in lat and sp.channel not in f["seen"] and w0 <= sp.timestamp - f["p"] < w1:
                    f["seen"].add(sp.channel)  # the first event after this pulse
                    lat[sp.channel].append((sp.timestamp - f["p"]) / MS)
        if len(fired) == j["pulses"] and ts >= fired[-1]["p"] + w1:
            break
    bad = sorted(c for c, v in lat.items() if len(v) >= max(2, j["min_share"] * j["pulses"]) and np.std(v) < j["max_sd_ms"])
    return bad, (D["y_window_start_if_electrical_ms"] if bad else D["y_window_ms"][0])


ECHO_TOL = 5 * MS  # an echo within 5 ms of the expected pulse time is that pulse's


def _flag(st, why):
    if why not in (st["flag"] or ""):
        st["flag"] = why if not st["flag"] else st["flag"] + "; " + why


def run_trials(neurons, plan, read, win, burst_thr, on_trial=None):
    """Play the plan at fixed onsets. Returns one record per trial; a trial with a late pulse (never fired), a pulse
    the device rejected or never echoed, or skipped ticks carries `flag`, and still counts."""
    pre, end = D["burst_lookback_ms"], D["probe_ms"] + D["y_window_ms"][1]
    spk = np.zeros((RING_MS, 64), dtype=np.int16)
    held = np.full(RING_MS, -1)  # the absolute ms each ring row holds: a row from another lap reads as empty
    period = int(D["onset_s"] * 1000)
    t0 = neurons.timestamp() // MS + pre + 1000  # first onset: a full lookback after a 1 s settle
    recs, st = [], dict(k=0, fired=0, flag=None, prev=None, pend=[])
    burst = rhythm.BURST
    for tick in neurons.loop(ticks_per_second=1000):
        now = tick.timestamp // MS
        for sp in tick.analysis.spikes:
            m = int(sp.timestamp) // MS
            if held[m % RING_MS] != m:
                spk[m % RING_MS], held[m % RING_MS] = 0, m
            spk[m % RING_MS, sp.channel] += 1
        for e in tick.analysis.stims:  # reconcile each delivered pulse with the device's echo
            if e.channel == rhythm.IN_CH:
                hit = [t for t in st["pend"] if abs(int(e.timestamp) - t) <= ECHO_TOL]
                if hit:
                    st["pend"].remove(hit[0])
                    if abs(int(e.timestamp) - hit[0]) > 2 * MS:  # paired, but delivered off schedule: same 2 ms rule as firing
                        _flag(st, "pulse delivered over 2 ms off schedule")
        k = st["k"]
        onset = t0 + k * period
        if st["prev"] is not None and now - st["prev"] > 2 and now >= onset - pre and not st["flag"]:
            _flag(st, "ticks skipped")  # the raster has a hole: flagged, still counted
        st["prev"] = now
        block, cond = plan[k]
        due = pulses_ms(cond)
        if st["fired"] < len(due) and now >= onset + due[st["fired"]]:
            if now - (onset + due[st["fired"]]) > 2:
                _flag(st, "pulse late, not fired")  # never fired late: catching up would bunch pulses
            else:
                p = neurons.create_stim_plan()
                p.stim(rhythm.INPUT, bci.STIM, burst)
                try:
                    st["pend"].append(p.run())
                except (bci.cl.TransactionRejected, bci.cl.ChannelQueueFull) as e:
                    _flag(st, f"pulse rejected by the device ({type(e).__name__})")
            st["fired"] += 1
        if now >= onset + end + 2:
            if st["pend"]:
                _flag(st, f"{len(st['pend'])} pulse(s) with no echo")
            ms = np.arange(onset - pre, onset + end)
            rows = np.where((held[ms % RING_MS] == ms)[:, None], spk[ms % RING_MS], 0).astype(int)
            rec = dict(i=k, block=block, cond=cond, flag=st["flag"], **trial_record(rows, read, win, burst_thr),
                       raster=[int(t) - pre for t in np.nonzero(rows[:, read].sum(1))[0] if t >= pre - 200])
            recs.append(rec)
            if on_trial:
                on_trial(rec)
            st.update(k=k + 1, fired=0, flag=None, pend=[])
            if st["k"] == len(plan):
                break
    return recs


# ---- the analysis --------------------------------------------------------------------------------
def block_d(recs, key="y", sub=None):
    """{block: D} for blocks with all four trials present."""
    by = {}
    for r in recs:
        by.setdefault(r["block"], {})[r["cond"]] = r[key] if sub is None else r["subs"][sub]
    return {b: (v["A+"] - v["A-"]) - (v["B+"] - v["B-"]) for b, v in sorted(by.items()) if len(v) == 4}


def _signs(n):
    """Every sign pattern up to 16 blocks (exact), else sign_flips random ones (seed 0): the same flips throughout."""
    if n <= 16:
        return 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1), True
    return np.random.default_rng(0).choice([-1, 1], (D["sign_flips"], n)), False


def sign_flip_p(d, delta=0.0, signs=None):
    """Two-sided sign-flip permutation p of mean(d) = delta."""
    x = np.asarray(d, float) - delta
    if not len(x) or not x.any():
        return 1.0
    sg, exact = signs or _signs(len(x))
    hits = int((np.abs(sg @ x) >= abs(x.sum()) - 1e-9).sum())
    return hits / len(sg) if exact else (hits + 1) / (len(sg) + 1)


def flip_ci(d, signs=None, alpha=0.05):
    """95% interval by inverting the same sign-flip test: every delta it does not reject. None: unbounded."""
    d = np.asarray(d, float)
    signs = signs or _signs(len(d))
    m, span = d.mean(), np.ptp(d) + abs(d.mean()) + 1
    out = []
    for far in (m - 10 * span, m + 10 * span):
        if sign_flip_p(d, far, signs) >= alpha:
            out.append(None)  # too few blocks to reject anything that far out
            continue
        a, b = far, m  # p(far) < alpha <= p(mean)
        for _ in range(50):
            c = (a + b) / 2
            a, b = (c, b) if sign_flip_p(d, c, signs) < alpha else (a, c)
        out.append(float(b))
    return out


def summarise_d(d):
    d = np.asarray(list(d), float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else 0.0, median=float(np.median(d)) if n else 0.0,
                    share_pos=float((d > 0).mean()) if n else 0.0, ci=[None, None], p=1.0, sd=0.0, mde=None)
    signs = _signs(n)
    sd = float(d.std(ddof=1))
    return dict(n=n, mean=float(d.mean()), median=float(np.median(d)), share_pos=float((d > 0).mean()),
                ci=flip_ci(d, signs), p=sign_flip_p(d, 0.0, signs), sd=sd,
                mde=float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) / math.sqrt(n) * sd))


def secondary(recs):
    """Labelled secondary: decode A vs B from the probe trials' per-electrode Y-window counts; train on the first
    half of the session's trials, score once on the second."""
    half = len(recs) // 2
    pr = [r for r in recs if r["cond"][1] == "+"]
    tr, te = [r for r in pr if r["i"] < half], [r for r in pr if r["i"] >= half]
    X = lambda rs: np.log1p(np.array([r["y_ch"] for r in rs], float))
    y = lambda rs: np.array([r["cond"][0] == "A" for r in rs], int)
    if len(set(y(tr))) < 2 or not te:
        return dict(n=len(te), correct=0, acc=0.0, ci=[0.0, 1.0], p=1.0, what="too few trials")
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X(tr), y(tr))
    k = int((m.predict(X(te)) == y(te)).sum())
    return dict(n=len(te), correct=k, acc=k / len(te), ci=list(wilson(k, len(te))), p=rhythm.p_chance(k, len(te)),
                what="A vs B from the probe trials' per-electrode counts, trained on the first half, scored once on the second")


def analyse(recs):
    d = block_d(recs)
    blocks = sorted(d)
    h = len(blocks) // 2
    clean = [r for r in recs if not r["burst_pre"]]
    by = {}
    for r in recs:
        by.setdefault(r["block"], {})[r["cond"]] = r["y"]
    left = [v["A-"] - v["B-"] for v in by.values() if "A-" in v and "B-" in v]
    out = dict(primary=summarise_d(d.values()), d_by_block={str(b): v for b, v in d.items()},
               means={c: float(np.mean([r["y"] for r in recs if r["cond"] == c])) if any(r["cond"] == c for r in recs) else None
                      for c in CONDS},
               halves=[summarise_d([d[b] for b in blocks[:h]]), summarise_d([d[b] for b in blocks[h:]])],
               sensitivity=dict(
                   no_burst=dict(summarise_d(block_d(clean).values()), excluded_trials=len(recs) - len(clean),
                                 what="without trials that had a spontaneous network burst in the 1 s before history onset"),
                   baseline_subtracted=dict(summarise_d(block_d(recs, "y_bs").values()),
                                            what="Y minus the same electrodes' count in a matched window before history onset")),
               descriptive=dict(
                   leftover=dict(summarise_d(left), what="Y[A-omit] - Y[B-omit]: the history's leftover activity alone"),
                   sub_windows=[dict(summarise_d(block_d(recs, sub=k).values()), window_ms=w)
                                for k, w in enumerate(D["sub_windows_ms"])]),
               bursts=dict(trials_with_burst_before=sum(r["burst_pre"] for r in recs),
                           burst_bins_in_trials=sum(r["bursts_in"] for r in recs)),
               baseline_hz=float(np.mean([r["base_hz"] for r in recs])) if recs else 0.0,
               flagged=dict(n=sum(bool(r["flag"]) for r in recs), reasons=sorted({r["flag"] for r in recs if r["flag"]})),
               n_trials=len(recs), secondary=secondary(recs))
    p = out["primary"]
    out["effect"] = bool(p["p"] < 0.05)
    bound = max((abs(v) for v in p["ci"] if v is not None), default=None)
    out["null_line"] = None if out["effect"] else (
        f"No history effect larger than {bound:.1f} spikes either way in this dish tonight." if bound is not None and None not in p["ci"]
        else "Too few blocks to bound the history effect.")
    return out


def run(neurons, seed=0, blocks=None, path=FROZEN, scan_s=30.0, on_progress=None, on_trial=None):
    """Quiet recording, scan, jitter check (all label-blind), freeze, then the trials. on_progress(stage, done, total)."""
    blocks = blocks or D["blocks"]
    prog = lambda *a: on_progress and on_progress(*a)
    prog("quiet", 0, 1)
    thr = burst_threshold(record(neurons, D["quiet_s"]))
    sc = rhythm.scan(neurons, scan_s, on_progress=lambda i, n: prog("scan", i, n), win_ms=D["select_window_ms"])
    read = sorted(sc["ranked"][:D["read_cap"]] if D["read_cap"] else sc["ranked"])
    prog("jitter", 0, 1)
    electrical, y0 = jitter_check(neurons, sc["input"], read)
    win = [y0, D["y_window_ms"][1]]
    rhythm.set_input(sc["input"], read)
    entry = freeze("omission", dict(design=D, input=sc["input"], read=read, scan_drive=sc["drive"], y_window_ms=win,
                                    electrical=electrical, burst_threshold=thr, seed=seed, blocks=blocks,
                                    source=bci.source(), stim=dict(us_per_phase=160, ua=2.0),
                                    spike_detection="the device's own detector, unchanged by this code during the run"
                                                    + (f" (simulator CL_SDK_SPIKE_PERCENTILE={os.environ['CL_SDK_SPIKE_PERCENTILE']})"
                                                       if "CL_SDK_SPIKE_PERCENTILE" in os.environ else "")), path)
    plan = schedule(np.random.default_rng(seed), blocks)
    done = [0]

    def seen(rec):
        done[0] += 1
        if on_trial:
            on_trial(rec)
        prog("trials", done[0], len(plan))
    recs = run_trials(neurons, plan, read, win, thr, seen)
    cfg = load(path)["omission"]  # the analysis runs on the frozen readout, after the hash check
    assert cfg["sha256"] == entry["sha256"], "frozen.json changed during the run"
    return dict(analyse(recs), frozen=dict(sha256=entry["sha256"], frozen_at=entry["frozen_at"], path=path),
                input=sc["input"], read=read, y_window_ms=win, electrical=electrical, burst_threshold=thr, blocks=blocks)


def fmt(r):
    p, sb, ss, (a, b), sec, de = (r["primary"], r["sensitivity"]["no_burst"], r["sensitivity"]["baseline_subtracted"],
                                  r["halves"], r["secondary"], r["descriptive"])
    f1 = lambda v: "unbounded" if v is None else f"{v:+.2f}"
    dl = lambda x: (f"mean {x['mean']:+.2f}, 95% CI {f1(x['ci'][0])} to {f1(x['ci'][1])}, p = {x['p']:.2g}, "
                    f"median {x['median']:+.1f}, D > 0 in {x['share_pos']:.0%}, {x['n']} blocks")
    w, mde = r["y_window_ms"], "n/a" if p["mde"] is None else f"{p['mde']:.1f}"
    return "\n".join([
        f"frozen {r['frozen']['sha256'][:12]} at {r['frozen']['frozen_at']}: input {r['input']}, {len(r['read'])} read "
        f"electrodes, Y window {w[0]}-{w[1]} ms (electrical: {r['electrical'] or 'none'}), burst threshold "
        f"{r['burst_threshold']} electrodes per {D['burst_bin_ms']} ms; {r['n_trials']} trials in {r['blocks']} blocks, "
        f"{r['flagged']['n']} flagged {r['flagged']['reasons'] or ''}",
        f"means of raw Y (pooled spikes {w[0]}-{w[1]} ms after the probe slot): "
        + ", ".join(f"{c} {'n/a' if r['means'][c] is None else round(r['means'][c], 2)}" for c in CONDS),
        f"PRIMARY D = (A+ - A-) - (B+ - B-): {dl(p)} (two-sided sign flips)",
        f"minimum detectable effect, 80% power: {mde} spikes "
        f"(sd of D {p['sd']:.1f})",
        *([r["null_line"]] if r["null_line"] else []),
        f"sensitivity, no pre-onset bursts ({sb['excluded_trials']} trials out): {dl(sb)}",
        f"sensitivity, baseline-subtracted: {dl(ss)}",
        f"descriptive, Y[A-] - Y[B-] (leftover alone): {dl(de['leftover'])}",
        *[f"descriptive, D in {x['window_ms'][0]}-{x['window_ms'][1]} ms: {dl(x)}" for x in de["sub_windows"]],
        f"drift: first half D {a['mean']:+.2f} ({a['n']} blocks), second half {b['mean']:+.2f} ({b['n']} blocks)",
        f"bursts: {r['bursts']['trials_with_burst_before']} trials with one in the 1 s before onset, "
        f"{r['bursts']['burst_bins_in_trials']} burst bins inside trials; baseline {r['baseline_hz']:.1f} Hz per read electrode",
        f"secondary (not the headline): decoder {sec['correct']}/{sec['n']} = {sec['acc']:.0%}, 95% CI "
        f"{sec['ci'][0]:.0%}-{sec['ci'][1]:.0%}, one-sided p = {sec['p']:.2g}",
        f"RESULT: {'history changes the response to the same pulse' if r['effect'] else 'no history effect measured'} "
        f"(rule: primary sign-flip p < 0.05)"])


if __name__ == "__main__":
    a = bci.parse_args()
    with bci.cl.open() as neurons:
        r = run(neurons, a.seed, D["blocks_short"] if a.short else D["blocks"])
    print(f"source: {bci.source()}{' (simulator check: cells off)' if os.environ.get('NO_CELLS') == '1' else ''}")
    print(fmt(r))
