"""Noise filter (exploratory, not the pre-registered test): does the dish stop answering what repeats and keep
answering what is rare? Stimulus-specific adaptation after Eytan, Brenner & Marom 2003 (J Neurosci 23:9349).

A flip-flop oddball on two input electrodes, A and B (the scan's two strongest drivers; A is the input the scan
picked). Condition 1: A frequent (90%), B rare (10%), 150 pulses 500 ms apart, no two rare pulses in a row, fixed
seed. 30 s rest. Condition 2: the same sequence with A and B swapped. About 3 minutes of chip time. Same safe pulse
as probe.py. Response = spikes 20-250 ms after each pulse (probe's window, so the same artefact blanking), summed
over every usable electrode except A and B. A pulse that is late (never fired), rejected or never echoed flags
its stimulus, which still counts.

For each electrode X: SSA index = (rare - frequent) / (rare + frequent), from its mean response when rare and when
frequent. Each index compares an electrode with itself, so how strongly A or B drives the dish does not enter it. Time
does: the conditions run in a fixed order, so slow drift or carry-over from condition 1 can make a contrast on its own.
Reported: both indices and their mean with a bootstrap 95% CI over stimuli (it resamples pulses only, not that drift).
The verdict is descriptive (positive or negative SSA contrast), never "filtering". No p-value.

    uv run python noisefilter.py [--seed 0]
"""
import numpy as np

import bci
import probe
import rhythm

N_STIM, RARE_SHARE, ISI_MS, REST_S, SEED = 150, 0.1, 500, 30.0, 0
WIN_MS = tuple(probe.D["y_window_ms"])  # 20-250 ms: probe's window and its artefact blanking
MS = bci.FPS // 1000
DEFAULT_B = 45  # before any scan: a usable electrode well away from the default input (27)
assert DEFAULT_B in bci.USABLE and WIN_MS[1] < ISI_MS


def schedule(seed=SEED, n=N_STIM, share=RARE_SHARE):
    """Boolean mask, True = rare: exactly round(n * share) rare, never two in a row."""
    k = round(n * share)
    pos = np.sort(np.random.default_rng(seed).choice(n - k + 1, k, replace=False)) + np.arange(k)  # gaps of 2 or more
    rare = np.zeros(n, bool)
    rare[pos] = True
    return rare


def plan(a, b, seed=SEED, n=N_STIM, share=RARE_SHARE):
    """[(condition, electrode, rare)]: condition 1 has A frequent, condition 2 the same mask with A and B swapped."""
    m = schedule(seed, n, share)
    return [(1, b if r else a, bool(r)) for r in m] + [(2, a if r else b, bool(r)) for r in m]


def electrodes():
    """A = the input the scan picked (rhythm.IN_CH); B = the next strongest driver, or DEFAULT_B before a scan."""
    a = rhythm.IN_CH
    b = next((c for c in rhythm.DRIVE_ORDER if c != a), None) if rhythm.DRIVE_ORDER else None
    return a, (b if b is not None else DEFAULT_B if a != DEFAULT_B else 27)


def duration_s(n=N_STIM, rest_s=REST_S):
    return 2 * n * ISI_MS / 1000 + rest_s


def run_stimuli(neurons, pl, a, b, rest_s=REST_S, on_stim=None):
    """Play the plan: one pulse every ISI_MS, rest_s between the conditions. One record per stimulus."""
    read = [c for c in bci.USABLE if c not in (a, b)]
    w0, w1 = WIN_MS[0] * MS, WIN_MS[1] * MS
    t0 = neurons.timestamp() // MS + 1000  # 1 s settle
    due = [t0 + k * ISI_MS + (int(rest_s * 1000) if cond == 2 else 0) for k, (cond, _, _) in enumerate(pl)]
    recs, k, cur, prev = [], 0, None, None

    def close(c):
        if c.pop("fired") and not c["echo"]:
            probe._flag(c, "pulse with no echo")
        del c["echo"], c["p"], c["at"]
        recs.append(c)
        if on_stim:
            on_stim(c)
    for tick in neurons.loop(ticks_per_second=1000):
        now = tick.timestamp // MS
        if cur:
            if prev is not None and now - prev > 2:
                probe._flag(cur, "ticks skipped")
            for e in tick.analysis.stims:  # echoes first: the window locks to when the pulse really landed
                if e.channel == cur["ch"] and not cur["echo"] and abs(int(e.timestamp) - cur["p"]) <= probe.ECHO_TOL:
                    cur["echo"], cur["at"] = True, int(e.timestamp)
                    if abs(int(e.timestamp) - cur["p"]) > 2 * MS:
                        probe._flag(cur, "pulse delivered over 2 ms off schedule")
            for sp in tick.analysis.spikes:
                if sp.channel in read and w0 <= int(sp.timestamp) - cur["at"] < w1:
                    cur["y"] += 1
            if now * MS >= cur["at"] + w1 + 2 * MS:
                close(cur)
                cur = None
                if k == len(pl):
                    break
        prev = now
        if cur is None and k < len(pl) and now >= due[k]:
            cond, ch, rare = pl[k]
            cur = dict(i=k, cond=cond, ch=ch, rare=rare, y=0, flag=None, echo=False, fired=False, p=due[k] * MS, at=due[k] * MS)
            if now - due[k] > 2:
                probe._flag(cur, "pulse late, not fired")  # counted from its scheduled time, never fired late
            else:
                p = neurons.create_stim_plan()
                p.stim(bci.ChannelSet(ch), bci.STIM, rhythm.BURST)
                try:
                    cur["p"] = cur["at"] = int(p.run())  # "at" moves to the echo when it arrives
                    cur["fired"] = True
                except (bci.cl.TransactionRejected, bci.cl.ChannelQueueFull) as e:
                    probe._flag(cur, f"pulse rejected by the device ({type(e).__name__})")
            k += 1
    if cur is not None:  # the loop ended inside a response window: the pulse went out, so keep it, flagged
        probe._flag(cur, "response window cut short (run stopped)")
        close(cur)
    return recs


def ssa(rare, frequent):
    """(rare - frequent) / (rare + frequent); 0 when neither drew a spike (nothing to filter)."""
    s = rare + frequent
    return 0.0 if s == 0 else (rare - frequent) / s


def _indices(g, a, b):
    sa, sb = ssa(g[a, True].mean(), g[a, False].mean()), ssa(g[b, True].mean(), g[b, False].mean())
    return sa, sb, (sa + sb) / 2


def analyse(recs, a, b, n_boot=2000, seed=0):
    g = {(x, r): np.array([q["y"] for q in recs if q["ch"] == x and q["rare"] == r], float) for x in (a, b) for r in (True, False)}
    if any(not len(v) for v in g.values()):
        return dict(error="a condition has no stimuli")
    rng = np.random.default_rng(seed)
    boot = np.array([_indices({k: rng.choice(v, len(v)) for k, v in g.items()}, a, b) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sa, sb, m = _indices(g, a, b)
    out = dict(a=a, b=b, ssa_a=sa, ssa_b=sb, mean=m, ci=[float(lo[2]), float(hi[2])],
               ci_a=[float(lo[0]), float(hi[0])], ci_b=[float(lo[1]), float(hi[1])],
               means={f"{x}_{'rare' if r else 'frequent'}": float(v.mean()) for (x, r), v in g.items()},
               n=len(recs), flagged=dict(n=sum(bool(q["flag"]) for q in recs), reasons=sorted({q["flag"] for q in recs if q["flag"]})),
               label="exploratory")
    out["verdict"] = ("positive" if lo[2] > 0 else "negative" if hi[2] < 0 else "none")  # a contrast, not a mechanism
    return out


def run(neurons, seed=SEED, n=N_STIM, rest_s=REST_S, on_stim=None, a=None, b=None):
    if a is None:
        a, b = electrodes()
    pl = plan(a, b, seed, n)
    recs = run_stimuli(neurons, pl, a, b, rest_s, on_stim)
    meta = dict(seed=seed, n_per_condition=n, window_ms=list(WIN_MS), source=bci.source())
    done = sum("cut short" not in (q["flag"] or "") for q in recs)   # a window the stop cut short is not a whole pulse
    if done < len(pl):  # a stopped run reads no verdict: missing or truncated pulses would bias the index
        return dict(error=f"run stopped after {done} of {len(pl)} pulses: no result", n=len(recs), **meta)
    return dict(analyse(recs, a, b), **meta)


CAVEAT = ("The conditions run in a fixed order, so drift or carry-over can make this contrast; "
          "the interval resamples pulses only.")


def fmt(r):
    if "error" in r:
        return f"EXPLORATORY noise filter: {r['error']}"
    v = {"positive": "rare pulses drew relatively more (positive SSA contrast)",
         "negative": "rare pulses drew relatively less (negative SSA contrast)", "none": "no filtering detected"}[r["verdict"]]
    return (f"EXPLORATORY noise filter, electrodes A {r['a']} / B {r['b']}: SSA A {r['ssa_a']:+.3f}, B {r['ssa_b']:+.3f}, "
            f"mean {r['mean']:+.3f} (bootstrap 95% CI {r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}); {v}. "
            f"{r['flagged']['n']} of {r['n']} stimuli flagged {r['flagged']['reasons'] or ''}. {CAVEAT}")


if __name__ == "__main__":
    args = bci.parse_args()
    a, b = electrodes()
    print(f"planned chip time {duration_s():.0f} s: 2 x {N_STIM} pulses every {ISI_MS} ms on electrodes {a} and {b}, "
          f"{REST_S:.0f} s rest")
    with bci.cl.open() as neurons:
        res = run(neurons, args.seed, a=a, b=b)
    print(f"source: {bci.source()}")
    print(fmt(res))
