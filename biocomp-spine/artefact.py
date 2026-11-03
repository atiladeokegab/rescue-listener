"""Measure the stimulation artefact on this dish and suggest a --skip-ms value.

    uv run python artefact.py [--electrode 27] [--pulses 20]    # about 12 s on the dish

Fires the usual safe pulse (bci.STIM) on one electrode every 0.5 s. Around each pulse's echo it takes the envelope
(max |uV| per 1 ms over every usable electrode, the channels the readers use) from 20 ms before to 100 ms after,
and the median over pulses. The artefact ends where that envelope falls under 3x its pre-pulse level for good.
Suggested --skip-ms = end + 2 ms, rounded up to 5, never below 10. Neural responses are tens of uV; an artefact
is hundreds or more, so 3x the noise envelope sits between them (re-check by eye on the printed profile).
"""
import argparse

import numpy as np

import bci
from bci import FPS, cl

PRE_MS, POST_MS, GAP_S = 20, 100, 0.5


def measure(neurons, electrode=27, pulses=20):
    ms = FPS // 1000
    ring = np.zeros((bci.RING, 64), dtype=np.float32)
    chs = bci.ChannelSet(electrode)
    st = dict(next=neurons.timestamp() + FPS, fired=[])
    envs = []
    for tick in neurons.loop(ticks_per_second=1000):
        ts = tick.timestamp
        bci.write_tick(ring, tick)
        f = st["fired"][-1] if st["fired"] else None
        if f:
            bci.relock(f, tick, (electrode,), ("p",))
            if not f.get("done") and ts >= f["p"] + POST_MS * ms + ms:
                w = np.abs(bci.ring_window(ring, f["p"] - PRE_MS * ms, (PRE_MS + POST_MS) * ms, ts)[:, bci.USABLE])
                envs.append(w.reshape(PRE_MS + POST_MS, ms, -1).max(axis=(1, 2)))
                f["done"] = True
                if len(envs) == pulses:
                    break
        if len(st["fired"]) < pulses and ts >= st["next"]:
            plan = neurons.create_stim_plan()
            plan.stim(chs, bci.STIM, bci.BURST)
            st["fired"].append(dict(p=plan.run()))
            st["next"] = ts + int(GAP_S * FPS)
    env = np.median(envs, axis=0)
    level = float(np.median(env[:PRE_MS]))
    above = np.nonzero(env[PRE_MS:] > 3 * level)[0]
    end = int(above[-1]) + 1 if len(above) else 0
    return dict(end_ms=end, skip_ms=max(10, int(np.ceil((end + 2) / 5) * 5)), level_uv=level,
                peak_uv=float(env[PRE_MS:].max()), profile=env[PRE_MS:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--electrode", type=int, default=27)
    ap.add_argument("--pulses", type=int, default=20)
    a = ap.parse_args()
    with cl.open() as neurons:
        r = measure(neurons, a.electrode, a.pulses)
    print(f"source: {bci.source()}  electrode {a.electrode}, {a.pulses} pulses")
    print("ms after pulse : " + " ".join(f"{k:>5d}" for k in range(0, 30, 2)))
    print("envelope (uV)  : " + " ".join(f"{v:5.0f}" for v in r["profile"][:30:2]))
    print(f"pre-pulse level {r['level_uv']:.0f} uV, peak {r['peak_uv']:.0f} uV, artefact ends {r['end_ms']} ms after "
          f"the pulse -> suggest --skip-ms {r['skip_ms']}")
