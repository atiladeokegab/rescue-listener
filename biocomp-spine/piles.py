"""Blind piles: three listening windows; on each pile the judge, in secret, taps the five-tap reply (a target pile)
or taps like rubble falling. Any of 0, 1, 2 or 3 piles can be targets.

The code never knows which piles are targets: every window is live taps, played into the dish as in knock mode.
A pile is flagged when its target-rhythm score is above the threshold frozen at calibration (the 95th percentile of
held-out scores on irregular sequences), so three rubble piles can all go unflagged. Live: `uv run python live.py
--rhythm`, calibrate, then "Blind piles"; the judge says which piles were targets and Reveal reports hits and false
flags. This script measures it on the practice dish with simulated human taps:

    uv run python piles.py [--rounds 100] [--seed 0]    # calibrate, then play the rounds (about 20 min on the mock)

Each round draws 0-3 target piles. Target piles get the five-tap reply (8% jitter); the others get human "random"
taps (4-7 taps, gaps 150-900 ms, skewed short so they clump), and, as the fallback, rescue-data debris trials.
"""
import argparse

import numpy as np

import bci
import ear
import rhythm
from bci import cl


def human_taps(rng):
    """How a person taps 'like rubble falling': 4-7 taps, gaps 150-900 ms, mostly short with a long tail (clumps)."""
    g = 0.15 + 0.75 * rng.beta(1.0, 2.2, rng.integers(3, 7))
    return np.concatenate([[0], np.cumsum(g)])


def flags(scores, threshold):
    """Flagged piles (1-based): every pile whose score is above the frozen threshold. Each pile is judged alone,
    so the order of the piles and ties between them change nothing. No score (no response) is never flagged."""
    return [n for n, v in enumerate(scores, 1) if v is not None and v > threshold]


def tally(flagged, targets):
    """Hits, false flags and misses for the reveal table."""
    return dict(hits=len(set(flagged) & set(targets)), false_flags=len(set(flagged) - set(targets)),
                misses=len(set(targets) - set(flagged)))


def outcome(scores, flagged):
    """The dish's one-line answer. A pile with no score makes the round unsure, whatever else was flagged;
    otherwise the flagged piles, or none."""
    def piles(ns):
        ns = [str(n) for n in ns]
        return ("Pile " if len(ns) == 1 else "Piles ") + (ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1])
    silent = [n for n, v in enumerate(scores, 1) if v is None]
    if silent:
        return "Dish unsure: no score for " + piles(silent)
    return "Dish flagged " + piles(flagged) if flagged else "Dish flagged no pile"


def simulate(neurons, rounds=100, seed=0):
    cal = rhythm.calibrate(neurons, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    idx = ear.index()
    debris = [rhythm.clean(t["knock_times_ms"]) for t in idx.values() if t["class"] == "debris"]
    targets, items = [], []  # items: (round, pile, kind) with kind reply | human | debris
    for r in range(rounds):
        tg = sorted(rng.choice([1, 2, 3], rng.integers(0, 4), replace=False).tolist())
        targets.append(tg)
        for n in (1, 2, 3):
            items += [(r, n, "reply", rhythm.five_tap(rng))] if n in tg else \
                [(r, n, "human", human_taps(rng)), (r, n, "debris", debris[rng.integers(len(debris))])]
    perm = rng.permutation(len(items))  # played in random order (dish adaptation can't favour one kind)
    F, _, _, _ = rhythm.run_rhythm(neurons, [items[i][3] for i in perm])
    assert len(F) == len(items), "a sequence was dropped: rerun"  # rows must line up with the rounds
    F = F[np.argsort(perm)]
    T = np.array([rhythm.timing_features(it[3]) for it in items])
    k = list(cal["model"].classes_).index(1)
    score = dict(dish=cal["model"].predict_proba(F)[:, k], software=cal["timing_model"].predict_proba(T)[:, k])
    thr = dict(dish=cal["threshold"], software=cal["software_threshold"])
    out = dict(cal=cal, rounds=rounds)
    for who in ("dish", "software"):
        s = {(r, n, kind): v for (r, n, kind, _), v in zip(items, score[who])}
        kinds = lambda kind: np.array([v for (r, n, kd), v in s.items() if kd == kind])
        out[who] = dict(threshold=thr[who], hit=float((kinds("reply") > thr[who]).mean()),
                        false_human=float((kinds("human") > thr[who]).mean()),
                        false_debris=float((kinds("debris") > thr[who]).mean()))
        for other in ("human", "debris"):  # a round is right when the flags are exactly the target piles
            right = [flags([s.get((r, n, "reply"), s.get((r, n, other))) for n in (1, 2, 3)], thr[who]) == tg
                     for r, tg in enumerate(targets)]
            out[who]["rounds_right_" + other] = float(np.mean(right))
    out["no_target_rounds"] = sum(not tg for tg in targets)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with cl.open() as neurons:
        r = simulate(neurons, a.rounds, a.seed)
    print(f"source: {bci.source()}  {rhythm.fmt(r['cal'])}")
    print(f"{r['rounds']} rounds, 0-3 target piles each ({r['no_target_rounds']} with none); flag = score above the "
          f"frozen threshold (95th percentile of cross-validated irregular scores). Chance is per pile: a coin that flags "
          f"q of piles flags q of targets and q of the rest; a fair coin gets {0.125:.1%} of rounds exactly right, and "
          f"never flagging gets the {r['no_target_rounds'] / r['rounds']:.0%} with no target")
    for who in ("dish", "software"):
        d = r[who]
        print(f"{who:8s} threshold {d['threshold']:.2f}: target piles flagged {d['hit']:.0%}; false flags on human rubble "
              f"taps {d['false_human']:.0%}, on debris trials {d['false_debris']:.0%}; rounds exactly right "
              f"{d['rounds_right_human']:.0%} (vs human taps), {d['rounds_right_debris']:.0%} (vs debris)")
