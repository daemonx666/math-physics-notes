"""Rebuild the numerical witnesses on the supplied finite histories and time grid.
Floating-point construction is exploratory. Use verify.py for certification.
"""

import json
import gzip
import argparse
import numpy as np
from scipy.special import erf, erfc

SCALE = 10**11
ORACLE = 2.67


def needed(current, desired):
    j = len(current) - 1
    need = 0.0
    for b in reversed(desired):
        if j >= 0 and current[j] >= b:
            j -= 1
        elif need == 0.0:
            need = b
        else:
            return np.inf
    return need


def rebuild(c, free_depth=0, only_ceiling=False):
    n = c["n"]
    refs = [tuple(v) for v in c["refs"]]
    tags = sorted({v for st in refs for v in st})
    x = np.exp2(-6 + 9 * np.arange(n) / n)
    y = x * 2 ** (9 / n)
    f = np.sqrt(2 / np.pi) * np.exp(-x * x / 2)
    g = np.sqrt(2 / np.pi) * np.exp(-y * y / 2)
    w = x * g
    moment = (x * f - y * g + erfc(x / np.sqrt(2)) - erfc(y / np.sqrt(2))) / x**2
    ts = np.array([float.fromhex(t) for t in c["times"]])
    ends = np.column_stack((ts[:-1], ts[1:])).reshape(-1, 1)
    l, r = ts[:-1, None], ts[1:, None]
    z = x / np.sqrt(2 * ends)
    q = erfc(z)
    s = erf(z)
    dur = ends * s + x * np.sqrt(2 * ends / np.pi) * np.exp(-z * z) - x * x * q
    err = np.repeat(
        3
        / (16 * np.sqrt(2 * np.pi))
        * (r / l - 1) ** 2
        * x
        / np.sqrt(l)
        * np.exp(-x * x / (2 * r)),
        2,
        axis=0,
    )
    masks = {v: erf(x / np.sqrt(2 * v)) for v in tags}
    oracle = ORACLE * x * x
    routes = c["routes"]

    def solve(ceiling):
        vals = np.zeros((len(refs), n))
        witness = np.zeros((len(refs), n), dtype=int)
        correction = 0.0 if ceiling else err
        terminal = {}
        # Feasible terminal costs for an unrestricted continuation. The policy
        # can choose its period given the reference suffix, before the target is sampled.
        periods = np.exp2(np.arange(-3, 41) / 6)
        zz = x / np.sqrt(2 * periods[:, None])
        tq = erfc(zz)
        tc = (
            periods[:, None] * erf(zz)
            + x * np.sqrt(2 * periods[:, None] / np.pi) * np.exp(-zz * zz)
            - x * x * tq
        ) / tq
        loss = q + correction
        for k in reversed(range(len(refs))):
            wk = w.copy()
            for tag in refs[k]:
                wk *= masks[tag]
            if ceiling and free_depth and len(refs[k]) >= free_depth:
                sums = np.cumsum((wk * tc)[:, ::-1], axis=1)[:, ::-1]
                best = sums.argmin(axis=0)
                cost = sums[best, np.arange(n)]
                vals[k] = np.maximum(0.0, (cost - np.r_[cost[1:], 0.0]) / wk)
                terminal[str(k)] = best.tolist()
                continue
            # Each segment has its own list of candidate continuations. Expand in columns.
            width = max(len(v) for _, v in routes[k])
            g = np.full((len(ts) - 1, width), k)
            for j, options in routes[k]:
                if width:
                    g[j:] = k
                for u, m in enumerate(options):
                    g[j:, u] = m
            # All arithmetic below is exploratory; the verifier independently checks the certificate.
            imm = [dur]
            losses = [loss]
            for col in g.T:
                dest = np.repeat(col, 2)
                req = np.array([needed(refs[k], refs[m]) for m in np.unique(col)])
                unique, inv = np.unique(col, return_inverse=True)
                required = req[inv]
                if not np.isfinite(required).all():
                    raise ArithmeticError("invalid route")
                short = np.repeat(ts[:-1] < required, 2)
                me = vals[dest]
                fixed = np.zeros_like(me)
                for m, v in zip(unique, req):
                    if v > 0:
                        fixed[dest == m] = masks[v]
                charge = np.where(
                    short[:, None], fixed * (me - oracle) + s * oracle, s * me
                )
                cc = dur + charge - correction * np.where(short[:, None], oracle, me)
                idle = dest == k
                cc[idle] = dur[idle]
                imm.append(cc)
                losses.append(np.where(idle[:, None], loss, 1.0))
            residual = np.zeros((len(imm), len(ends)))
            for j in reversed(range(n)):
                rhs = np.array(
                    [
                        np.divide(
                            mat[:, j] + residual[o] / wk[j],
                            losses[o][:, j],
                            out=np.full(len(ends), np.inf),
                            where=losses[o][:, j] > 1e-210,
                        )
                        for o, mat in enumerate(imm)
                    ]
                )
                scores = (
                    rhs.max(0)
                    if ceiling
                    else rhs.reshape(len(imm), -1, 2).min(2).max(0)
                )
                action = scores.argmin()
                val = scores[action]
                if not np.isfinite(val) or val < -0.0001:
                    raise ArithmeticError((k, j, val))
                val = max(val, 0.0)
                vals[k, j] = val
                witness[k, j] = action
                for o, mat in enumerate(imm):
                    residual[o] += wk[j] * (mat[:, j] - losses[o][:, j] * val)
        val = np.rint(vals * SCALE * (1.000002 if ceiling else 0.999998)).astype(
            np.int64
        )
        return val, witness, float(moment @ (val[0] / SCALE)), terminal

    high, act, uv, term = solve(True)
    if only_ceiling:
        return {
            "depth": free_depth,
            "values": high.tolist(),
            "witness": act.tolist(),
            "terminal": term,
        }, uv
    low, _, lv, _ = solve(False)
    return {
        **c,
        "lower": low.tolist(),
        "upper": high.tolist(),
        "witness": act.tolist(),
        "scale": SCALE,
    }, (lv, uv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("out")
    p.add_argument("--free-depth", type=int, default=0)
    args = p.parse_args()
    op = gzip.open if args.source.endswith(".gz") else open
    with op(args.source, "rt") as f:
        c = json.load(f)
    original = c
    if "certificates" in c:
        c = c["certificates"]["main"]
    ret, summary = rebuild(c, args.free_depth, bool(args.free_depth))
    if args.free_depth:
        ret = {
            "certificates": {"main": c},
            "late_ceiling": ret,
            "schedule": original["schedule"],
        }
    with open(args.out, "x") as f:
        json.dump(ret, f, separators=(",", ":"))
    print(summary, flush=True)
