"""Independent outward-interval verifier for history certificates and their model ceilings.
Requires python-flint and NumPy. Supports plain JSON and gzip JSON bundles.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from flint import acb, arb, ctx

ctx.prec = 144
PI = arb.pi()
INF = np.inf


def exact_float(v):
    f = float.fromhex(v) if isinstance(v, str) else float(v)
    a, b = f.as_integer_ratio()
    return arb(a) / b


def bounds(ball):
    # One binary64 ULP outwards also covers rounding the Arb endpoint to nearest.
    if not ball.is_finite():
        raise ArithmeticError("nonfinite input enclosure")
    return (
        math.nextafter(float(ball.lower()), -math.inf),
        math.nextafter(float(ball.upper()), math.inf),
    )


def separate(balls):
    lo, hi = zip(*(bounds(v) for v in balls))
    return np.asarray(lo), np.asarray(hi)


def down(x):
    return np.nextafter(x, -INF)


def up(x):
    return np.nextafter(x, INF)


def suffix_endpoint(terms, lower):
    """A conservative endpoint for each suffix sum of these binary64 entries."""
    rev = terms[:, ::-1]
    n = terms.shape[1]
    sums = np.cumsum(rev, axis=1)[:, ::-1]
    magnitude = np.cumsum(np.abs(rev), axis=1)[:, ::-1]
    # 8*n*eps dominates both the addition error and rounding of the sum of
    # absolute values; the final term covers absolute underflow errors.
    allowance = up(up(magnitude * (8 * n * np.finfo(float).eps)) + n * 1e-320)
    result = down(sums - allowance) if lower else up(sums + allowance)
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite suffix sums")
    return result


def required(current, desired):
    # Cover the largest wanted tags first; at most one can be supplied by
    # the new attempt. Its minimum necessary time is the unmatched tag.
    need = 0.0
    j = len(current) - 1
    for b in reversed(desired):
        if j >= 0 and current[j] >= b:
            j -= 1
        elif need == 0.0:
            need = b
        else:
            return math.inf
    return need


def certify_oracle():
    # Uniformly a_1(t) >= 2.67 q_1(t), hence a known distance x costs at
    # least 2.67*x*x for ANY sequence by Brownian scaling and telescoping.
    target = arb(267) / 100

    def aq(t):
        z = 1 / (2 * t).sqrt()
        return t * z.erf() + (2 * t / PI).sqrt() * (-z * z).exp() - z.erfc(), z.erfc()

    l, r = arb(1) / 32, arb(64)
    assert 2 * l <= 1 and l * (1 - aq(l)[1]) > target * aq(l)[1]
    assert aq(r)[0] > target
    pending = [(l, r)]
    while pending:
        a, b = pending.pop()
        if aq(a)[0] > target * aq(b)[1]:
            continue
        assert b - a > arb(1) / 10**10
        mid = (a + b) / 2
        pending.extend(((a, mid), (mid, b)))


def verify(c, early=None):
    if (
        np.finfo(float).nmant != 52
        or np.nextafter(0.0, 1.0) + np.nextafter(0.0, 1.0) == 0.0
    ):
        raise RuntimeError("IEEE binary64 with gradual underflow is required")
    certify_oracle()
    n = c["n"]
    assert 2 <= n <= 10000
    ts = [exact_float(t) for t in c["times"]]
    raw_times = [float.fromhex(t) for t in c["times"]]
    assert 0 < ts[0] < ts[-1] and all(a < b for a, b in zip(ts, ts[1:]))
    histories = c["refs"]
    assert histories[0] == [] and all(
        st == sorted(st) and all(t > 0 and math.isfinite(t) for t in st)
        for st in histories
    )
    states = len(histories)
    cells = len(ts) - 1
    assert len(c["routes"]) == states
    # Reference w is for the far edge of a geometric cell. Revealing the
    # common scale u of the target gives the integral objective moment.
    ratio = (arb(2).log() * 9 / n).exp()
    nodes = [(arb(2).log() * (arb(-6) + arb(9 * i) / n)).exp() for i in range(n)]
    norm = (2 / PI).sqrt()
    weights, moments = [], []
    for x in nodes:
        y = x * ratio
        px = norm * (-x * x / 2).exp()
        py = norm * (-y * y / 2).exp()
        weights.append(x * py)
        cell_moment = (
            x * px - y * py + (x / arb(2).sqrt()).erfc() - (y / arb(2).sqrt()).erfc()
        ) / (x * x)
        assert cell_moment > 0
        moments.append(cell_moment)
    mask = {
        t: [(x / (2 * exact_float(t)).sqrt()).erf() for x in nodes]
        for t in {v for h in histories for v in h}
    }
    value_levels = []
    input_sets = [c["lower"], c["upper"]]
    if early is not None:
        input_sets.append(early["values"])
    for rows in input_sets:
        assert len(rows) == states and c["scale"] > 0
        arr = np.zeros((2, states, n))
        for k, row in enumerate(rows):
            assert len(row) == n and all(isinstance(v, int) and v >= 0 for v in row)
            lo, hi = separate([arb(v) / c["scale"] for v in row])
            arr[0, k] = np.maximum(0.0, lo)
            arr[1, k] = hi
        value_levels.append(arr)
    L = value_levels[0]
    time_index = {t: i for i, t in enumerate(raw_times)}
    a_lo, a_hi, q_lo, q_hi, s_lo, s_hi = [np.empty((len(ts), n)) for _ in range(6)]
    for j, t in enumerate(ts):
        root = (2 * t).sqrt()
        beta = (2 * t / PI).sqrt()
        for i, x in enumerate(nodes):
            z = x / root
            q = z.erfc()
            surv = z.erf()
            aa = t * surv + x * beta * (-z * z).exp() - x * x * q
            a_lo[j, i], a_hi[j, i] = bounds(aa)
            q_lo[j, i], q_hi[j, i] = bounds(q)
            s_lo[j, i], s_hi[j, i] = bounds(surv)
    q_lo = np.maximum(q_lo, 0.0)
    s_lo = np.maximum(s_lo, 0.0)
    # Uniform curvature reserve on each cell for the lower comparison.
    error = np.empty((cells, n))
    for j, (left, right) in enumerate(zip(ts, ts[1:])):
        prefactor = (
            arb(3) / (16 * (2 * PI).sqrt()) * (right / left - 1) ** 2 / left.sqrt()
        )
        for i, x in enumerate(nodes):
            error[j, i] = bounds(prefactor * x * (-x * x / (2 * right)).exp())[1]
    ends = np.column_stack((np.arange(cells), np.arange(1, cells + 1))).ravel()
    reserve = np.repeat(error, 2, axis=0)
    q_safe = up(q_hi[ends] + reserve)
    s_safe = down(s_lo[ends] - reserve)
    assert (s_safe >= 0).all()
    if early is not None:
        assert isinstance(early["depth"], int) and early["depth"] > 0
        feasible_hi = []
        for idx in range(44):
            t = (arb(2).log() * (idx - 3) / 6).exp()
            sig = (2 * t).sqrt()
            beta = (2 * t / PI).sqrt()
            costs = []
            for x in nodes:
                z = x / sig
                q = z.erfc()
                costs.append(
                    bounds((t * z.erf() + x * beta * (-z * z).exp() - x * x * q) / q)[1]
                )
            feasible_hi.append(costs)
        feasible_hi = np.array(feasible_hi)
    for k, st in enumerate(histories):
        assert all(x * x >= 2 * ts[0] for x in nodes)
        llo, lhi = L[:, k]
        assert (
            down(float(ts[0].lower()) * down(1 - q_hi[0])) > up(lhi * q_hi[0])
        ).all()
        assert (a_lo[-1] > lhi).all()
        weights_k = weights.copy()
        for v in st:
            weights_k = [w * h for w, h in zip(weights_k, mask[v])]
        wlo, whi = separate(weights_k)
        assert min(wlo) > 0
        routes = c["routes"][k]
        assert (
            routes
            and routes[0][0] == 0
            and all(p[0] < r[0] for p, r in zip(routes, routes[1:]))
        )
        assert routes[-1][0] < cells and all(isinstance(j, int) for j, _ in routes)
        # The option to stay is always available. A destination m>k can
        # be used early: only the mass surviving max(t,b) is kept there.
        count = 1 + max(len(options) for _, options in routes)
        grid = np.full((cells, count), k, dtype=int)
        for start, options in routes:
            assert all(k < nxt < states for nxt in options)
            grid[start:] = k
            for col, nxt in enumerate(options, 1):
                grid[start:, col] = nxt
        following = np.repeat(grid, 2, axis=0)
        all_needed = {int(m): required(st, histories[m]) for m in np.unique(grid)}
        assert all(math.isfinite(b) and b >= 0 for b in all_needed.values())
        # Every clipping boundary must be a time-grid edge.
        ref_lo = {}
        ref_hi = {}
        for m, b in all_needed.items():
            if b:
                assert b in mask and b in time_index
                blo, bhi = separate(mask[b])
                ref_lo[m] = np.maximum(blo, 0.0)
                ref_hi[m] = bhi
        oracle_lo, oracle_hi = separate([arb(267) * x * x / 100 for x in nodes])

        def residual(indices, destinations, lower, vals):
            groups = indices // 2
            ep = ends[indices]
            stay = destinations == k
            if lower:
                res = down(
                    down(a_lo[ep] + down(s_safe[indices] * vals[0, destinations]))
                    - vals[1, k]
                )
                res[stay] = down(
                    a_lo[ep[stay]] - up(q_safe[indices][stay] * vals[1, k])
                )
            else:
                res = up(
                    up(a_hi[ep] + up(s_hi[ep] * vals[1, destinations])) - vals[0, k]
                )
                res[stay] = up(a_hi[ep[stay]] - down(q_lo[ep[stay]] * vals[0, k]))
            for m in np.unique(destinations[~stay]):
                b = all_needed[int(m)]
                if not b:
                    continue
                short = (destinations == m) & (groups < time_index[b])
                if not short.any():
                    continue
                ee = ep[short]
                if lower:
                    # Combine the curvature reserve with s*O, retaining signs
                    # for the oracle residual s-s_b (the reserve can make it negative).
                    coef = down(s_safe[indices[short]] - ref_hi[m])
                    oracle_term = down(coef * np.where(coef >= 0, oracle_lo, oracle_hi))
                    kept = down(ref_lo[m] * vals[0, m])
                    res[short] = down(
                        down(down(a_lo[ee] + kept) + oracle_term) - vals[1, k]
                    )
                else:
                    coef = up(s_hi[ee] - ref_lo[m])
                    oracle_term = up(coef * np.where(coef >= 0, oracle_hi, oracle_lo))
                    kept = up(ref_hi[m] * vals[1, m])
                    res[short] = up(up(up(a_hi[ee] + kept) + oracle_term) - vals[0, k])
            if lower:
                terms = down(res * np.where(res >= 0, wlo, whi))
            else:
                terms = up(res * np.where(res >= 0, whi, wlo))
            return suffix_endpoint(terms, lower)

        for lo in range(0, 2 * cells, 256):
            rows = np.arange(lo, min(lo + 256, 2 * cells))
            best = None
            for dest in np.unique(following[rows].T, axis=0):
                check = residual(rows, dest, True, L).reshape(-1, 2, n).min(axis=1)
                best = check if best is None else np.maximum(best, check)
            if best.min() < 0:
                j, i = np.unravel_index(best.argmin(), best.shape)
                raise ArithmeticError(("lower", k, lo // 2 + j, i, best[j, i]))
        for proof_id, U in enumerate(value_levels[1:]):
            proof = c if proof_id == 0 else early
            if proof_id == 1 and len(st) >= early["depth"]:
                choice = np.asarray(early["terminal"][str(k)], dtype=int)
                assert (
                    choice.shape == (n,) and (choice >= 0).all() and (choice < 44).all()
                )
                # Compare with a feasible periodic continuation separately on each suffix.
                res = up(feasible_hi[choice] - U[0, k])
                terms = up(res * np.where(res >= 0, whi, wlo))
                check = np.diag(suffix_endpoint(terms, False))
                if check.max() > 0:
                    raise ArithmeticError(("terminal", k, check.max()))
                continue
            witness = np.array(proof["witness"][k], dtype=int)
            assert (
                witness.shape == (n,)
                and (witness >= 0).all()
                and (witness < 2 * cells).all()
            )
            for dest in np.unique(following[witness].T, axis=0):
                check = np.diag(residual(witness, dest, False, U))
                if check.max() > 0:
                    j = check.argmax()
                    raise ArithmeticError(("model ceiling", k, j, witness[j], check[j]))
    a = sum(w * arb(v) / c["scale"] for w, v in zip(moments, c["lower"][0]))
    b = sum(w * arb(v) / c["scale"] for w, v in zip(moments, c["upper"][0]))
    assert a <= b
    if early is None:
        return a, b
    limit = sum(w * arb(v) / c["scale"] for w, v in zip(moments, early["values"][0]))
    return a, b, limit


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def schedule_enclosure(ticks, stop=12):
    times = [arb(t) / 10000 for t in ticks]
    assert all(t > 0 for t in times) and times[-1] > 1
    tail = times[-1]
    density_scale = (2 / PI).sqrt()
    constants = [(t, 1 / (2 * t).sqrt(), (2 * t / PI).sqrt()) for t in times]

    def integrand(x, analytic):
        # Meromorphic on each complex integration ball. All square roots are constants.
        x2 = x * x
        cost = 0
        for i in reversed(range(len(times))):
            t, alpha, beta = constants[i]
            z = x * alpha
            survival, success = z.erf(), z.erfc()
            duration = t * survival + x * beta * (-z * z).exp() - x2 * success
            if i == len(times) - 1:
                cost = duration / success
            else:
                cost = duration + survival * cost
        return density_scale * (-x2 / 2).exp() * cost

    bounded_part = sum(
        acb.integral(integrand, i, i + 1, abs_tol=1e-13, rel_tol=1e-13).real
        for i in range(stop)
    )
    b = (1 - 1 / tail) / 2
    tail_bound = sum(times[:-1]) * (arb(stop) / arb(2).sqrt()).erfc() + tail * (
        1 / tail.sqrt() + tail.sqrt() / stop**2
    ) * (-b * stop**2).exp() / (2 * b)
    assert bounded_part.is_finite() and tail_bound.is_finite()
    return bounded_part.lower(), (bounded_part + tail_bound).upper()


def prefix_ceiling(ticks, K, upper):
    # After K agreed attempts grant the conditional lower cost 2.67*x^2.
    factors = []
    for v in ticks[:K]:
        t = arb(v) / 10000
        factors.append((t, 1 / (2 * t).sqrt(), (2 * t / PI).sqrt()))

    def integrand(x, analytic):
        xx = x * x
        cost = arb(267) / 100 * xx
        for t, alpha, beta in reversed(factors):
            z = x * alpha
            surv = z.erf()
            cost = t * surv + x * beta * (-z * z).exp() - xx * z.erfc() + surv * cost
        return (2 / PI).sqrt() * (-xx / 2).exp() * cost

    part = sum(
        acb.integral(integrand, j, j + 1, rel_tol=1e-12, abs_tol=1e-12).real
        for j in range(12)
    )
    assert part.is_finite()
    return (upper - part.lower()).upper()


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("certificates.json.gz"),
    )
    parser.add_argument(
        "--case", help="Verify one named case instead of the entire bundle"
    )
    args = parser.parse_args()
    bundle = read(args.file)
    if "certificates" not in bundle:
        print(verify(bundle), flush=True)
    else:
        cases = bundle["certificates"]
        if args.case and args.case not in cases:
            raise SystemExit("Unknown case")
        for name, c in cases.items():
            if args.case and name != args.case:
                continue
            print(
                name,
                verify(c, bundle.get("late_ceiling") if name == "main" else None),
                flush=True,
            )
        if not args.case or args.case == "main":
            ticks = bundle["schedule"]
            lo, hi = schedule_enclosure(ticks)
            print("Feasible schedule (global upper):", lo, hi, flush=True)
            for K in (8, 16, 20, 28):
                print(
                    "Fixed prefix",
                    K,
                    "maximum tail improvement",
                    prefix_ceiling(ticks, K, hi),
                    flush=True,
                )
