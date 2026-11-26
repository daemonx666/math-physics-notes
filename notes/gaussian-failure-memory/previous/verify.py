"""Validate the enclosed certificates with python-flint 0.9.0 (Arb, 110 bits).
Usage: python verify.py [certificate.json]
"""
import json
import sys
from pathlib import Path

from flint import acb, arb, ctx

ctx.prec = 110
PI = arb.pi()


def dyadic(value):
    numerator, denominator = float.fromhex(value).as_integer_ratio()
    return arb(numerator) / denominator


def attempt(x, t):
    z = x / (2 * t).sqrt()
    success = z.erfc()
    duration = t * z.erf() + x * (2 * t / PI).sqrt() * (-z*z).exp() - x*x*success
    return duration, success


def validate_lower(c):
    x = [arb(j) / c["xd"] for j in c["xi"]]
    mass = [arb(j) / c["wd"] for j in c["wi"]]
    values = [[arb(j) / c["ld"] for j in level] for level in c["levels"]]
    times = [dyadic(s) for s in c["times"]]
    thresholds = [dyadic(s) for s in c["thresholds"]]
    n = len(x)
    assert len(mass) == n and all(len(level) == n for level in values)
    assert len(values) == len(thresholds) + 1
    assert times[0] > 0 and all(a < b for a, b in zip(times, times[1:]))
    assert all(times[0] < b < times[-1] for b in thresholds)
    # Each threshold must be an exact grid edge.
    switches = [c["times"].index(s) for s in c["thresholds"]] + [len(times)]
    weights_by_stage = [mass]
    for threshold in thresholds:
        weights_by_stage.append([
            w * (v / (2 * threshold).sqrt()).erf()
            for w, v in zip(weights_by_stage[-1], x)
        ])
    for i, v in enumerate(x):
        end = x[i+1] if i+1 < n else arb(c["last_right"]) / c["xd"]
        assert 0 < v < end and mass[i] > 0
        exact_mass = (v / arb(2).sqrt()).erfc() - (end / arb(2).sqrt()).erfc()
        assert mass[i] < exact_mass
        tiny_success = attempt(v, times[0])[1]
        long_duration = attempt(v, times[-1])[0]
        assert v*v >= 2*times[0]
        for level in values:
            assert level[i] >= 0
            assert times[0]*(1-tiny_success) > level[i]*tiny_success
            assert long_duration > level[i]

    durations = [attempt(v, times[0])[0] for v in x]
    for j, end in enumerate(times[1:]):
        right = [attempt(v, end) for v in x]
        for k, (level, weights) in enumerate(zip(values, weights_by_stage)):
            margin = arb(0)
            if j >= switches[k]:
                next_level = values[k+1]
                for i in reversed(range(n)):
                    margin += weights[i] * (
                        durations[i] + (1-right[i][1])*next_level[i] - level[i]
                    )
                    assert margin >= 0, (k, j, i, margin)
            else:
                for i in reversed(range(n)):
                    margin += weights[i]*(durations[i] - right[i][1]*level[i])
                    assert margin >= 0, (k, j, i, margin)
        durations = [item[0] for item in right]
    return sum(w*l for w, l in zip(mass, values[0]))


def schedule_enclosure(ticks, stop=12):
    times = [arb(t) / 10000 for t in ticks]
    assert all(t > 0 for t in times) and times[-1] > 1
    tail = times[-1]
    density_scale = (2/PI).sqrt()
    constants = [(t, 1/(2*t).sqrt(), (2*t/PI).sqrt()) for t in times]

    def integrand(x, analytic):
        # Meromorphic on each complex integration ball. All square roots are constants.
        x2 = x*x
        cost = 0
        for i in reversed(range(len(times))):
            t, alpha, beta = constants[i]
            z = x*alpha
            survival, success = z.erf(), z.erfc()
            duration = t*survival + x*beta*(-z*z).exp() - x2*success
            if i == len(times)-1:
                cost = duration/success
            else:
                cost = duration + survival*cost
        return density_scale*(-x2/2).exp()*cost

    bounded_part = sum(acb.integral(integrand, i, i+1, abs_tol=1e-13,
                                    rel_tol=1e-13).real for i in range(stop))
    b = (1-1/tail)/2
    tail_bound = (sum(times[:-1])*(arb(stop)/arb(2).sqrt()).erfc()
                  + tail*(1/tail.sqrt()+tail.sqrt()/stop**2)
                  * (-b*stop**2).exp()/(2*b))
    assert bounded_part.is_finite() and tail_bound.is_finite()
    return bounded_part.lower(), (bounded_part+tail_bound).upper()


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O: certificate checks use assert.")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("certificate.json")
    bundle = json.loads(path.read_text())
    for name, ticks in bundle.get("schedules", {}).items():
        print(name, schedule_enclosure(ticks), flush=True)
    certificate = bundle.get("lower", bundle)
    print("Certified global lower bound:", validate_lower(certificate), flush=True)
