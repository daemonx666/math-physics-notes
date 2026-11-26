"""Reproduce the periodic-clock benchmark for Var(B_t)=t and a N(0,1) target.
Run with Python 3, using only its standard library.
"""

import math


def mean_time(period, cutoff, panels):
    """Simpson quadrature of the exact conditional renewal mean against |Z|."""
    step = cutoff / panels
    total = 0.0
    for i in range(1, panels + 1):
        x = i * step
        z = x / math.sqrt(2 * period)
        success = math.erfc(z)
        duration = (
            period * math.erf(z)
            + x * math.sqrt(2 * period / math.pi) * math.exp(-z * z)
            - x * x * success
        )
        density = math.sqrt(2 / math.pi) * math.exp(-0.5 * x * x)
        weight = 1 if i == panels else (4 if i % 2 else 2)
        total += weight * density * duration / success
    return total * step / 3


def benchmark(cutoff, panels):
    # This bracket checks the published periodic benchmark. The analytic
    # theorem in the paper provides the comparison with arbitrary clocks.
    lo, hi = 1.6, 5.0
    for _ in range(50):
        left = (2 * lo + hi) / 3
        right = (lo + 2 * hi) / 3
        if mean_time(left, cutoff, panels) < mean_time(right, cutoff, panels):
            hi = right
        else:
            lo = left
    period = (lo + hi) / 2
    return period, mean_time(period, cutoff, panels)


if __name__ == "__main__":
    for cutoff, panels in [(18, 1600), (22, 3200)]:
        period, value = benchmark(cutoff, panels)
        print(f"cutoff={cutoff} panels={panels}: t={period:.7f}, J={value:.9f}")
        if abs(period - 2.81592) > 1e-5 or abs(value - 3.34972) > 1e-6:
            raise SystemExit("Periodic benchmark differs from the quoted precision")
