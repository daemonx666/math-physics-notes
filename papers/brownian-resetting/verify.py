"""Check the optimum via conditional exponential clocks, in units L = D = 1."""

import math


def renewal_integrand(u, b):
    """Target-averaged attempt duration / success, with u = log(1 + x).

    Mixing (1 - exp(-k*x))/k**2 against Gamma(b, 1) gives an attempt
    duration whose derivative in x is (1 + x)**(1-b)/(b-1).
    Integrate this derivative from zero and multiply by p(x)*dx/q(x).
    """
    factor = (2 * b - 1) * math.exp(-(b - 1) * u)
    if b == 2:
        return factor * u
    return factor * (-math.expm1(-(b - 2) * u)) / ((b - 1) * (b - 2))


def integrate(b, stop=100.0, intervals=16000):
    """Composite Simpson rule on a finite interval for the four cases below."""
    step = stop / intervals
    total = renewal_integrand(0, b) + renewal_integrand(stop, b)
    for i in range(1, intervals):
        weight = 4 if i % 2 else 2
        total += weight * renewal_integrand(i * step, b)
    return total * step / 3


def main():
    print("b      renewal integral    theorem value     absolute error")
    for b in (1.75, 2.0, 2.5, 4.0):
        expected = (2 * b - 1) / ((b - 1) ** 2 * (2 * b - 3))
        observed = integrate(b)
        error = abs(observed - expected)
        print(f"{b:4.2f}   {observed:15.10f}  {expected:15.10f}  {error:.2e}")
        if not math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-9):
            raise SystemExit("Numerical cross-check failed")


if __name__ == "__main__":
    main()
