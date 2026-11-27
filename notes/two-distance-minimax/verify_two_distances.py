"""Two-distance Brownian restart certificate. Requires python-flint (tested with 0.9.0).

Run: python3 verify_two_distances.py
Only the generation of the proposed polygon used a floating-point optimizer.
All acceptance checks below use integers, fractions or outward-rounded Arb balls.
"""
from fractions import Fraction
from functools import cache
from flint import arb, ctx

ctx.prec = 144
N, P, K = 200, 10**7, 10**8
RATIO = Fraction(25, 8)
ORACLE_LO = Fraction(267177, 100000)
ORACLE_HI = Fraction(2671772, 1000000)
# Polygon value at odds i/N, in units 1/P. Nodes were proposed by gridded
# Bellman iteration, discounted by 4 parts per million, then rounded down.
V = [
    26717606, 26978076, 27236792, 27493642, 27748506, 28001744, 28253025, 28502602,
    28750592, 28996599, 29241158, 29484016, 29725290, 29965152, 30203143, 30439836,
    30674904, 30908536, 31140819, 31371513, 31601037, 31828803, 32055370, 32280529,
    32504391, 32726981, 32948105, 33168126, 33386608, 33604097, 33820065, 34035057,
    34248429, 34460771, 34671728, 34881686, 35090124, 35297576, 35503511, 35708465,
    35909060, 36100769, 36291643, 36481917, 36671293, 36860115, 37048333, 37235774,
    37422623, 37608789, 37794567, 37979497, 38164043, 38348048, 38531404, 38714290,
    38896547, 39078375, 39259546, 39440224, 39614372, 39787408, 39960000, 40132269,
    40304137, 40475649, 40646943, 40817862, 40988439, 41158745, 41328639, 41498293,
    41667645, 41836750, 42005630, 42174077, 42342315, 42510400, 42678136, 42845631,
    43012887, 43179839, 43346584, 43513062, 43679288, 43845314, 44011037, 44174248,
    44335162, 44495780, 44656261, 44816608, 44976687, 45136626, 45296415, 45456052,
    45615564, 45774901, 45934034, 46093054, 46251935, 46410588, 46569124, 46727532,
    46885776, 47043917, 47201920, 47359807, 47517576, 47675114, 47832541, 47989896,
    48147153, 48304257, 48461243, 48618085, 48774818, 48931456, 49087914, 49244277,
    49400558, 49556679, 49712689, 49868611, 50024410, 50180079,
]
SEQ_A = [1002, 1048, 1145, 4040, 6419, 7839, 8509, 8784, 8891, 8931]
SEQ_B = [947, 958, 976, 1007, 4993, 7030, 8143, 8637, 8834, 8910, 8938]


def ball(value):
    f = Fraction(value)
    return arb(f.numerator) / f.denominator


def kernel(u):
    t = ball(u)
    z = 1 / (2*t).sqrt()
    s, h = z.erf(), z.erfc()
    a = t*s + (2*t/arb.pi()).sqrt()*(-z*z).exp() - h
    return s, h, a


def down(value):
    return int((K*value).lower().floor().unique_fmpz())


def up(value):
    return int((K*value).upper().ceil().unique_fmpz())


@cache
def bounds(t):
    s, h, a = kernel(t)
    S, _, A = kernel(t/RATIO**2)
    return down(s), down(S), down(a), down(A), up(h)


def verify_polygon():
    end = len(V)-1
    differences = [v1-v0 for v0, v1 in zip(V, V[1:])]
    assert end == 125 and all(a >= b for a, b in zip(differences, differences[1:]))
    for j, slope in enumerate(differences):
        # Both coefficients of each homogeneous supporting line lie in [0,16].
        assert 0 <= slope*N <= 16*P and 0 <= V[j]-j*slope <= 16*P
    small, large = Fraction(1,16), Fraction(256)
    # For u <= 1/16, h(u)/u <= sqrt(2/(pi*u))*exp(-1/(2u)),
    # increasing on this interval. Hence a(u) >= u*s(u) >= u/2 >= 16*h(u).
    u = ball(small)
    assert (2/(arb.pi()*u)).sqrt()*(-1/(2*u)).exp() < ball(Fraction(1,32))
    assert kernel(large/RATIO**2)[2] > ball(Fraction(max(V),P))
    assert kernel(large)[2] > ball(ORACLE_LO)
    stack = [(small, large, list(range(end+1)), True)]
    accepted = 0
    while stack:
        lo, hi, nodes, oracle = stack.pop()
        _, _, a, A, _ = bounds(lo)
        s, S, _, _, h = bounds(hi)
        assert 0 < S and 0 <= s <= S
        pending = []
        for i in nodes:
            # Evaluate the homogeneous polygon at down-rounded survival weights.
            j, rem = divmod(i*s, S)
            continuation = S*V[j] + (rem*differences[j] if rem else 0)
            if P*(i*a + N*A) + N*continuation < N*K*V[i]:
                pending.append(i)
        check_oracle = oracle and a*ORACLE_LO.denominator < h*ORACLE_LO.numerator
        if pending or check_oracle:
            assert hi-lo > Fraction(1,2**45), "certificate too close to resolve"
            mid = (lo+hi)/2
            stack.extend([(lo, mid, pending, check_oracle), (mid, hi, pending, check_oracle)])
        else:
            accepted += 1
    return Fraction(V[-1]*N,P*(N+end)), accepted


def policy_cost(seq, x):
    survive, total = arb(1), arb(0)
    for period in seq:
        s, _, a = kernel(Fraction(period,1000)/x**2)
        total += survive*a
        survive *= s
    _, h, a = kernel(Fraction(8955,1000)/x**2)
    return total + survive*a/h


def main():
    lower, leaves = verify_polygon()
    _, h, a = kernel(Fraction(917,1000))
    assert a/h < ball(ORACLE_HI)
    print("Checked time intervals:", leaves)
    print("Exact two-target lower bound:", lower)
    print("Certified lower gap from oracle:", lower-ORACLE_HI)
    prob = Fraction(59221,100000)
    for x in [Fraction(1), RATIO]:
        cost = ball(prob)*policy_cost(SEQ_A,x) + ball(1-prob)*policy_cost(SEQ_B,x)
        assert cost < ball(Fraction(3088081,1000000))
        print("Normalized cost at distance", x, ":", cost)
    print("Known-distance optimum enclosed by", ORACLE_LO, "and", ORACLE_HI)


if __name__ == "__main__":
    main()

