"""Three-distance Brownian restart lower bound and fixed-triple upper witness.

Run with Python assertions enabled. The default verification uses only the
saved integer array, NumPy and python-flint. --build also requires SciPy.
"""
import argparse
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from flint import arb, ctx
import numpy as np
def propose(n, imax, jmax, distances, ticks):
    """Floating-point candidates only. verify() proves every required inequality."""
    from scipy.special import erf, erfc
    time = np.geomspace(0.075, 3*float(distances[-1])**2, 3001)
    def kernels(x):
        u = time / float(x)**2
        z = 1/np.sqrt(2*u)
        s = erf(z)
        return s, u*s + np.sqrt(2*u/np.pi)*np.exp(-z*z) - erfc(z)
    (s, a), (v, b), (S, A) = map(kernels, distances)
    values = np.zeros((imax+2, jmax+2))
    for level in range(imax+jmax+1):
        i = np.arange(max(0,level-jmax),min(imax,level)+1)[:,None]
        j = level-i
        x, y = np.minimum(i*s/S,i), np.minimum(j*v/S,j)
        k, l = x.astype(int), y.astype(int)
        f, g = x-k, y-l
        cost = (i*a+j*b)/n + A
        self_weight = np.zeros_like(cost)
        for di, dj, mass in [(0,0,1-np.maximum(f,g)),(1,0,np.maximum(f-g,0)),
                             (0,1,np.maximum(g-f,0)),(1,1,np.minimum(f,g))]:
            cost += S*mass*values[k+di,l+dj]
            self_weight += ((k+di==i)&(l+dj==j))*mass
        den = 1-S*self_weight
        candidate = np.full_like(cost,np.inf)
        np.divide(cost,den,out=candidate,where=den>0)
        values[i[:,0],j[:,0]] = candidate.min(axis=1)
    return np.floor(values*ticks*(1-0.000085)).astype(np.int64)


DISTANCES = (Fraction(1), Fraction(63,20), Fraction(143,16))
N, P, K = 256, 10**6, 10**8
IMAX, JMAX = 140, 296

def ball(value):
    return arb(value.numerator)/value.denominator

def floor_tick(value):
    return int((K*value).lower().floor().unique_fmpz())

def exact_kernel(t, x):
    u = ball(t/x**2)
    z = 1/(2*u).sqrt()
    s, h = z.erf(), z.erfc()
    return s, h, u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-h

@lru_cache(maxsize=2048)
def enclosure(t):
    survival, cost = [], []
    for x in DISTANCES:
        s, _, a = exact_kernel(t, x)
        survival.append(floor_tick(s))
        cost.append(floor_tick(a))
    return survival, cost

def verify(values):
    ctx.prec = 144
    lo, hi = Fraction(1,32), Fraction(4096)
    cap = 10
    assert values.dtype == np.int64 and values.shape == (IMAX+2,JMAX+2)
    assert np.all(values[-1,:] == 0) and np.all(values[:,-1] == 0)
    real_nodes = values[:IMAX+1,:JMAX+1]
    assert real_nodes.min() >= 0 and real_nodes.max() <= cap*P
    # Small times: min survival is at the nearest target. The far-target
    # cost alone covers cap * (1 - min survival).
    u = ball(lo)
    tail_ratio = (2/(arb.pi()*u)).sqrt()*(-1/(2*u)).exp()
    assert cap*tail_ratio < 1/(2*ball(DISTANCES[-1]**2))
    assert tail_ratio*u < arb(1)/2
    max_a = [int((K*exact_kernel(hi,x)[2]).upper().ceil().unique_fmpz()) for x in DISTANCES]
    assert enclosure(hi)[1][2] > cap*K
    # All vector operations below are exact signed int64. These bounds also
    # cover the padded zero nodes. Costs and survival are nonnegative.
    assert max(IMAX,JMAX)*K < 2**63
    assert P*(IMAX*max_a[0]+JMAX*max_a[1]+N*max_a[2])+N*K*cap*P < 2**63
    assert max(IMAX,JMAX)*K+K < 2**63
    on_grid = np.indices(real_nodes.shape,dtype=np.int64).reshape(2,-1)
    stack = [(lo,hi,*on_grid)]
    leaves = checks = 0
    while stack:
        left,right,i,j = stack.pop()
        a,b,A = enclosure(left)[1]
        (s,v,S),_ = enclosure(right)
        assert 0 <= s <= v <= S <= K and S > 0 and min(a,b,A)>=0
        assert all(cost <= bound for cost,bound in zip((a,b,A),max_a))
        k,f = np.divmod(i*s,S)
        l,g = np.divmod(j*v,S)
        # Integer barycentric masses of the lower survival vector. Split
        # each odds rectangle along its bottom-left to top-right diagonal.
        cont = ((S-np.maximum(f,g))*values[k,l]
                +np.maximum(f-g,0)*values[k+1,l]
                +np.maximum(g-f,0)*values[k,l+1]
                +np.minimum(f,g)*values[k+1,l+1])
        bad = P*(i*a+j*b+N*A) + N*cont < N*K*values[i,j]
        checks += len(i)
        if np.any(bad):
            assert right-left > Fraction(1,2**42), "candidate not certified"
            mid = (left+right)/2
            i,j = i[bad],j[bad]
            stack.extend([(left,mid,i,j),(mid,right,i,j)])
        else:
            leaves += 1
    return leaves, checks


# Each prefix is in thousandths of x0 squared. After this prefix the
# deadline is 73.25 times x0 squared in every subsequent round.
POLICIES = [
    (12143, [1047,1137,4249,6778,8645,10234,14399,35230,51753,62726,68651,71398,72528,72813]),
    (168956, [960,972,1003,5191,7506,9169,10940,22172,41543,56420,65493,69737,71679,72528]),
    (818901, [953,968,999,5211,7506,9242,11201,21913,41380,56199,65236,69737,71679,72528]),
]

def check_upper():
    assert sum(p for p,_ in POLICIES) == 10**6
    for x in DISTANCES:
        total = arb(0)
        for p, prefix in POLICIES:
            _,h,a = exact_kernel(Fraction(293,4),x)
            cost = a/h
            for period in reversed(prefix):
                s,_,a = exact_kernel(Fraction(period,1000),x)
                cost = a+s*cost
            total += p*cost/10**6
        assert total < ball(Fraction(3262947,1000000))
        print("Explicit mixture cost at", x, ":", total)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="generate a new nodes.npz (must not exist)")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Run with Python assertions enabled")
    path = Path(__file__).with_name("nodes.npz")
    if args.build:
        proposed = propose(N,IMAX,JMAX,DISTANCES,P)
        with path.open("xb") as out:
            np.savez_compressed(out, nodes=proposed)
    with np.load(path,allow_pickle=False) as data:
        values = data["nodes"]
    leaves, checks = verify(values)
    lower, i, j = max((Fraction(int(values[i,j])*N,P*(N+i+j)),i,j)
                      for i in range(IMAX+1) for j in range(JMAX+1))
    assert lower >= Fraction(8410849,2578125)
    print("Distances:", DISTANCES, "weight ratios:", (i,j,N))
    print("Certified lower:", lower, float(lower))
    print("Improvement:", lower-Fraction(50180079,16250000))
    print("Gap to 3.61:", Fraction(361,100)-lower)
    print("Time intervals:", leaves, "node-interval checks:", checks)
    check_upper()

if __name__ == "__main__":
    main()

