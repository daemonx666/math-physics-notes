"""Compute a four-distance certificate using outward-rounded Brownian kernels.

Use standard Python, python-flint and a C++17 compiler. Optional distance
search (--search) uses NumPy/SciPy; its floating-point scores select candidates.
"""
import argparse
from contextlib import nullcontext
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
import subprocess
import tempfile

from flint import arb, ctx

ROOT = Path(__file__).resolve().parent
P, K, CAP = 100_000, 3_000_000, 20
SEARCH_K = 10_000_000
FIRST, LAST = -6, 19
PREVIOUS = Fraction(8410849,2578125)
REFERENCE = Fraction(361,100)

def ball(x):
    return arb(x.numerator)/x.denominator

def down(x):
    return int((x*K).lower().floor().unique_fmpz())

def endpoints(parts):
    for o in range(FIRST,LAST):
        scale = Fraction(2)**o
        for j in range(parts):
            yield scale*Fraction(parts+j,parts)
    yield Fraction(2)**LAST

def kernel(t, distance):
    u = ball(t/distance**2)
    z = 1/(2*u).sqrt()
    s, h = z.erf(), z.erfc()
    return s, u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-h

def exact_table(xs, parts):
    ctx.prec = 144
    start = ball(Fraction(2)**FIRST)
    tail = (2/(arb.pi()*start)).sqrt()*(-1/(2*start)).exp()
    assert CAP*tail < 1/(2*ball(xs[-1]**2)) and tail*start < arb(1)/2
    assert kernel(Fraction(2)**LAST,xs[-1])[1] > CAP
    rows = []
    for t in endpoints(parts):
        values = [kernel(t,x) for x in xs]
        rows.append(" ".join(str(down(v[c])) for c in (0,1) for v in values))
    return "\n".join(rows)

@lru_cache(maxsize=128)
def floating_column(x,parts):
    import numpy as np
    from scipy.special import erf, erfc
    t = np.fromiter(map(float,endpoints(parts)),dtype=float)/float(x)**2
    z = 1/np.sqrt(2*t)
    s = erf(z)
    a = t*s+np.sqrt(2*t/np.pi)*np.exp(-z*z)-erfc(z)
    return np.floor(SEARCH_K*s).astype(np.int64),np.floor(SEARCH_K*a).astype(np.int64)

def float_table(xs,parts):
    import numpy as np
    from io import StringIO
    cols = [floating_column(x,parts) for x in xs]
    buffer = StringIO()
    np.savetxt(buffer,np.column_stack([v[c] for c in (0,1) for v in cols]),fmt="%d")
    return buffer.getvalue()

def run(binary,xs,d,limits,parts,rigorous):
    rows = exact_table(xs,parts) if rigorous else float_table(xs,parts)
    params = (d,*limits,LAST-FIRST,parts,P,K if rigorous else SEARCH_K,CAP)
    data = " ".join(map(str,params))+"\n"+rows+"\n"
    command = [binary,"--audit"] if rigorous else [binary]
    cmd = subprocess.run(command,input=data,capture_output=True,text=True)
    if cmd.returncode:
        raise RuntimeError(f"Verifier exited {cmd.returncode}: {cmd.stderr}")
    num,den,i,j,k,nodes,checks,audit = map(int,cmd.stdout.split())
    return Fraction(num,den),(i*i,j*j,k*k,d*d),nodes,checks,audit

def profile(binary):
    def sample(triples,d,limits,parts):
        rows=[]
        for a,b,c in triples:
            xs=tuple(map(Fraction,(1,a,b,c)))
            lower,weights,*_=run(binary,xs,d,limits,parts,False)
            rows.append((float(lower),tuple(map(float,xs)),weights))
        print("Top three candidates on grid",d,":",sorted(rows,reverse=True)[:3],flush=True)

    def ratios(a,b,c):
        return ((x,x*y,x*y*z) for x,y,z in product(a,b,c))

    sample(ratios((2.75,3.125,3.5),(2.5,2.875,3.25),(2.5,2.875,3.25)),10,(10,12,14),64)
    sample(ratios((3.0625,3.1875,3.3125),(2.75,2.875,3),(2.5,2.6875,2.875)),18,(17,22,25),128)
    sample([(3.125,8.875,24),(3.125,9,24.75),(3.125,9.25,25.25),(3.1875,9,24.5),
            (3.1875,9.125,25),(3.1875,9.25,25.5),(3.1875,9.375,25.5),(3.25,9.125,24.75),
            (3.25,9.25,25.25),(3.25,9.375,25.5),(3.25,9.5,26),(3.3125,9.375,25.25),
            (3.3125,9.5,25.75)],32,(27,38,43),256)
    sample([(3.1875,9.125,24.875),(3.21875,9.125,25),(3.21875,9.25,25.25),
            (3.25,9.25,25.25),(3.25,9.375,25.5),(3.28125,9.375,25.5)],48,(35,53,62),512)

# Deadline prefixes are in thousandths of the squared nearest distance.
MIXTURE = [
    (276580,[951,963,982,5754,8349,10742,24290,46028,63741,78007,92718,136966,270299,397555,488506,539934]),
    (170712,[1032,1098,4624,7401,9603,12932,35936,55475,71758,86642,115267,233940,367341,469202,529900]),
    (375635,[944,953,965,5727,8280,10512,21539,43991,62225,77361,93699,151306,289694,413236,498111,544795]),
    (177073,[930,934,937,5508,7990,9945,14439,39323,58262,73943,88927,124360,252957,383497,479698,535407]),
]

def check_upper(xs):
    assert sum(w for w,_ in MIXTURE)==10**6
    for x in xs:
        total=arb(0)
        for w,periods in MIXTURE:
            s,a=kernel(Fraction(584645,1000),x)
            value=a/(1-s)
            for t in reversed(periods):
                s,a=kernel(Fraction(t,1000),x)
                value=a+s*value
            total+=w*value/10**6
        assert total<ball(Fraction(3356404,10**6))
        print("Upper witness at",x,":",total)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search",action="store_true")
    parser.add_argument("--binary",type=Path,help="reuse a locally compiled verifier")
    args=parser.parse_args()
    if not __debug__:
        raise RuntimeError("Python assertions must be enabled")
    with tempfile.TemporaryDirectory() if args.binary is None else nullcontext(None) as tmp:
        binary=args.binary
        if binary is None:
            binary=Path(tmp)/"verify"
            subprocess.run(["c++","-O3","-std=c++17",str(ROOT/"grid.cpp"),"-o",str(binary)],check=True)
        if args.search:
            profile(binary)
            return
        xs=tuple(map(Fraction,("1","3.25","9.25","25.25")))
        lower,weights,nodes,checks,audit=run(binary,xs,120,(86,135,154),2048,True)
        assert lower>Fraction(3353,1000) and audit>0
        print("distances",xs,"weights",weights,"nodes",nodes,"construction tests",checks,"audit tests",audit)
        print("lower",lower,float(lower),"gain",lower-PREVIOUS,"gap",REFERENCE-lower)
        check_upper(xs)

if __name__=="__main__":
    main()

