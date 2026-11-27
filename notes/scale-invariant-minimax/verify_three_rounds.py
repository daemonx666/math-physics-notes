"""Three-round Brownian restart: full parameter exclusion and local shape comparison.

Default: python-flint and a C++17 compiler with 128-bit integers.
Optional search: NumPy/SciPy and obj() in verify_interleaved.py.
"""
from fractions import Fraction as F
from flint import arb,ctx
from pathlib import Path
from time import perf_counter
import subprocess

N=1024
K=8192
LOG=1<<26
EXP=4096
def ball(f):
    f=F(f)
    return arb(f.numerator)/f.denominator
def floor(x):
    return int(x.lower().floor().unique_fmpz())
def ceil(x):
    return int(x.upper().ceil().unique_fmpz())
def endpoint_table():
    ctx.prec=160
    small_b=ball(F(1,512)).exp()
    small=((small_b+1)/(6*small_b**2*(small_b-1)*(1+small_b**(ball(F(2,3)))+small_b**(ball(F(4,3)))))).sqrt()/2
    survival=1-arb(-2).exp()/(1-(-ball(F(64,3))).exp())
    coeff=(2/arb.pi()).sqrt()*ball("-.5").exp()
    large=survival*coeff*(ball(F(64,6))-ball("1.5")).exp()/64
    assert floor(10**6*small)>3540000 and floor(10**6*large)>3540000
    delta=arb(-48).exp()/(1-(-ball(F(1,512))).exp())
    assert delta<ball(".5") and 2*delta*LOG<1
    rows=[]
    for j in range(-48*N,96*N+1):
        u=ball(F(j,N)).exp()
        z=1/(2*u).sqrt()
        s=z.erf()
        a=u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
        rows.append(f"{max(0,ceil(-LOG*s.log()))} {max(0,min(60*K,floor(K*a)))}")
    rows.extend(str(floor(K*ball(F(-j,EXP)).exp())) for j in range(24*EXP+1))
    rows.append("0")
    return "\n".join(rows)+"\n"

def verify(binary):
    start=perf_counter()
    res=subprocess.run([binary],input=endpoint_table(),text=True,check=True,stdout=subprocess.PIPE)
    boxes,visits,queries,num,den=map(int,res.stdout.split())
    lower=F(num,den)
    assert lower>F("3.54")
    print("Boxes, visited, cached integrals:",boxes,visits,queries)
    print("Smallest central lower bound:",lower)
    print("Seconds:",round(perf_counter()-start,2),flush=True)

def tail_bounds():
    """Uniform strip and differentiated truncation bounds for rounds -128..48."""
    hmin,hmax=ball(".47520"),ball(".47530")
    R=ball(F(1,16))
    degree_bound=1/R.cos().sqrt()
    # Complex log size disks around the real shape box |eps_i|<=.001.
    # Their real shifts have magnitude below .1.
    spread=ball(".1")
    E=(3*hmax+spread-129*hmin).exp()/(1-(-hmin).exp())
    Fmax=2*degree_bound**177*sum(((n*(hmin if n<0 else hmax)+3*hmax+spread).exp() for n in range(-128,49)),arb(0))
    last_product=degree_bound**138
    for n in range(10,49):
        last_product*=(-(n*hmin-spread)/2).exp()
    ratio=(hmax-(49*hmin-spread)/2).exp()
    tail=2*last_product*(52*hmax+spread).exp()/(1-ratio)
    error=(2*E).exp()*(2*E+tail)+(2*E).expm1()*Fmax
    second_error=2*error/R**2
    assert R+ball(".001")<spread and degree_bound**3<2 and ratio<1 and second_error<ball(".000001")
    strip=(2*E).exp()*(2*E+Fmax+tail)
    return hmin,hmax,R,E,tail,strip,second_error

def local_shape():
    hmin,hmax,_,_,_,_,second_error=tail_bounds()
    h=hmin.union(hmax)
    eps=arb(0,ball(".001").upper())
    total=[arb(0),arb(0),arb(0)]
    cells=512
    root=arb.pi().sqrt()
    for j in range(cells):
        phase=h*3*ball(F(j,cells)).union(ball(F(j+1,cells)))
        V=D1=D2=A=B=C=arb(0)
        for n in range(48,-129,-1):
            axis=n%3
            v=n*h+phase+(eps if axis else 0)
            u=v.exp(); z=(-v/2).exp()/arb(2).sqrt()
            f=(-z*z).exp();s=z.erf()
            a=u*s+(2*u/arb.pi()).sqrt()*f-z.erfc()
            ds=-z*f/root; dd=ds*(z*z-ball(".5"))
            A,B,C=s*A,s*B,s*C
            old1,old2=D1,D2
            D1,D2=s*D1,s*D2
            if axis==1:
                A+=2*ds*old1+u*(s+ds)+dd*V
                B+=ds*old2;D1+=u*s+ds*V
            elif axis==2:
                C+=2*ds*old2+u*(s+ds)+dd*V
                B+=ds*old1;D2+=u*s+ds*V
            V=a+s*V
        for i,val in enumerate((A,B,C)):
            total[i]+=val
    A,B,C=[v/cells+arb(0,second_error.upper()) for v in total]
    margin=min((A-abs(B)).lower(),(C-abs(B)).lower())
    print("local Hessian",A,B,C,"margin",margin,"Cauchy error",second_error,flush=True)
    assert floor(10**6*margin)>60000

def mean_at(h,offsets):
    hmin,hmax,R,E,tail,strip,_=tail_bounds()
    assert hmin<h and h<hmax and all(abs(e)<ball(".001") for e in offsets)
    nodes=384
    quad=2*strip/(2*arb.pi()*R*nodes/(3*h)).expm1()
    logt=[n*h+offsets[n%3] for n in range(-128,49)]
    total=arb(0)
    for j in range(nodes):
        phase=3*h*j/nodes; value=arb(0)
        for t in reversed(logt):
            u=(phase+t).exp()
            z=1/(2*u).sqrt(); s=z.erf()
            a=u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
            value=a+s*value
        total+=value
    finite=total/nodes
    lo=(1-E)*finite-quad
    hi=E+finite+tail+quad
    # At each phase the legal policy removes early rounds of total duration
    # at most E*x0^2, so its normalized cost is <= ideal/(1-E).
    return lo.lower(),hi.upper(),(hi/(1-E)).upper()

def search():
    import itertools
    import numpy as np
    from scipy.optimize import minimize
    from verify_interleaved import obj
    # Each seed includes all six orders of the same three logarithmic sizes.
    rows=[]
    for g in (.33,1.42,3.5,8.0):
        for shape in ((0,.2,1.1),(0,1.5,5.),(-7.,0,4.)):
            for order in itertools.permutations(shape):
                h=g/3
                x0=np.array([h,order[1]-order[0],order[2]-order[0]-2*h])
                opt=minimize(obj,x0,args=(3,48),jac=True,method="L-BFGS-B",
                             bounds=[(.08,4),(-16,16),(-16,16)],
                             options={"ftol":2e-15,"gtol":1e-10,"maxiter":180})
                value=obj(opt.x,3,160)[0]
                rows.append((value,opt.x))
    value,p=min(rows,key=lambda row:row[0])
    print("starts",len(rows),"best float",value,"h,d1,offset2",p,"log sizes",0,p[1],2*p[0]+p[2],flush=True)


def compare():
    ctx.prec=176
    local_shape()
    offsets=[0,ball("1e-8"),-ball("1e-8")]
    reference=mean_at(ball(".47525052"),[0]*3)
    geometric=mean_at(ball(".475250513"),[0]*3)
    unequal=mean_at(ball(".475250513"),offsets)
    scale=10**23
    def bounds(row):
        return F(floor(row[0]*scale),scale),F(ceil(row[2]*scale),scale)
    old_lo,old_hi=bounds(reference)
    geom_lo,geom_hi=bounds(geometric)
    uns_lo,uns_hi=bounds(unequal)
    assert old_lo-uns_hi>F("1.39e-16")
    assert uns_lo-geom_hi>F("2.56e-17")
    assert geom_hi<F("3.577824965832792861") and uns_hi<F("3.577824965832792886")
    for label,row in zip(("previous geometric","same-rate geometric","unequal"),(reference,geometric,unequal)):
        print(label,"ideal lower",row[0],"admissible upper",row[2],flush=True)

if __name__=="__main__":
    import argparse
    from tempfile import TemporaryDirectory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search",action="store_true",help="run exploratory floating-point optimization")
    args=parser.parse_args()
    if not __debug__: raise RuntimeError("Run with assertions enabled")
    start=perf_counter()
    if args.search: search()
    else:
        compare()
        with TemporaryDirectory() as temp:
            executable=Path(temp)/"three_rounds"
            source=Path(__file__).resolve().with_name("three_rounds.cpp")
            subprocess.run(["c++","-std=c++17","-O3",str(source),"-o",str(executable)],check=True)
            verify(executable)
    print("Total seconds:",round(perf_counter()-start,2))

