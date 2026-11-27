"""Certify randomized log-periodic search and five-distance Brownian lower bounds."""
from fractions import Fraction as F
from pathlib import Path
import subprocess
from flint import arb, ctx
ctx.prec=144
P,K,CAP=100000,1000000,24
FIRST,LAST=-6,23
def ball(x):
    x=F(x)
    return arb(x.numerator)/x.denominator
def kernels(u):
    z=1/(2*u).sqrt()
    s=z.erf()
    return s,u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
def table(xs,parts,rigorous):
    endpoints=[F(2)**o*F(parts+j,parts) for o in range(FIRST,LAST) for j in range(parts)]+[F(2)**LAST]
    if rigorous:
        start=ball(endpoints[0])
        rate=(2/(arb.pi()*start)).sqrt()*(-1/(2*start)).exp()
        assert CAP*rate<1/(2*ball(xs[-1]**2)) and rate*start<ball(F(1,2))
        assert kernels(ball(endpoints[-1]/xs[-1]**2))[1]>CAP
        rows=[]
        for t in endpoints:
            pairs=[kernels(ball(t/x**2)) for x in xs]
            rows.append(" ".join(str(int((K*p[c]).lower().floor().unique_fmpz())) for c in (0,1) for p in pairs))
        return "\n".join(rows)
    import numpy as np
    from scipy.special import erf,erfc
    from io import StringIO
    u=np.array(endpoints,dtype=float)[:,None]/np.array(xs,dtype=float)**2
    z=1/np.sqrt(2*u);s=erf(z)
    a=u*s+np.sqrt(2*u/np.pi)*np.exp(-z*z)-erfc(z)
    out=StringIO();np.savetxt(out,np.floor(K*np.column_stack((s,a))).astype(np.int64),fmt="%d")
    return out.getvalue()
def lower(binary,xs,D,limits,parts,rigorous):
    xs=tuple(map(F,xs))
    assert xs[0]==1 and list(xs)==sorted(xs)
    args=" ".join(map(str,(D,*limits,LAST-FIRST,parts,P,K,CAP)))
    data=args+"\n"+table(xs,parts,rigorous)+"\n"+str(int(rigorous))+"\n"
    res=subprocess.run([binary],input=data,capture_output=True,text=True)
    if res.returncode:raise RuntimeError(res.stderr)
    num,den,i,j,k,l,equal,tests,audit=map(int,res.stdout.split())
    if rigorous: assert audit>0
    return F(num,den),(i*i,j*j,k*k,l*l,D*D),F(equal,5*P),tests,audit

def upper(parts=32768):
    # Distance x ranges from 1 to sqrt(b); each cell has the same log mass.
    b=ball(F(193,120))
    first,last=-44,20
    first_t=b**first
    prefix=first_t/(b-1)
    lost=b*prefix   # all removed t/x^2 when x >= x0
    assert lost<1
    # Endpoint kernels are reused by the two adjacent distance cells.
    def endpoint(j):
        u=first_t*(-b.log()*j/parts).exp()
        survival=[];cost=[]
        for _ in range(first,last+1):
            s,a=kernels(u)
            survival.append(s);cost.append(a);u*=b
        return survival,cost
    survival_tail=kernels(b**last)[0]
    ratio=b*survival_tail
    assert ratio<1
    tail=b**(last+1)/(1-ratio)
    # Two directed endpoint envelopes, summed in uniform log-distance.
    previous_s,cost=endpoint(0)
    total=arb(0); lower_sum=arb(0)
    for j in range(1,parts+1):
        survival,next_cost=endpoint(j)
        p=arb(1);value=prefix
        for a,s in zip(cost,survival):
            value+=p*a;p*=s
        total+=value+p*tail
        p=1-prefix; cell_low=arb(0)
        for a,s in zip(next_cost,previous_s):
            cell_low+=p*a;p*=s
        lower_sum+=cell_low
        previous_s,cost=survival,next_cost
    result=total/parts/(1-lost)
    print("all-distance upper enclosure of the envelope:",result,flush=True)
    lower_bound=lower_sum/parts-lost
    scale=10**9
    strict_lo=F(int((scale*lower_bound).lower().floor().unique_fmpz()),scale)
    strict_hi=F(int((scale*result).upper().ceil().unique_fmpz()),scale)
    assert strict_lo>F(35777,10000) and strict_hi<F(357787,100000)
    print("lower envelope for the same policy:",lower_bound,flush=True)
    return result

def search(binary):
    from itertools import product
    def rank(candidates,D,limits,parts):
        rows=[]
        for xs in candidates:
            val,w,*_=lower(binary,xs,D,limits,parts,False)
            rows.append((float(val),xs,w))
        print(sorted(rows,reverse=True)[:4],flush=True)
        return max(rows)[1]
    candidates=[]
    for a,b,c,d in product((3,3.375),(2.75,3),(2.625,2.875),(2.5,2.75)):
        candidates.append((1,a,a*b,a*b*c,a*b*c*d))
    rank(candidates,8,(8,10,11,12),64)
    fine=[(1,a,b,c,d) for a,b,c,d in [(3.125,9,25,68),(3.25,9.25,25.5,68),(3.25,9.5,26.5,72),(3.25,9.5,27,74),(3.375,9.75,27,73),(3.375,10,28,76),(3.125,9.25,26,70),(3.25,9.5,26,69)]]
    rank(fine,15,(13,18,20,22),128)
    rank([(1,3.25,9.25,25.5,68),(1,3.25,9.5,26,69),(1,3.25,9.375,26,70),(1,3.1875,9.125,25.25,67.5)],20,(15,24,28,29),256)

def avg(par):
    import numpy as np
    from scipy.special import erf,erfc
    L=par[0];b=np.r_[0,par[1:]]
    if L<.15 or L>9 or np.max(np.abs(b))>10:return 100+abs(L)
    m=len(b);phase=(np.arange(20)+.5)*L/20
    j=np.arange(np.floor((-18-max(b))/L),np.ceil((12-min(b))/L)+1)
    logs=(j[:,None]*L+b).ravel()
    u=np.exp(logs[:,None]+phase);z=1/np.sqrt(2*u);s=erf(z)
    a=u*s+np.sqrt(2*u/np.pi)*np.exp(-z*z)-erfc(z)
    return np.sum(a*np.vstack([np.ones(len(phase)),np.cumprod(s[:-1],axis=0)]),axis=0).mean()
def search_periods():
    import numpy as np
    from scipy.optimize import minimize
    for m in range(1,9):
        rows=[]
        for mode in range(3):
            start=np.r_[m*.46,np.arange(1,m)*.46*(1 if mode==0 else 0.1 if mode==1 else -0.3)]
            sol=minimize(avg,start,method="BFGS",options={"gtol":1e-7,"maxiter":90})
            rows.append((sol.fun,sol.x))
        print(m,*min(rows,key=lambda row:row[0]),flush=True)
    rng=np.random.default_rng(29)
    for m in (3,5,8,12):
        rows=[]
        for _ in range(8):
            L=m*rng.uniform(.4,.9)
            y=np.clip(np.arange(m)*L/m+rng.normal(0,1.6,m),-5,9)
            v=np.r_[min(8,L),y[1:]-y[0]]
            sol=minimize(avg,v,method="L-BFGS-B",bounds=[(.2,9)]+[(-10,10)]*(m-1),
                         options={"maxiter":100,"ftol":5e-12})
            rows.append((sol.fun,sol.x))
        print("random starts",m,*min(rows,key=lambda row:row[0]),flush=True)

def sparse_window():
    # Round the five support points upwards on a multiplicative grid.
    q=F(1001,1000)
    offsets=[]
    for d in (1,F(13,4),F(19,2),26,69):
        lo,hi=-1,8192
        while hi-lo>1:
            m=(hi+lo)//2
            if q**m>=d: hi=m
            else: lo=m
        assert q**hi>=d and (hi==0 or q**(hi-1)<d)
        offsets.append(hi)
    n=10**6
    edge=F(offsets[-1],n)
    witness=(1-edge)*F(15756304,4621875)/q**2+edge*F(267177,100000)
    assert witness>F(3399168,10**6)
    print("Million-point uniform prior:",offsets,witness,flush=True)

def main():
    import argparse
    from tempfile import TemporaryDirectory
    from time import perf_counter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=["upper","five","window"])
    parser.add_argument("--search",action="store_true",help="exploratory floating-point comparisons")
    args=parser.parse_args()
    if not __debug__: raise RuntimeError("Run without -O: verification requires assertions")
    start=perf_counter()
    if not args.search and args.only in (None,"upper"):
        upper()
    if args.only=="upper" and not args.search: return
    with TemporaryDirectory() as tmp:
        binary=Path(tmp)/"five_grid"
        src=Path(__file__).resolve().with_name("five_grid.cpp")
        subprocess.run(["c++","-O3","-std=c++17",str(src),"-o",str(binary)],check=True)
        if args.search:
            search(binary);search_periods()
            return
        if args.only in (None,"five"):
            result=lower(binary,(1,F(13,4),F(19,2),26,69),32,(23,37,44,45),1024,True)
            value,weights,_,tests,audit=result
            assert value==F(15756304,4621875) and weights==(441,1089,1681,1681,1024)
            print("Five-distance lower:",value,"weights:",weights,"construction and audit:",tests,audit,flush=True)
            print("gain:",value-F(1314788,391875),"gap to 3.61:",F(361,100)-value,
                  "certified gap:",F(357787,100000)-value,flush=True)
        if args.only in (None,"window"):
            value,weights,equal,_,_=lower(binary,tuple(F(23,8)**i for i in range(5)),17,(17,21,24,24),256,True)
            assert value==F(551123,162500) and equal==F(167527,50000)
            n=200
            window=(n-4)*value/n+F(4,n)*F(267177,100000)
            assert window>F(3377,1000)
            print("Sliding window:",value,weights,"equal five:",equal,"uniform 200:",window,flush=True)
            sparse_window()
    print("Verification seconds:",round(perf_counter()-start,2))

if __name__=="__main__":
    main()

