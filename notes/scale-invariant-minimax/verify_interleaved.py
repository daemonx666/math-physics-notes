"""Interval certificates for interleaved Brownian-reset deadlines.
Default: rigorously check policy bounds and two restricted-family exclusions.
--search: reproduce the numerical search (SciPy), with no proof assertions.
"""
from fractions import Fraction as F
from flint import arb, arb_series, ctx
ctx.prec=160
ORDER=8
ctx.cap=ORDER+1

def ball(value):
    f=F(value)
    return arb(f.numerator)/f.denominator

# h is a change of log phase. a'(h)=u*s(h), s'(h)=-z*exp(-z^2)/sqrt(pi).
h=arb_series([0,1],prec=ORDER+1)
eh=h.exp()
emh=(-h).exp()
half=(-h/2).exp()
root_pi=arb.pi().sqrt()
def kernel_series(logu):
    u=logu.exp()
    z=(-logu/2).exp()/arb(2).sqrt()
    s0=z.erf()
    a0=u*s0+u.sqrt()*(2/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
    ds=(-z/root_pi)*half*(-z*z*emh).exp()
    s=arb_series([s0]+[ds[k]/(k+1) for k in range(ORDER)],prec=ORDER+1)
    da=u*eh*s
    a=arb_series([a0]+[da[k]/(k+1) for k in range(ORDER)],prec=ORDER+1)
    return s,a

def enclose(log_period,offsets):
    """Enclose ideal phase mean and the uniformly truncated legal policy."""
    L=log_period
    b=L.exp();c_sum=sum(v.exp() for v in offsets)
    left=0
    while b*(left*L).exp()*c_sum/(b-1)>ball("1e-21"):
        left-=1
    eta=b*(left*L).exp()*c_sum/(b-1)
    # Verify the chosen negative cutoff separately, including near-equality.
    assert eta<ball("2e-21")
    right=0
    survival=arb(1)
    while True:
        for v in offsets:
            z=(-(right*L+v)/2).exp()/arb(2).sqrt()
            survival*=z.erf()
        largest=max(offsets)  # One attempt suffices to bound a whole block's survival.
        q=b*((-((right+1)*L+largest)/2).exp()/arb(2).sqrt()).erf()
        if q<1:
            tail=survival*b*((right+1)*L).exp()*c_sum/(1-q)
            if tail<ball("1e-21"): break
        right+=1
        if right>250: raise ArithmeticError("positive tail not certified")
    logs=[j*L+v for j in range(left,right+1) for v in offsets]
    def series_at(p):
        value=arb_series([0],prec=ORDER+1)
        for logt in reversed(logs):
            s,a=kernel_series(logt+p)
            value=a+s*value
        return value
    cells=1
    while L/cells>ball("0.06"): cells*=2
    radius=L/(2*cells)
    value=arb(0)
    for i in range(cells):
        mid=(2*i+1)*radius
        center=series_at(mid)
        enclosure=series_at(mid+arb(0,radius.upper()))
        polynomial=sum(center[k]*radius**k/(k+1) for k in range(0,ORDER,2))
        error=abs(enclosure[ORDER])*radius**ORDER/(ORDER+1)
        value+=polynomial+arb(0,error.upper())
    value/=cells
    low=(1-eta)*value
    high=eta+value+tail
    return low.lower(),high.upper(),(high/(1-eta)).upper(),left,right

def permutation_bound(log_r,m=10,bins=256):
    """Lower bound for every nonidentity permutation of (1,r,...,r^(m-1))."""
    import numpy as np
    K=10**6
    start,end=-48,20
    r=log_r.exp()
    count=(end-start)*bins
    # One sample per endpoint; all brackets use exact log-ratio indices.
    s=[];a=[]
    for k in range(start*bins,(end+m)*bins+1):
        u=(log_r*k/bins).exp()
        z=1/(2*u).sqrt()
        surv=z.erf()
        cost=u*surv+(2*u/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
        s.append(max(0,int((K*surv).lower().floor().unique_fmpz())))
        a.append(max(0,int((K*cost).lower().floor().unique_fmpz())))
    # On [kh,(k+1)h], costs increase and all survival factors decrease.
    sf=np.array([s[1+j*bins:1+j*bins+count] for j in range(m)],dtype=np.int64)
    cf=np.array([a[j*bins:j*bins+count] for j in range(m)],dtype=np.int64)
    cf=np.minimum(cf,250*K)  # capping a lower integrand remains a lower bound
    # At the first octave's right endpoints the omitted earlier sum of
    # geometric deadlines is at most r^(start+1)/(r-1).
    pre=np.empty(count,dtype=np.int64)
    pre[:bins]=int((K*(1-r**(start+1)/(r-1))).lower().floor().unique_fmpz())
    assert 0<=pre[0]<=K and 0<=sf.min() and sf.max()<=K
    for j in range(bins,count,bins):
        pre[j:j+bins]=(pre[j-bins:j]*sf[0,j-bins:j])//K
    assert K*K<2**63 and int(cf.min())>=0 and K*int(cf.max())*count<2**63
    # Precompute one scalar cost per directed subset edge. The order used
    # to round products down does not affect validity for any order of play.
    full=(1<<m)-1
    edges={}
    def visit(mask,first_index,reached):
        for j in range(m):
            if not mask&(1<<j):
                edges[mask,j]=int(np.dot(reached,cf[j]))
        for j in range(first_index,m):
            if (mask|(1<<j))!=full:
                visit(mask|(1<<j),j+1,(reached*sf[j])//K)
    visit(0,0,pre)
    dp={(0,False):0}
    for mask in range(full):
        for different in (False,True):
            if (mask,different) not in dp:continue
            base=dp[mask,different]
            for j in range(m):
                if mask&(1<<j):continue
                key=(mask|(1<<j),different or j!=mask.bit_count())
                val=base+edges[mask,j]
                if key not in dp or val<dp[key]:dp[key]=val
    return F(dp[full,True],K*K*bins*m)

def exclude_two_rounds():
    """Cover B>1 and 0<q<=sqrt(B): rotating and rescaling covers all two-round blocks."""
    import numpy as np
    from functools import lru_cache
    n=512; K=10**6; log_scale=2**24; exp_scale=512
    g0=ball(F(1,n)); b0=g0.exp()
    small=1/(2*arb(6).sqrt()*b0*(b0-1).sqrt())
    assert small>ball("3.54")
    # For g>=32 integrate over [g/2-3,g/2-2].
    survival=1-arb(-2).exp()/(1-arb(-16).exp())
    coeff=(2/arb.pi()).sqrt()*arb("-0.5").exp()
    assert survival*coeff*arb("6.5").exp()/32>ball("3.54")
    start=-43*n; end=34*n
    def floor_int(x):
        return int(x.lower().floor().unique_fmpz())
    def ceil_int(x):
        return int(x.upper().ceil().unique_fmpz())
    neglog=[];sf=[];cost=[]
    for j in range(start,end+1):
        u=ball(F(j,n)).exp()
        z=1/(2*u).sqrt()
        s=z.erf()
        # -log(s)>=0, even if the ball touches 1.
        neglog.append(max(0,ceil_int(-log_scale*s.log())))
        sf.append(max(0,floor_int(K*s)))
        a=u*s+(2*u/arb.pi()).sqrt()*(-z*z).exp()-z.erfc()
        cost.append(min(1000*K,max(0,floor_int(K*a))))
    sf=np.array(sf,dtype=np.int64);cost=np.array(cost,dtype=np.int64)
    neglog=np.array(neglog,dtype=np.int64)
    # Any previous factor with exponent below start is omitted. At the
    # first endpoint of each residue class the geometric sum of all its
    # earlier deadlines is at most the following delta.
    delta=arb(-43).exp()/(1-(-g0).exp())
    assert delta<ball(".5") and 2*delta*log_scale<1
    lookup=np.array([floor_int(K*ball(F(-k,exp_scale)).exp()) for k in range(24*exp_scale+1)]+[0],dtype=np.int64)
    assert int(lookup.min())>=0 and int(lookup.max())<=K and int(sf.max())<=K
    assert (int(neglog.max())*len(neglog)+1)*exp_scale+log_scale<2**63
    @lru_cache(maxsize=32)
    def prior(g):
        # Right endpoint cost of the omitted part is one neg-log tick.
        # The preceding survival at v is the product at v-g, v-2g, ...
        pad=(-len(neglog))%g
        grouped=np.pad(neglog,(0,pad)).reshape(-1,g)
        loss=np.ones_like(grouped)
        loss[1:]+=np.cumsum(grouped[:-1],axis=0,dtype=np.int64)
        idx=(loss.ravel()[:len(neglog)]*exp_scale+log_scale-1)//log_scale
        return lookup[np.minimum(idx,len(lookup)-1)]
    lo=-24*n-start;hi=18*n-start
    assert cost.max()<10**11 and 2*K*int(cost.max())<2**63
    assert (hi-lo)*2*int(cost.max())<2**63
    nbox=0
    min_lower=None
    def cover(gl,gr,dl,dr):
        nonlocal nbox,min_lower
        if dr is not None and 2*dr < -gr: return  # outside q<=sqrt(B)
        P=prior(gl)
        alive=(P[lo+1:hi+1]*P[lo+1-dl:hi+1-dl])//K
        stage=cost[lo:hi]
        if dr is not None:
            stage=stage+(sf[lo+1:hi+1]*cost[lo-dr:hi-dr])//K
        numer=int(((alive*stage)//K).sum())
        denom=gr*K
        if 100*numer>354*denom:
            nbox+=1
            value=F(numer,denom)
            min_lower=value if min_lower is None else min(min_lower,value)
            return
        if gr>gl+1 and (dr is None or dr<=dl+1 or (gr-gl)*n>=gl*(dr-dl)):
            mid=(gl+gr)//2
            cover(gl,mid,dl,dr);cover(mid,gr,dl,dr)
        elif dr is not None and dr>dl+1:
            mid=(dl+dr)//2
            cover(gl,gr,dl,mid);cover(gl,gr,mid,dr)
        else:
            raise ArithmeticError((gl,gr,dl,dr,numer/denom))
    g_edges=[2**i for i in range(15)]
    assert g_edges[0]==1 and g_edges[-1]==32*n
    ds=[-16*n,-8*n,-4*n,-2*n,-n,-n//2,0,n//2,n,2*n,4*n,8*n,16*n]
    for gl,gr in zip(g_edges,g_edges[1:]):
        for dl,dr in zip(ds,ds[1:]+[None]): cover(gl,gr,dl,dr)
    return nbox,min_lower

def obj(params,m,phase_count):
    import numpy as np
    from scipy.special import erf, erfc
    h=params[0]
    offsets=np.r_[0.,params[1]-h,params[2:]]
    # Include complete small and large tails to well below search precision.
    phase=(np.arange(phase_count)+.5)*m/phase_count
    first=int(np.floor((-39-offsets.max())/h-m))
    last=int(np.ceil((13-offsets.min())/h))
    n=np.arange(first,last+1)
    coordinate=n[:,None]+phase
    u=np.exp(coordinate*h+offsets[n%m,None])
    z=1/np.sqrt(2*u); s=erf(z); density=z*np.exp(-z*z)/np.sqrt(np.pi)
    a=u*s+np.sqrt(2*u/np.pi)*np.exp(-z*z)-erfc(z)
    reach=np.vstack((np.ones(phase_count),np.cumprod(s[:-1],axis=0)))
    cost=reach*a
    future=np.vstack((np.cumsum(cost[:0:-1],axis=0)[::-1],np.zeros(phase_count)))
    grad=reach*u*s - density*future/s
    dg=np.bincount(n%m,weights=grad.mean(axis=1),minlength=m)
    gradient=np.r_[(grad*coordinate).sum()/phase_count-dg[1],dg[1:]]
    return cost.sum()/phase_count,gradient
def search():
    import numpy as np
    from scipy.optimize import minimize
    rng=np.random.default_rng(271828)
    for m in (2,3,4,6,8,12,24,48):
        for forced in (False,True):
            rows=[]
            count=max(24,2*m)
            for style in range(5):
                h=(.45,.35,.6,.5,.475)[style]
                j=np.arange(m)
                if style==0: offset=np.zeros(m)
                elif style==1: offset=(-1.)**j*1.3
                elif style==2: offset=1.6*np.sin(2*np.pi*j/m)+rng.normal(0,.4,m)
                elif style==3: offset=np.where(j%4<2,1.8,-1.8)+rng.normal(0,.4,m)
                else: offset=rng.uniform(-4,4,m)
                offset-=offset[0]
                d=(np.clip(h+offset[1],-np.log(8),-np.log(2)) if forced else h+offset[1])
                initial=np.r_[h,d,offset[2:]]
                limits=[(.22,.95),((-np.log(8),-np.log(2)) if forced else (-8,8))]+[(-12,12)]*(m-2)
                fit=minimize(obj,initial,args=(m,count),jac=True,method='L-BFGS-B',
                             bounds=limits,options={'maxiter':160,'ftol':2e-13,'gtol':1e-8,'maxcor':12})
                rows.append((fit.fun,fit.x))
            val,params=min(rows,key=lambda row:row[0])
            print(m,forced,round(obj(params,m,max(96,4*m))[0],10),np.round(params[:4],5),flush=True)


def rounded(endpoint, upper):
    scale=10**14
    value=(endpoint*scale).upper().ceil() if upper else (endpoint*scale).lower().floor()
    return F(int(value.unique_fmpz()),scale)

def show_policy(name, result):
    low,high,legal,first,last=result
    print(name, "ideal limits", rounded(low,False),rounded(high,True),
          "legal upper",rounded(legal,True),"computed blocks",first,last,flush=True)

def verify():
    rate=ball(".47525052")
    old=enclose(ball(F(193,120)).log(),[arb(0)])
    base=enclose(rate,[arb(0)])
    swap=enclose(2*rate,[rate,arb(0)])
    six=enclose(ball("1.42575"),[ball(s) for s in
                    ("0","-2.079442","-10.165846","12.237651","12.712905","13.188126")])
    short=enclose(rate,[arb(0),-arb(10**7).log()])
    for name,result in [("old geometric (legal upper uses the deeper displayed cutoff)",old),("retuned geometric",base),
                        ("swap every pair",swap),("six-round block",six),("short probe in every pair",short)]:
        show_policy(name,result)
    assert rounded(old[0],False)>F("3.5778249744078")
    assert rounded(base[2],True)<F("3.577824966")
    assert rounded(six[2],True)<F("3.577833426")
    assert rounded(short[2],True)<F("3.577825256")
    assert rounded(swap[0],False)>rounded(base[1],True)
    # Legal realization: pick a residue and a one-step phase, keep n>=-110.
    r=rate.exp()
    trunc=r**(-108)/(r-1)
    assert trunc<1
    for M,limit in [(4096,"3.577863477"),(2**25,"3.577824971"),(2**30,"3.577824966")]:
        upper=((M-2)*base[1]+2*swap[1])/(M*(1-trunc))
        rational=rounded(upper,True)
        assert rational<F(limit)
        print("One ordinary swap per",M,"rounds: legal upper",rational,flush=True)
    bound=permutation_bound(rate)
    assert bound>F("3.58799")
    print("Nonidentity permutations, M=10: lower",bound,flush=True)
    nbox,minimum=exclude_two_rounds()
    assert minimum>F("3.54")
    print("All two-round blocks: central boxes",nbox,"minimum lower",minimum,flush=True)

if __name__=="__main__":
    import argparse
    from time import perf_counter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search",action="store_true")
    args=parser.parse_args()
    if not __debug__: raise RuntimeError("Run without -O; verification uses assertions")
    start=perf_counter()
    if args.search: search()
    else: verify()
    print("Elapsed seconds:",round(perf_counter()-start,2))

