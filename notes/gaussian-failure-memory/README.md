# Gaussian search with failure memory

A target is drawn once from $N(0,1)$. The searcher starts at zero with standard Brownian motion, returns instantaneously to zero at each timeout, and starts a fresh independent path. Only success or timeout is observed. Let $J_*$ be the infimum of the mean time over positive deadline sequences with $\sum t_j=\infty$. Noncompletion costs infinity. Independent randomization cannot lower this infimum: fixing the random seeds fixes the sequence on the all-failure path.

## Certified results

| Memory on the same numerical grid | Global lower bound, rounded down |
|---|---:|
| No stored failure factors | 3.00737109 |
| Four successive thresholds near 0.70, 1.22, 1.70 and 2.30 | 3.02866150 |
| Branching histories and partial retention of failed attempts | **3.07196953** |

A feasible sequence in the bundle costs less than **3.0907986351464**. The global gap is below **0.01882910**. The [previous four-threshold certificate](previous/certificate.json) gave 3.00847710211, reproducible with its [archived verifier](previous/verify.py). The four-threshold control here uses the same distance and time grid as the final result, so it separates the contribution of richer histories from the change of discretization.

**Stopping criterion.** With the 842 reference histories, 252 distance nodes and continuation options in this bundle, a reverse witness caps any lower bound satisfying these local inequalities at **3.07220501**. Better potential coefficients on this representation can gain less than **0.00023548**. A coarser run with the same reference library certified 3.06802819 before the last refinement. The ceiling is about the certificate family; only the executable sequence gives the global upper bound.

Keeping all earlier decompositions fixed but allowing *arbitrary valid continuation bounds* upon reaching a reference history of length 16 gives the independently checked ceiling **3.07230842**. Thus extensions confined to those later histories can add at most **0.00033889** to the current lower bound. Changing the earlier spatial decomposition is outside this ceiling.

There is a separate bound on the value of changing the tail of the executable strategy. Fix its first $K$ deadlines; after those attempts grant perfect distance information and use the known-distance lower cost $2.67x^2$. Subtracting this optimistic cost from the original verified cost gives the following upper limits on the savings.

| Fixed initial attempts | Maximum further saving, rounded up |
|---:|---:|
| 8 | 0.014671 |
| 16 | 0.000650 |
| 20 | 0.000143 |
| 28 | 0.00000719 |

This table conditions on the stated initial decisions. The length of a certificate history counts retained likelihood factors and can differ from the actual attempt count.

## Continuous target and a conditional cost bound

At distance $x$ the success probability, failure probability and mean duration in one attempt are
\[
q_x(t)=\operatorname{erfc}(x/\sqrt{2t}),\quad s_x(t)=1-q_x(t),\quad
a_x(t)=t s_x(t)+x\sqrt{2t/\pi}e^{-x^2/(2t)}-x^2q_x(t).
\]
These follow from the Brownian reflection principle and integration of the survival function. The verifier certifies $a_1(t)\ge2.67q_1(t)$ by monotone interval subdivision and analytic bounds on the exterior intervals. By scaling, $a_x(t)\ge O_x q_x(t)$ with $O_x=2.67x^2$. Summing over attempts proves this lower bound even if the searcher is told the distance at the start of the continuation.

For $n$ distance nodes use $x_i=2^{-6+9i/n}$, $0\le i<n$, and $r=2^{9/n}$. On $[1/64,8]$ each distance is $u x_i$, $1\le u<r$. For fixed $u$, replace each deadline by $t/u^2$ and multiply the resulting search time by $u^2$. For the half-normal density $p$, take $w_i=x_i p(rx_i)$. The ratio $x_i p(ux_i)/w_i$ is nondecreasing in $i$. Thus a bound valid for all nonnegative nondecreasing tilts of $w$ integrates exactly over $u$. Its root coefficient for each distance is
\[
m_i=\int_1^r u^2 x_i p(ux_i)\,du
=\frac{x_ip(x_i)-rx_i p(rx_i)+\operatorname{erfc}(x_i/\sqrt2)-\operatorname{erfc}(rx_i/\sqrt2)}{x_i^2}.
\]
Discarding the two exterior regions is conservative for the lower bound. The upper-bound calculation uses the whole distribution.

## Retaining information after a timeout

State $k$ stores a sorted list of reference deadlines. Set $v_{k,i}=w_i\prod_{d\in k}s_i(d)$ and assign a nonnegative potential $L_{k,i}$ to each distance. State zero is empty. The root claim is $\sum_i m_iL_{0,i}$. Every nonnegative nondecreasing tilt is a nonnegative combination of suffix indicators $1_{i\ge j}$.

After a deadline $t$ one option is to stay in state $k$, using all the failed mass $v_{k,i}s_i(t)$. An advance to state $h>k$ uses a matching: each entry in $h$ must be matched to a different, at least as large entry of $k$ or to one additional deadline. Let $b$ be the least additional time needed, zero if none is needed. Matching the largest requirements first computes $b$. For $z=\max(t,b)$ split the true failed mass as follows.

1. Keep $v_{k,i}s_i(z)$ at $h$. Its ratio to $v_{h,i}$ increases weakly in $i$. Each unmatched failure factor increases with distance; for $d_2\ge d_1$ the ratio $s_x(d_2)/s_x(d_1)$ also increases weakly. Indeed the ratio of the two derivatives in $x$ is proportional to $\exp[(1/d_1-1/d_2)x^2/2]$ and the failure-probability ratio is its weighted running average.
2. On the excess $v_{k,i}(s_i(t)-s_i(z))$, disclose the distance and credit only $O_i$. This allows a short actual failure to retain part of the evidence of a longer reference failure.

For each $k,j,t$ we require at least one allowed option to satisfy
\[
\sum_{i\ge j}v_{k,i}
[a_i(t)+s_i(z)L_{h,i}+(s_i(t)-s_i(z))O_i-L_{k,i}]\ge0. \tag{1}
\]
For a stay take $h=k,z=t$. The choice may depend on $j,t$. Fix a real sequence, split its posterior into suffixes, and apply (1) to each component. Expand the retained mass into suffixes at the next state; the remaining mass has a valid disclosed-distance bound. All these splits have nonnegative coefficients. After finitely many rounds, cumulative cost plus outstanding valid claims dominates the initial claim. If the sequence has finite cost, the probability of continued failure tends to zero on the finite discrete support. All potentials are bounded there, so the outstanding potential vanishes; disclosed components already have the guaranteed continuation bound for that same real sequence. The auxiliary suffixes and disclosures are used only for the proof.

## Continuous deadlines and rounding

Every positive matching threshold and option switch is a grid edge. On each cell $[l,r']$ the residual at a node is a constant plus $a_i(t)+s_i(t)M_i$, with $M_i\ge0$ equal to a continuation potential or to $O_i$. The maximum possible interpolation loss below the chord is $M_i e_i$ with
\[
e_i=\frac{3}{16\sqrt{2\pi}}(r'/l-1)^2
       \frac{x_i}{\sqrt l}\exp[-x_i^2/(2r')].
\]
Indeed $a_i'=s_i$, $q_i'=x_i e^{-x_i^2/(2t)}/(\sqrt{2\pi}t^{3/2})$, and $(a_i+s_iM_i)''\le3M_iq_i'/(2l)$. Multiply the uniform upper bound on the second derivative by $(r'-l)^2/8$. After subtracting this reserve, the **same option** must pass both endpoints for the particular suffix. No two-sided curvature estimate is needed.

Below the first endpoint $\epsilon$, stay uses $a_i(t)\ge t[1-q_i(\epsilon)]$ and $q_i(t)/t\le q_i(\epsilon)/\epsilon$ for $x_i^2\ge2\epsilon$ (Gaussian Mills bound). Above the last endpoint $M$, stay suffices once $a_i(M)>L_{k,i}$. Both checks are carried out at every node.

Arb at 144 bits encloses the special functions and the integrals. For the numerous Bellman comparisons the verifier converts Arb endpoints outward to IEEE binary64. It rounds each basic operation outward, chooses weight endpoints according to sign, and pads each suffix sum by $8n\epsilon_{\rm mach}$ times the computed sum of magnitudes plus $n\,10^{-320}$. This covers addition error, rounding in the magnitude sum and gradual underflow. The code rejects nonfinite enclosures and requires assertions and IEEE arithmetic.

## What the reverse witness proves

Use suffix variables $B_{k,j}=\sum_{i\ge j}v_{k,i}L_{k,i}$. If $H_i=v_{k,i}s_i(z)/v_{h,i}$, the kept term in (1) becomes
\[
 H_j B_{h,j}+\sum_{i>j}(H_i-H_{i-1})B_{h,i}.
\]
All coefficients are nonnegative. A stay has self coefficient $s_j(t)<1$. For each $k,j$ the verifier exhibits one finite positive deadline at which **all** allowed options give residual at most zero for an upper potential $U$. A one-sided limit at a finite positive cell edge is valid by continuity inside that cell. Compare any admissible certificate with $U$ by descending state and suffix index. An advance has $h>k$; for a stay subtract the self term and divide by $1-s_j(t)>0$. This proves $B^L_{k,j}\le B^U_{k,j}$. Since $m_i/w_i$ is nondecreasing, the same upper comparison holds for the root claim.

For the late-history ceiling, stop at the specified reference depth. Every universally valid continuation bound on a given suffix is at most the cost of a feasible periodic policy on that suffix. At each terminal suffix the verifier checks such a period from a menu of 44. Earlier states retain their allowed decompositions and the same reverse argument.

## Reproduction

The [certificate bundle](certificates.json.gz) contains the grid, histories, routing table, potentials, witnesses and the 36-deadline feasible sequence. Its times are in units of $10^{-4}$; the final deadline repeats forever. From this directory run the [independent verifier](verify.py):

~~~sh
uv run --no-project --python 3.12 --with numpy==2.5.5 --with python-flint==0.9.0 python verify.py
~~~

To run only the quick no-history control, append the option shown here:

~~~sh
uv run --no-project --python 3.12 --with numpy==2.5.5 --with python-flint==0.9.0 python verify.py --case no_history
~~~

The [constructor](rebuild.py) reproduces candidate potentials by backward recursion on the archived histories and grid. Floating optimization is not trusted by the verification step.

~~~sh
uv run --no-project --python 3.12 --with numpy==2.5.5 --with scipy==1.18.1 python rebuild.py certificates.json.gz regenerated.json
uv run --no-project --python 3.12 --with numpy==2.5.5 --with python-flint==0.9.0 python verify.py regenerated.json
~~~

The feasible sequence is evaluated by backward renewal recursion and validated quadrature on $[0,12]$. For final period $T>1$, finite-prefix time sum $P$, $H=12$ and $d=(1-1/T)/2$, its omitted tail is bounded by
\[
 P\,\operatorname{erfc}(H/\sqrt2)+T(1/\sqrt T+\sqrt T/H^2)e^{-dH^2}/(2d).
\]
This follows from $P+T/q_x(T)$ and the Gaussian Mills inequality. The fixed-prefix test integrates its nonnegative prefix-plus-disclosed-cost expression on $[0,12]$ and discards its positive tail.

This computational working note was prepared with AI assistance and has not undergone external peer review.
