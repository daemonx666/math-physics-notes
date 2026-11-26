# 数学物理研究笔记 · Mathematical Physics Notes

这个仓库存放数学与理论物理的证明、研究笔记和论文预稿。每篇工作有独立目录，包含模型假设、推导和引用。涉及计算的结果附复算脚本。

## 由目标距离分布判断固定周期重置何时最优

[阅读论文（PDF）](papers/gaussian-periodic-resetting/paper.pdf) · [LaTeX 源稿](papers/gaussian-periodic-resetting/paper.tex) · [高斯特例复算脚本](papers/gaussian-periodic-resetting/verify.py)

目标距离只抽取一次。一维布朗粒子搜索该目标，超时后瞬间返回原点。每轮时限几乎必然为正且有限，各轮独立同分布，并与目标及布朗路径独立；平均时限可以是无穷大。设距离在正半轴上的密度可归一化且正比于 $e^{-V(x)}$，其中 $V$ 在正半轴连续可微。论文证明，只要 $V'(x)/x$ 单调不减，就存在一个固定周期，在所有独立同分布的重置时钟中达到最短平均搜索时间。

在布朗运动方差为 $t$ 的单位下，令 $\ell$ 为 $V'(x)/x$ 在无穷远处的极限。它必为正数或正无穷。周期 $t$ 的平均搜索时间有限当且仅当 $1/t<\ell$。当 $\ell$ 有限时，临界周期 $1/\ell$ 的平均搜索时间发散；当 $\ell=\infty$ 时，每个正周期都有有限平均搜索时间。

标准高斯目标给出最优周期约 $2.81592$、平均时间约 $3.34972$。Pinsky 已在固定周期族内算出这个基准。本文证明该周期对所有上述时钟全局最优，并给出适用于一类非高斯目标的充分条件。[Pinsky，定理 3](https://arxiv.org/pdf/2205.02063)

**稿件状态**　未经同行评审的英文工作稿。AI 参与推导、写作和复核。可在仓库根目录复算高斯特例：

```sh
python3 papers/gaussian-periodic-resetting/verify.py
```

## 一维布朗搜索的最优重置时钟

[阅读论文（PDF）](papers/brownian-resetting/paper.pdf) · [LaTeX 源稿](papers/brownian-resetting/paper.tex) · [复算脚本](papers/brownian-resetting/verify.py)

一个粒子从原点出发寻找目标。目标的位置只抽取一次，搜索失败后粒子可以回到原点。每轮重置时钟都从同一个分布独立抽取。已知目标距离的分布，怎样设置时钟，才能让找到目标的平均时间最短？

预稿证明了一个适用于上述全部时钟的下界，并对一族幂律目标分布构造了达到下界的时钟。目标距离密度正比于 $(1+x/L)^{-2b}$ 时，有两个临界点。

| 参数 | 搜索结果 |
| --- | --- |
| $1/2 < b \le 3/2$ | 每个允许的时钟都产生无穷大的平均搜索时间。 |
| $3/2 < b \le 2$ | 最优平均搜索时间有限，任何最优时钟的平均设定间隔都为无穷大。 |
| $b > 2$ | 最优平均搜索时间有限，达到最优值的时钟具有有限平均间隔。 |

例如，$b=2$ 时最优平均搜索时间恰好为 $3L^2/D$，其中 $D$ 是扩散系数。这里的“设定间隔”指闹钟原定多久后响起。如果粒子先找到目标，当轮搜索会提前结束。

**稿件状态**　工作预稿，未经同行评审。AI 参与了推导、写作及脚本实现。Evans 与 Ray 已给出目标密度与单轮成功概率的平方关系；本稿讨论全局下界、取等条件和上述幂律族的精确解。原创优先权尚待独立核实。[Evans 与 Ray，Physical Review Letters（2025）](https://doi.org/10.1103/2wpq-gl2j)

在仓库根目录运行以下命令，即可用 Python 标准库核对四组参数的积分结果。

```sh
python3 papers/brownian-resetting/verify.py
```

已安装 LaTeX 的读者可重新编译论文。

```sh
cd papers/brownian-resetting
pdflatex -halt-on-error paper.tex
pdflatex -halt-on-error paper.tex
```
