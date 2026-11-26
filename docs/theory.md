# CoFrGeNet-FNG 表达力的理论分析

> 问题：FFN 与 FNG 的参数公式只说明 FNG 参数更少，能否**不依赖训练实证**、
> 从数学上论证 FNG 的表达力优势？
>
> 简短结论：无条件的"FNG 表达力更强"是假命题（两者都是万能逼近器）。
> 可以严格证明的是三种特定意义下的**不对称**：精确表示分离、逼近速率
> 分离、模拟成本不对称。本文按"命题 + 证明思路 + 出处"组织。

## 0. 记号与立场

- FFN（一层隐层，扩展因子 α）：$f(x) = W_2\,\sigma(W_1 x)$，
  $W_1 \in \mathbb{R}^{\alpha p \times p}$，参数量 $2\alpha p^2$。
- 梯子（ladder）：深度 $d$，系数 $a_k = w_k^\top z$（$k=0..d$），输出
  $$f(z) = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{\ddots + \cfrac{1}{a_d}}}}$$
  FNG block = 门控投影（$x \mapsto z$）+ $L$ 条梯子 + 线性读出。
- **立场**：参数量只给出假设空间维度的上界，本身推不出表达力。比较表达力
  必须固定"目标函数类 + 误差度量"。

**否定性结论（比较的"天花板"）**：FFN 与梯子集成都是万能逼近器——
前者见 Cybenko (1989)、Hornik (1991)；后者（CoFrNet 集成 + 线性读出）
见 Puri et al. (2021)。宽度/梯子数不限时，两类函数都稠密于紧集上的连续
函数空间。**因此"更强"只能在以下三种限定意义下成立。**

## 1. 命题 1：梯子 ⊆ 低双次数有理函数

**命题**：深度 $d$ 的梯子是 $z$ 的有理函数，分子次数 ≤ $d+1$，分母次数 ≤ $d$。

**证明思路**：continuant 递推 $K_0=1$，$K_1=a_d$，
$K_k = a_{d-k+1}K_{k-1} + K_{k-2}$。每个 $a_k$ 是 $z$ 的线性型，对 $k$
归纳得 $\deg K_k \le k$（乘积项次数 1+(k−1)，另一项 k−2）。由
$f = a_0 + K_{d-1}/K_d = (K_{d-1} + a_0 K_d)/K_d$，分子次数
≤ max(d−1, 1+d) = d+1，分母 ≤ d。∎

**推论**：梯子的假设空间被双次数 $(d+1, d)$ 有理函数类**严格包含**——
一般多元有理函数的系数数目是组合数级的，梯子只有 $(d+1)p$ 个参数，
是高度结构化的有理函数。

## 2. 命题 2：一维截口上，梯子 = 全部 Padé 有理函数

**命题**（$p \ge 2$，非退化情形）：固定直线 $z = tu + z_0$（$t \in \mathbb{R}$），
深度 $d$ 梯子在该直线上的限制恰好是 $t$ 的全体次数 ≤ $(d+1, d)$ 的有理函数
（Padé 表 normal 的部分）。

**证明思路**：
- **限制方向**：$a_k(t) = (w_k^\top u)\,t + w_k^\top z_0$ 是 $t$ 的线性函数，
  梯子退化为部分分母线性的连分数。
- **张成方向**：经典 Padé–连分数对应——对任意有理函数 $P/Q$，欧几里得
  算法（quotient-difference）给出部分分母为 $t$ 的线性函数的连分数展开；
  两个线性约束 $w_k^\top u = \alpha_k$、$w_k^\top z_0 = \beta_k$ 在
  $p \ge 2$ 时对 $w_k$ 可解。见 Baker & Graves-Morris (1996)。∎

**意义**：梯子在每条一维截口上继承**一元有理逼近（Padé）的全部逼近
能力**——这是把经典有理逼近定理"搬运"到 FNG 上的桥。

## 3. 命题 3：精确表示的分离

**命题 3a**：深度 1 梯子精确表示 $x \mapsto 1/(w^\top x)$。在任何与极点
超平面 $\{w^\top x = 0\}$ 相交的开集上，**任何**连续激活的有限宽 FFN
都不能精确表示它。

**证明思路**：前者显然（$a_0=0, a_1 = w^\top x$，$d=1$ 时 $\tilde f = 1/a_1$）。
后者：连续激活的 FFN 处处连续、因而在该开集上有界；目标函数在极点附近
无界，二者不可能在该开集上点点相等。∎
（注：实现中的 $\varepsilon$ clamp 会把极点"削平"，命题按无 clamp 的
数学形式陈述。）

**命题 3b**：$|x|$ 被一个 ReLU 神经元精确表示；任何梯子（有理函数）
都不能在包含 0 的区间上精确表示它。

**证明思路**：设有理函数 $r$ 在 $(0,\varepsilon)$ 上等于 $x$，则
$r(x) - x$ 有无穷多零点，由有理函数的恒等定理 $r \equiv x$；同理由
$(-\varepsilon, 0)$ 得 $r \equiv -x$，矛盾。∎

**结论**：两函数类**互不包含**。对带奇点的目标，梯子严格占优；对带
折角/分段结构的目标，FFN（ReLU）严格占优。"谁更强"依赖目标函数类。

## 4. 命题 4：逼近速率分离（核心结果）

**命题**：对带奇异性（极点、折角、割线）的一元函数，深度 $d$ 梯子的
误差-参数曲线是**指数**的，FFN 是**代数**的。

**依据链**：
1. 由命题 2，深度 $d$ 梯子在一维截口上 = $[\sim d/d]$ Padé 类，
   参数代价 $O(dp)$。
2. **Newman (1964)**：$|x|$ 在 $[-1,1]$ 上，$n$ 次有理函数最优逼近误差
   $\le e^{-\pi\sqrt{n+1}}$（匹配下界见 Gonchar）；而**多项式**最优逼近
   误差 $\sim \beta/n$（Bernstein 常数 $\beta \approx 0.1079$）。
   指数速率 vs 代数速率。对 $1/x$、$\sqrt{x}$、带割线的解析函数，同类
   差距普遍成立（Stahl 的系列工作）。
3. **FFN 侧**：宽度 $W$ 的单层 MLP 对这类函数的最优速率受多项式/分段
   多项式速率限制，仅为代数阶。神经网络语境下的显式分离见
   Boullé, Nakatsukasa & Townsend (2020)《Rational Neural Networks》
   （$1/x$、$\sqrt{x}$、$|x|$：有理网络 $O(\log(1/\varepsilon))$ 参数
   vs ReLU 网络 $\mathrm{poly}(1/\varepsilon)$），以及 Telgarsky (2017)。∎

**意义**：这是"FNG 每参数表达力更强"能拥有的最强定理形式——
**对带奇异性结构的函数类，指数 vs 代数**。

## 5. 命题 5：互相模拟的成本不对称

**命题 5a（FNG 模拟 FFN，对数开销）**：$e^{-x}$ 的 $[d/d]$ Padé 逼近是
显式已知的连分数，在有界区间上误差随 $d$ 超指数衰减（Baker &
Graves-Morris 1996）。sigmoid / tanh / GELU 均为整函数或亚纯函数，
在有界域上有同类有理逼近速率。因此每个神经元 $\sigma(w^\top x + b)$
可被**一条深度 $O(\log(1/\varepsilon))$ 的梯子**以 $\varepsilon$ 精度
替代；宽度 $k$ 的 FFN ≈ $k$ 条对数深度梯子 + 同一个线性读出。

**命题 5b（FFN 模拟 FNG，多项式开销且不精确）**：由命题 3a，FFN 无法
精确表示 $1/(w^\top x)$；在极点附近其逼近宽度随与极点距离恶化
（命题 4 的代数速率）。

**结论**：模拟成本不对称——一个方向是 $O(\log(1/\varepsilon))$，
另一个方向是 $\mathrm{poly}(1/\varepsilon)$ 且永远不可能精确。

## 6. 证不出来的部分（诚实清单）

- **一般光滑高维函数类上无渐近分离**：两者的 worst-case 速率都是
  $n^{-s/p}$（维数灾难；Yarotsky 2017；DeVore, Hanin & Petrova 2021
  的综述）。
- **统计容量同阶**：伪维数/VC 型界对两者都是 $\tilde O(W)$ 量级
  （Harvey, Liaw & Mehrabian 2017；Bartlett, Maiorov & Meir 1998；
  有理函数类由 Warren 型符号计数可得同阶结果）。差异在**逼近偏差**
  （bias），不在容量（variance）。
- **多元有理逼近理论本身不完备**，不存在"多元 Newman 定理"；梯子的
  连分数结构比一般多元有理函数受限更多（命题 1 的推论）。
- **因此在语言建模上的优势是经验命题**：它等价于"自然语言数据对应的
  函数落在有理逼近快的区域"（低有效维、存在近奇异结构）。这只能靠
  实验检验。反例演示见本仓库 `fng_vs_ffn.py` 的玩具实验：随机 MLP
  教师函数是 FFN 的主场，FNG 收敛明显更慢——与上述理论完全一致。

## 7. 证明路线图（自助推导用）

1. 命题 1：continuant 次数归纳（初等）。
2. 命题 2：欧几里得算法 + 线性约束可解性（初等 + 经典对应）。
3. 命题 3：有界性论证 + 有理函数恒等定理（初等）。
4. 命题 4：代入 Newman / Stahl 的经典速率 + Boullé et al. 的 NN 版本。
5. 命题 5：exp 的 Padé 收敛定理 + 命题 3a/4。
6. 用 Puri et al. (2021) 的 universality 给两边"封顶"。

## 参考文献

- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal
  function. *Mathematics of Control, Signals and Systems*.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward
  networks. *Neural Networks*.
- Puri, I., Dhurandhar, A., Pedapati, T., Shanmugam, K., Wei, D.,
  Varshney, K. R. (2021). CoFrNets: Interpretable Neural Architecture
  Inspired by Continued Fractions. *NeurIPS 34*.
- Dhurandhar, A., et al. (2026). CoFrGeNet: Continued Fraction
  Architectures for Language Generation. *ICML 2026*. arXiv:2601.21766.
- Dhurandhar, A., Chenthamarakshan, V., Ramamurthy, K. N., Wei, D.,
  Nair, R. (2026). Non-Expanding Gated Continued Fraction Architecture
  for Feedforward Layers in Language Models. *NeurIPS 2026*.
  arXiv:2611.26835.
- Baker, G. A., Graves-Morris, P. (1996). *Padé Approximants*, 2nd ed.
  Cambridge University Press.
- Newman, D. J. (1964). Rational approximation to $|x|$.
  *Michigan Mathematical Journal*.
- Boullé, N., Nakatsukasa, Y., Townsend, A. (2020). Rational neural
  networks. *NeurIPS 33*.
- Telgarsky, M. (2017). Neural networks and rational functions. *ICML*.
- Yarotsky, D. (2017). Error bounds for approximations with deep ReLU
  networks. *Neural Networks*.
- DeVore, R., Hanin, B., Petrova, G. (2021). Neural network
  approximation. *Acta Numerica*.
- Harvey, N., Liaw, C., Mehrabian, A. (2017). Nearly-tight VC-dimension
  and pseudodimension bounds for piecewise linear neural networks. *COLT*.
- Bartlett, P., Maiorov, V., Meir, R. (1998). Almost linear VC-dimension
  bounds for piecewise polynomial networks. *NeurIPS*.
