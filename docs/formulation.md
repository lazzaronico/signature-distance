# Formulas and conventions

Let μ and ν be probability measures on Euclidean space with finite first moments.
For an anchor z, R_z μ is the distribution of ||x−z|| under x~μ. Write

    h_z(μ,ν) = W₁(R_z μ, R_z ν)
              = integral_0^1 |q_z,μ(u) − q_z,ν(u)| du.

The first argument supplies the anchors:

    SDiv(μ || ν) = integral h_z(μ,ν) μ(dz)
    S(μ,ν) = SDiv(μ || ν) + SDiv(ν || μ)
    SD(μ,ν) = sqrt(S(μ,ν)).

For equal empirical sizes, each radial W₁ is the mean absolute difference of
sorted rows. For unequal sizes, integrate over the union of the uniform
empirical quantile breakpoints. Keep all empirical atoms and zero self-distances.
Repeated points carry their repeated empirical mass.

## Energy and full transport

With independent draws, C=E||X−Y||, A=E||X−X′||, B=E||Y−Y′||:

    E = 2C − A − B
    d_E = sqrt(E)
    E <= S <= 2 W₁ <= 2C.

E is the raw energy expression, not its square. S equals E if and only if the cross-radial
quantiles dominate the within-radial quantiles at both sets of anchors, almost
everywhere in anchor and quantile. This follows because S−E is the sum of
nonnegative integrals of |Δ|−Δ at μ anchors and |Δ|+Δ at ν anchors, where
Δ=q_z,ν−q_z,μ. Energy
still separates distinct distributions outside that regime. Neither this lower
bound nor the name “distance” gives SD a triangle inequality.

For g_z(x,y)=| ||x−z||−||y−z|| | and couplings Π(μ,ν):

    S = integral_z [inf_πz integral g_z dπz] (μ+ν)(dz)
    W₁ = inf_π integral [sup_z g_z] dπ.

The supremum is ||x−y|| and, for empirical points, a maximum over X∪Y suffices.
SD uses independent radial matchings and sums anchor averages; W₁ uses one common
matching and the maximum radial discrepancy. A cubic anchor-cost tensor is not
needed to compute W₁: solve transport on the ordinary cross-distance matrix.

For δ=W₁(μ,μ′)+W₁(ν,ν′), the values also obey

    |S(μ,ν)−S(μ′,ν′)| <= 4δ
    |SD(μ,ν)−SD(μ′,ν′)| <= 2 sqrt(δ).

The anchor-wise discrepancy changes by at most δ when the compared measures
change, and is 2-Lipschitz in the anchor. Splitting these two changes gives
2δ+2δ. This controls values under input perturbations, not gradients or neural
optimization trajectories.

## Optional related metric

For one fixed Q shared across every comparison:

    A_Q(μ,ν) = integral h_z(μ,ν) Q(dz)
    D_Q(μ,ν) = sqrt(2A_Q(μ,ν) + E(μ,ν)).

A_Q is a pseudometric; its square root is also one. A_Q already separates when
Q has full support, since continuity makes a zero integral imply h_z=0 at every
anchor and hence S=0. On a finite domain, a fixed bank containing every possible
atom with positive weight also identifies all atom masses through the radial
laws at radius zero. An arbitrary finite anchor subset lacks that general
guarantee. Minkowski combines sqrt(2A_Q) with
the separating energy metric to prove D_Q is a metric for any fixed Q. Any finite fixed bank
works, even a single origin, but sparse anchors may discard useful geometry.
Choosing Q=(μ+ν)/2 anew for each pair makes 2A_Q=S and loses this guarantee.

## Generated-first loss API

- `sd(X,Y)` is raw S; `SignatureDistance(mode="signature").score(Y)` is sqrt(S).
- `sdiv(X,Y)` is SDiv(Y || X): the real-anchor direction.
- `cd(X,Y)` sums the two column-distribution comparisons over radial quantiles.
  `csd(X,Y)` is sqrt(S+C), with a numerical floor inside the square root.
- `gsd(X,Y)` is sqrt(S+GDiv(X||Y)+GDiv(Y||X)), with the same numerical floor.
  GDiv matches a point's cross-profile to the within-profile at its nearest
  opposing point. The selected target profiles are detached during training.
- `energy(X,Y)` is E with complete empirical masses. `ediv(X,Y)` is 2C−A;
  the generated-point gradients agree when Y is fixed. Rooting changes this
  constant-removal argument. Off-diagonal n(n−1) estimation is a different loss.

CSD/GSD are additional objectives, not established Hilbert or metric constructions.
The synthetic experiments optimize the raw E and S or CSD/GSD directly, and the
expansion experiment uses the raw E, S and S + C. The glocal TCGA experiment and
its global-only control use sqrt(E) and sqrt(S) within each global/local term
before averaging. The [experiment guide](../examples/paper_revision/README.md)
specifies the validation protocols and comparator settings.

`mode="simplified"` is retained as a legacy alias of the full signature mode.
`normalize=True` instead selects a per-anchor normalized descriptive variant;
it is not the main S or the global index sqrt(S/(2C)). Neither pair-dependent
normalization nor a non-Euclidean `p` receives the Euclidean metric guarantees.
