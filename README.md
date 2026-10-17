# Opposite gaps defeat large-time unimodality for one-sided stable processes

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (LaTeX source in [paper/main.tex](paper/main.tex)).

Hasebe and Ueda (ALEA 15 (2018), 353-374, Conjecture 5.5) conjectured that if S_t (resp. T_t) is the law at time t of a
classical (resp. free) strictly alpha-stable process and the initial law mu has a finite moment of order 2 + alpha, then
mu * S_t (resp. mu ⊞ T_t) is unimodal for all large t.

The paper shows that this fails for every alpha in (0,1), in both settings, for the stable laws concentrated on the
positive half-line. With A_n = 2^(2^n) and w_n = 2^(-n-2) exp(-A_n^2), the measure

    mu = (1 - W) delta_0 + sum_{n>=1} w_n delta_{-A_n},    W = sum w_n,

has finite exponential moments of every order, and for every alpha in (0,1) and every t > 0 neither mu * S_t nor
mu ⊞ T_t is unimodal. Here S_t has Laplace transform exp(-t lambda^alpha) and T_t has Voiculescu transform
-t e^(i pi alpha) z^(1-alpha). In the classical case the convolution density is strictly decreasing on intervals
arbitrarily far to the left; in the free case the density vanishes on an interval inside each late gap of mu while
staying positive at the images of the atoms on either side (Huang-Wang, arXiv:2203.00421, Section 3).

At alpha = 1/2 this answers the question the authors raise for the positive 1/2-stable law just before the conjecture:
a finite 5/2-moment does not give eventual unimodality, although compact support does (their Theorem 5.3).

The cases alpha in [1,2) and the two-sided stable laws, in particular the symmetric ones, remain open.

## Code

[code/checks.py](code/checks.py) checks the 1/2-stable kernel (Laplace transform, derivative, scaling), the series for
the positive alpha-stable density, the constants of the initial law, the decrease of the classical density on the gap
intervals, and the estimates of the free argument for alpha in {0.1, 0.5, 0.9}. Output: [code/checks_output.txt](code/checks_output.txt).
The proofs do not depend on these computations.

    python code/checks.py      # needs mpmath and sympy
