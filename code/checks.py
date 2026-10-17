"""Checks for 'Opposite gaps defeat large-time unimodality for one-sided stable processes'.

Run: python checks.py   (numpy-free; needs mpmath and sympy). Every printed line is cited in the paper by its tag.
The proofs in the paper do not depend on these computations; they check the formulas and constants.
"""
import mpmath as mp
import sympy as sp
from fractions import Fraction as Fr

mp.mp.dps = 60
ok = True


def report(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"[{tag}] {'PASS' if cond else 'FAIL'}  {msg}")


# ---------------------------------------------------------------- K: the positive 1/2-stable kernel
u, t, lam = sp.symbols('u t lam', positive=True)
g = t * sp.exp(-t**2 / (4 * u)) / (2 * sp.sqrt(sp.pi) * u**sp.Rational(3, 2))
lap = sp.simplify(sp.integrate(g * sp.exp(-lam * u), (u, 0, sp.oo)))
report('K1', sp.simplify(lap - sp.exp(-t * sp.sqrt(lam))) == 0, f"Laplace transform of g_t = {lap}")
dg = sp.simplify(sp.diff(g, u) / (t * sp.exp(-t**2 / (4 * u)) / (2 * sp.sqrt(sp.pi) * u**sp.Rational(7, 2))))
report('K2', sp.simplify(dg - (t**2 / 4 - sp.Rational(3, 2) * u)) == 0, f"g_t'(u) = t e^(-t^2/4u) (t^2/4 - 3u/2) / (2 sqrt(pi) u^(7/2)); factor = {sp.expand(dg)}")
report('K3', sp.simplify(g.subs(t, 1).subs(u, u / t**2) / t**2 - g) == 0, "scaling g_t(u) = t^(-2) g_1(u/t^2)")

# ---------------------------------------------------------------- S: the series for the positive alpha-stable density


def series_density(x, a, terms=200):
    """(1/(pi x)) sum_k (-1)^(k+1) Gamma(k a+1)/k! sin(k pi a) x^(-k a): density with Laplace transform exp(-lam^a)."""
    x = mp.mpf(x)
    y = x**(-a)
    s = mp.mpf(0)
    for k in range(1, terms):
        s += (-1)**(k + 1) * mp.gamma(k * a + 1) / mp.factorial(k) * mp.sin(k * mp.pi * a) * y**k
    return s / (mp.pi * x)


def kanter_density(x, a):
    """Kanter (1975): f(x) = a/(1-a) x^(-1/(1-a)) (1/pi) int_0^pi K(u) exp(-K(u) x^(-a/(1-a))) du,
    K(u) = (sin(a u)/sin u)^(1/(1-a)) sin((1-a) u)/sin(a u). An independent formula for the same density."""
    x = mp.mpf(x)
    K = lambda u: (mp.sin(a * u) / mp.sin(u))**(1 / (1 - a)) * mp.sin((1 - a) * u) / mp.sin(a * u)
    z = x**(-a / (1 - a))
    integral = mp.quad(lambda u: K(u) * mp.exp(-K(u) * z), [0, mp.pi / 2, mp.pi])
    return a / (1 - a) * x**(-1 / (1 - a)) * integral / mp.pi


mp.mp.dps = 30
exact_half = lambda x: mp.mpf(1) / (2 * mp.sqrt(mp.pi)) * x**mp.mpf(-1.5) * mp.exp(-1 / (4 * mp.mpf(x)))
err = max(abs(series_density(x, mp.mpf(1) / 2) - exact_half(x)) for x in [0.3, 0.7, 1, 2, 5, 20])
report('S1', err < mp.mpf('1e-20'), f"alpha=1/2: series = (1/(2 sqrt pi)) x^(-3/2) e^(-1/(4x)) at 6 points, max error {mp.nstr(err, 3)}")

for a in [mp.mpf('0.5'), mp.mpf('0.3'), mp.mpf('0.7')]:
    pts = [1, 2, 5, 20]
    worst = max(abs(series_density(x, a) - kanter_density(x, a)) / kanter_density(x, a) for x in pts)
    report('S2', worst < mp.mpf('1e-15'), f"alpha={a}: series vs Kanter's integral at x in {pts}: max relative error {mp.nstr(worst, 3)}")
mp.mp.dps = 60

for a in [mp.mpf('0.1'), mp.mpf('0.3'), mp.mpf('0.5'), mp.mpf('0.7'), mp.mpf('0.9')]:
    c1 = mp.gamma(1 + a) * mp.sin(mp.pi * a) / mp.pi
    report('S3', c1 > 0, f"alpha={a}: leading coefficient c_1 = Gamma(1+alpha) sin(pi alpha)/pi = {mp.nstr(c1, 8)} > 0")

# ---------------------------------------------------------------- M: the initial law
# A_n = 2^(2^n), w_n = 2^(-n-2) exp(-A_n^2), w_0 = 1 - W
A = lambda n: mp.mpf(2)**(2**n)
W_bound = sum(Fr(1, 2**(n + 2)) for n in range(1, 200))
report('M1', W_bound < Fr(1, 4), f"W = sum w_n < sum 2^(-n-2) = 1/4 (partial sums {float(W_bound):.6f}); so w_0 > 3/4")
report('M2', all(A(n - 1)**2 == A(n) and A(n - 1) <= A(n) / 4 for n in range(2, 8)), "A_(n-1) = sqrt(A_n) <= A_n/4 for n >= 2 (checked n = 2..7)")
for lamb in [1, 10, 100]:
    mom = sum(mp.mpf(2)**(-n - 2) * mp.exp(lamb * A(n) - A(n)**2) for n in range(1, 9))
    report('M3', mp.isfinite(mom), f"sum_(n<=8) w_n e^(lambda A_n) at lambda={lamb}: {mp.nstr(mom, 6)} (terms n>=9 are below e^(-A_9^2/2))")

# ---------------------------------------------------------------- C: the classical density decreases on I_n (alpha = 1/2)
weights = {0: None}
Wsum = sum(mp.mpf(2)**(-n - 2) * mp.exp(-A(n)**2) for n in range(1, 8))
atoms = [(mp.mpf(0), 1 - Wsum)] + [(-A(n), mp.mpf(2)**(-n - 2) * mp.exp(-A(n)**2)) for n in range(1, 8)]


def dq(x, tt):
    """derivative of q_t(x) = sum_j p_j g_t(x - a_j) for the 1/2-stable kernel, atoms at a_j <= 0."""
    s = mp.mpf(0)
    for aj, pj in atoms:
        v = x - aj
        if v > 0:
            s += pj * tt * mp.exp(-tt**2 / (4 * v)) * (tt**2 / 4 - mp.mpf(3) / 2 * v) / (2 * mp.sqrt(mp.pi) * v**mp.mpf(3.5))
    return s


for tt, n in [(mp.mpf(1), 2), (mp.mpf(5), 3), (mp.mpf(50), 4)]:
    b = tt**2 / 6
    lo, hi = -A(n) + b, -A(n - 1)
    pts = [lo + (hi - lo) * k / 20 for k in range(1, 20)]
    report('C1', lo < hi and all(dq(x, tt) < 0 for x in pts), f"t={tt}, n={n}: q_t' < 0 at 19 points of I_n = (-A_n + t^2/6, -A_(n-1)) = ({mp.nstr(lo, 6)}, {mp.nstr(hi, 6)})")

# ---------------------------------------------------------------- F: the free estimates
mp.mp.dps = 120


def G_and_I2(x, n_max):
    G = mp.mpf(0)
    I2 = mp.mpf(0)
    W = sum(mp.mpf(2)**(-n - 2) * mp.exp(-A(n)**2) for n in range(1, n_max + 1))
    for s, w in [(mp.mpf(0), 1 - W)] + [(-A(n), mp.mpf(2)**(-n - 2) * mp.exp(-A(n)**2)) for n in range(1, n_max + 1)]:
        G += w / (x - s)
        I2 += w / (x - s)**2
    return G, I2


for a in [mp.mpf('0.1'), mp.mpf('0.5'), mp.mpf('0.9')]:
    for tt in [mp.mpf(1), mp.mpf(100)]:
        n = 2
        while A(n)**a < 18 * (1 - a) * tt:
            n += 1
        AA = A(n)
        c = -AA / 2
        G, I2 = G_and_I2(c, n + 2)
        aa = -G
        Hp = 1 - tt * (1 - a) * I2 * aa**(a - 2)
        report('F1', aa > 1 / AA and I2 < 9 / AA**2 and Hp > mp.mpf(1) / 2,
               f"alpha={a}, t={tt}, n={n}: -G(c_n) A_n = {mp.nstr(aa * AA, 6)} > 1, I_2(c_n) A_n^2 = {mp.nstr(I2 * AA**2, 6)} < 9, H_t'(c_n) = {mp.nstr(Hp, 8)} > 1/2")
        # the interval: H' > 0 and G < 0 on a grid of (-A_n, -A_(n-1)) around c_n
        xs = [-AA + (AA - A(n - 1)) * k / 40 for k in range(1, 40)]
        good = [x for x in xs if G_and_I2(x, n + 2)[0] < 0 and 1 - tt * (1 - a) * G_and_I2(x, n + 2)[1] * (-G_and_I2(x, n + 2)[0])**(a - 2) > 0]
        report('F2', len(good) > 0, f"   grid of 39 points in (-A_n, -A_(n-1)): {len(good)} have G < 0 and H_t' > 0 (zero-density set of the free convolution)")

# boundary value of the Voiculescu transform at a negative real point, from the upper half plane
for a in [mp.mpf('0.3'), mp.mpf('0.5'), mp.mpf('0.8')]:
    aa = mp.mpf('0.37')
    w = -1 / aa + mp.mpf('1e-40') * 1j
    phi = -mp.exp(1j * mp.pi * a) * mp.power(w, 1 - a)
    report('F3', abs(phi - aa**(a - 1)) < mp.mpf('1e-30'), f"alpha={a}: phi(-1/a + i0) = -e^(i pi alpha)(-1/a)^(1-alpha) = a^(alpha-1) (got {mp.nstr(phi, 12)})")

# -phi maps the upper half plane to itself: arg(-phi(r e^{i th})) = pi alpha + th (1-alpha) in (0, pi)
report('F4', all(0 < float(a) * mp.pi + th * (1 - float(a)) < mp.pi for a in [0.05, 0.5, 0.95] for th in [1e-9, 1.0, mp.pi - 1e-9]), "arg(-phi(r e^(i theta))) = pi alpha + theta (1-alpha) lies in (0, pi)")

print('ALL PASS' if ok else 'SOME CHECK FAILED')
