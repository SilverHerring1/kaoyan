#!/usr/bin/env python3
r"""
Exact symbolic & numerical verification suite (SymPy) for Math I (27 Zhang Yu 6-book system).
Run with workspace .venv:
    .\.venv\Scripts\python.exe scripts/verify_math1.py
"""

from __future__ import annotations

import math
import sys

import mpmath as mp
import sympy as sp


def run_all_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, bool(condition), detail))

    x, t, u = sp.symbols("x t u", real=True)
    xp = sp.symbols("xp", positive=True)
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", real=True)
    ap = sp.symbols("ap", positive=True)

    # ==================== Calculus Lecture 1: Function Limits & Continuity ====================

    # 1. Core equivalent infinitesimals & difference-type infinitesimals (zy30 P.44,50-53 & zy18 P.4)
    equiv_pairs = [
        (sp.sin(x), x),
        (sp.tan(x), x),
        (sp.asin(x), x),
        (sp.atan(x), x),
        (sp.log(1 + x), x),
        (sp.exp(x) - 1, x),
        (ap**x - 1, x * sp.log(ap)),
        (1 - sp.cos(x), x**2 / 2),
        ((1 + x)**a - 1, a * x),  # symbolic a != 0 tested at a=3/2 below
        (x - sp.sin(x), x**3 / 6),
        (x - sp.asin(x), -x**3 / 6),
        (x - sp.tan(x), -x**3 / 3),
        (x - sp.atan(x), x**3 / 3),
        (sp.tan(x) - sp.sin(x), x**3 / 2),
        (x - sp.log(1 + x), x**2 / 2),
        (sp.exp(x) - 1 - x, x**2 / 2),
        (1 - sp.cos(x)**sp.Rational(3, 2), sp.Rational(3, 4) * x**2),
        (1 - sp.sqrt(sp.cos(x)), x**2 / 4),
        ((1 + x)**x - 1, x**2),
        (sp.cos(x) - sp.exp(-x**2 / 2), -x**4 / 12),
    ]
    equiv_ok = True
    for num_e, den_e in equiv_pairs:
        if num_e.has(a):
            lim_v = sp.limit(num_e.subs(a, sp.Rational(3, 2)) / den_e.subs(a, sp.Rational(3, 2)), x, 0)
        else:
            lim_v = sp.limit(num_e / den_e, x, 0)
        if sp.simplify(lim_v - 1) != 0:
            equiv_ok = False
    lim_e_pow = sp.limit(((1 + xp)**(1 / xp) - sp.E) / (-sp.E * xp / 2), xp, 0, "+")
    check("m1_c01_f_equiv_infinitesimals", equiv_ok and sp.simplify(lim_e_pow - 1) == 0)

    # 2. All 8 core Maclaurin expansions + sec(x) (zy30 P.50)
    taylor_checks = [
        (sp.sin(x), x - x**3 / 6, 4),
        (sp.cos(x), 1 - x**2 / 2 + x**4 / 24, 5),
        (sp.asin(x), x + x**3 / 6, 4),
        (sp.tan(x), x + x**3 / 3, 4),
        (sp.atan(x), x - x**3 / 3, 4),
        (sp.log(1 + x), x - x**2 / 2 + x**3 / 3, 4),
        (sp.exp(x), 1 + x + x**2 / 2 + x**3 / 6, 4),
        ((1 + x)**a, 1 + a * x + a * (a - 1) * x**2 / 2, 3),
        (sp.sec(x), 1 + x**2 / 2, 4),
    ]
    taylor_ok = all(
        sp.simplify(sp.series(expr, x, 0, ord_n).removeO() - expected) == 0
        for expr, expected, ord_n in taylor_checks
    )
    check("m1_c01_f_taylor_8", taylor_ok)

    # 3. Hyperbolic triad: y = ln(x + sqrt(x^2 + 1)) (zy30 P.5-7 例1.3 & 注)
    asinh_e = sp.log(x + sp.sqrt(x**2 + 1))
    odd_ok = sp.simplify(sp.logcombine(asinh_e.subs(x, -x) + asinh_e, force=True)) == 0
    inv_ok = sp.simplify((sp.sinh(t) + sp.cosh(t)).rewrite(sp.exp) - sp.exp(t)) == 0 and sp.simplify(sp.sinh(t)**2 + 1 - sp.cosh(t)**2) == 0
    lim_asinh = sp.limit(asinh_e / x, x, 0)
    deriv_asinh = sp.simplify(sp.diff(asinh_e, x) - 1 / sp.sqrt(x**2 + 1)) == 0
    int_asinh = sp.simplify(sp.logcombine(sp.integrate(asinh_e + x**2, (x, -1, 1)), force=True))
    check("m1_c01_f_asinh_triad",
          odd_ok and inv_ok and lim_asinh == 1 and deriv_asinh and int_asinh == sp.Rational(2, 3))

    # 4. Variable upper-limit integral equivalence notes (zy18 P.5-6)
    # Note 1: int_0^x (e^{t^3}-1)dt ~ x^4/4
    l_v1 = sp.limit((sp.exp(x**3) - 1) / (x**3), x, 0)  # via L'Hopital on integral / (x^4/4)
    # Note 2: int_0^{2-2cos x} (e^{t^2}-1)dt ~ x^6/3 -> deriv = (e^{(2-2cosx)^2}-1)*(2sinx) / (2x^5)
    g2 = 2 - 2 * sp.cos(x)
    l_v2 = sp.limit((sp.exp(g2**2) - 1) * sp.diff(g2, x) / (2 * x**5), x, 0)
    # Note 3: int_0^{x^2} (e^{t^3}-1)dt ~ x^8/4 -> deriv = (e^{x^6}-1)*2x / (2x^7)
    l_v3 = sp.limit((sp.exp(x**6) - 1) * (2 * x) / (2 * x**7), x, 0)
    # Note 4: int_0^{1-cos x} sqrt(sin(t^3))dt ~ (sqrt(2)/20) x^5 as x->0+
    g4 = 1 - sp.cos(xp)
    l_v4 = sp.limit(sp.sqrt(sp.sin(g4**3)) * sp.diff(g4, xp) / (sp.sqrt(2) / 4 * xp**4), xp, 0, "+")
    check("m1_c01_f_var_limit_equiv",
          l_v1 == 1 and l_v2 == 1 and l_v3 == 1 and sp.simplify(l_v4 - 1) == 0)

    # 5. zy30 Ex 1.1 & Ex 1.2 & Ex 1.4 (function rules, P.3-7)
    f11 = lambda s: s / (s**2 - 2)
    ex11_ok = sp.simplify(f11(xp + 1 / xp) - (xp + xp**3) / (1 + xp**4)) == 0
    f12 = lambda s: s / sp.sqrt(1 + s**2)
    ex12_ok = sp.simplify(2 * f12(xp) + xp**2 * f12(1 / xp) - (xp**2 + 2 * xp) / sp.sqrt(1 + xp**2)) == 0
    f12_note = lambda s: (s + s**2) / (1 + s**2)
    ex12_note_ok = sp.simplify(f12_note(x) + x * f12_note(-x) - x) == 0
    # Ex 1.4: -x^2 + 2x + 3 = -(x-1)^2 + 4
    ex14_ok = sp.solve(-x**2 + 2 * x + 3, x) == [-1, 3] and sp.simplify((-x**2 + 2 * x + 3).subs(x, 1)) == 4
    check("m1_c01_ex_zy30_01_04", ex11_ok and ex12_ok and ex12_note_ok and ex14_ok)

    # 6. zy30 Ex 1.6 & Ex 1.10 (boundedness & log-diff max, P.9, 15-16)
    f16_crit = sp.solve(sp.diff(x / (1 + x**2), x), x)
    # Ex 1.10: y = x^6 (1-x)^2 (1-2x)^4 on (0, 1/2)
    lny110 = 6 * sp.log(xp) + 2 * sp.log(1 - xp) + 4 * sp.log(1 - 2 * xp)
    roots110 = sp.solve(sp.diff(lny110, xp), xp)
    r_valid = [r for r in roots110 if 0 < float(r.evalf()) < 0.5]
    check("m1_c01_ex_zy30_06_10",
          set(f16_crit) == {-1, 1}
          and len(r_valid) == 1
          and sp.simplify(r_valid[0] - (7 - sp.sqrt(13)) / 12) == 0)

    # 7. zy30 Ex 1.14, 1.15, 1.17, 1.18, 1.19, 1.21 (P.29-47)
    # Ex 1.14: A = 1/6 + 2A -> A = -1/6
    A_sym = sp.Symbol("A")
    sol114 = sp.solve(A_sym - (sp.limit((x - sp.sin(x)) / x**3, x, 0) + 2 * A_sym), A_sym)[0]
    # Ex 1.15 & Ex 1.37 (CATCHES P.67 TYPO: x->1+ is -oo, NOT +oo!)
    f115 = sp.exp(1 / (x - 1)) * sp.log(sp.Abs(1 + x)) / ((sp.exp(x) - 1) * (x - 2))
    l115_m = sp.limit(f115, x, 1, "-")
    l115_p = sp.limit(f115, x, 1, "+")
    l137_0 = sp.limit(f115, x, 0)
    l137_m1 = sp.limit(f115, x, -1, "+")
    # Ex 1.17: x*sin(x-3)/((x-1)*(x-3)^2)
    f117 = x * sp.sin(x - 3) / ((x - 1) * (x - 3)**2)
    l117_m1 = sp.limit(f117, x, -1, "+")
    l117_0 = sp.limit(f117, x, 0, "-")
    l117_1 = sp.limit(sp.Abs(f117), x, 1, "+")
    l117_3 = sp.limit(sp.Abs(f117), x, 3, "-")
    # Ex 1.19: e^(tan x) - e^(sin x) ~ x^3/2
    l119 = sp.limit((sp.exp(sp.tan(x)) - sp.exp(sp.sin(x))) / x**3, x, 0)
    # Ex 1.21: lim sin(x)/(e^x - 1)*(cos x - b) = 5 -> b = -4
    b121 = sp.solve(sp.limit(sp.sin(x) / (sp.exp(x) - 1) * (sp.cos(x) - b), x, 0) - 5, b)[0]
    check("m1_c01_ex_zy30_14_21",
          sol114 == sp.Rational(-1, 6)
          and l115_m == 0 and l115_p == -sp.oo
          and sp.simplify(l137_0 - (-1 / (2 * sp.E))) == 0 and l137_m1 == -sp.oo
          and l117_m1.is_finite and l117_0 == 0 and l117_1 == sp.oo and l117_3 == sp.oo
          and l119 == sp.Rational(1, 2) and b121 == -4,
          f"Ex1.37 x->1+={l115_p} (book P.67 typo +oo), x->0={l137_0}")

    # 8. zy30 Ex 1.22, 1.23, 1.24, 1.26, 1.27 (P.49-58)
    l123_1 = sp.limit(xp / sp.log(xp)**10, xp, sp.oo)
    l123_2 = sp.limit(sp.exp(xp / 10) / xp, xp, sp.oo)
    l124 = sp.limit(((1 + xp)**(1 / xp) - sp.E) / xp, xp, 0, "+")
    # Ex 1.27: tan(2x) + x*f(x) = o(x^3) -> f(x) = (-tan(2x) + o(x^3))/x -> (2+f(x))/x^2 -> -8/3
    l127 = sp.limit((2 - sp.tan(2 * x) / x) / x**2, x, 0)
    check("m1_c01_ex_zy30_22_27",
          l123_1 == sp.oo and l123_2 == sp.oo
          and sp.simplify(l124 + sp.E / 2) == 0
          and l127 == sp.Rational(-8, 3))

    # 9. zy30 Ex 1.28 - Ex 1.36 & Ex 1.38 (P.58-68)
    l128 = sp.limit(sp.log(x) * sp.log(1 - x), x, 1, "-")
    # Ex 1.29 numeric squeeze check near 0+ and 0-
    v129_p = 1e-6 * math.floor(10 / 1e-6)
    v129_m = (-1e-6) * math.floor(10 / (-1e-6))
    l130 = sp.limit(1 / sp.log(1 + x) - 1 / x, x, 0)
    # Ex 1.31 BOTH +oo and -oo!
    l131_p = sp.limit(x**2 * (sp.exp(1 / x) - 1) - x, x, sp.oo)
    l131_m = sp.limit(x**2 * (sp.exp(1 / x) - 1) - x, x, -sp.oo)
    l132 = sp.limit((xp + sp.sqrt(1 + xp**2))**(1 / xp), xp, sp.oo)
    l133 = sp.limit(((sp.exp(x) + sp.exp(2 * x) + sp.exp(3 * x)) / 3)**(sp.E / x), x, 0)
    # Ex 1.34 & Ex 1.35
    s134 = sp.series(sp.exp(x), x, 0, 3).removeO()
    s135 = sp.series(sp.sin(x) / (1 + x**2), x, 0, 4).removeO()
    l136 = sp.limit(sp.cos(x)**(1 / x**2), x, 0)
    check("m1_c01_ex_zy30_28_38",
          l128 == 0
          and abs(v129_p - 10.0) < 1e-4 and abs(v129_m - 10.0) < 1e-4
          and l130 == sp.Rational(1, 2)
          and l131_p == sp.Rational(1, 2) and l131_m == sp.Rational(1, 2)
          and l132 == 1 and sp.simplify(l133 - sp.exp(2 * sp.E)) == 0
          and sp.simplify(s134 - (1 + x + x**2 / 2)) == 0
          and sp.simplify(s135 - (x - sp.Rational(7, 6) * x**3)) == 0
          and sp.simplify(l136 - sp.exp(-sp.Rational(1, 2))) == 0)

    # 10. zy18 Intensive Ex 1.1 - Ex 1.6 (P.2-7)
    li11 = sp.limit((xp**xp - 1) / (sp.log(xp) * sp.log(1 - xp)), xp, 0, "+")
    # Ex 1.2: int_0^x sin(t)/t dt / x -> 1, plus 1/x - 1/ln(1+x) -> -1/2 -> total 1/2
    li12 = sp.limit(sp.sin(x) / x, x, 0) + sp.limit(1 / x - 1 / sp.log(1 + x), x, 0)
    # Ex 1.3: lim_{t->x} ((x-1)/(t-1))^(t/(x-t)) = exp(x/(x-1))
    li13 = sp.limit(((x - 1) / (t - 1))**(t / (x - t)), t, x)
    # Ex 1.4: cos(e^(x^2/2)-1) - 1 ~ -x^4/8 -> c*k = -1/2
    s_i14 = sp.series(sp.cos(sp.exp(x**2 / 2) - 1) - 1, x, 0, 5).removeO()
    # Ex 1.6: int_0^x ln(1+t^2) dt / x^alpha: near 0+ is x^(3-a)/3 -> a<=3; near +oo is ln(1+x^2)/(a*x^(a-1)) -> a>1
    l_i16_0 = sp.limit(sp.log(1 + xp**2) / (3 * xp**2), xp, 0, "+")
    l_i16_inf_a1 = sp.limit(sp.log(1 + xp**2) / (1 * xp**0), xp, sp.oo)
    l_i16_inf_a2 = sp.limit(sp.log(1 + xp**2) / (2 * xp**1), xp, sp.oo)
    check("m1_c01_ex_zy18_01_06",
          li11 == -1 and li12 == sp.Rational(1, 2)
          and sp.simplify(li13 - sp.exp(x / (x - 1))) == 0
          and sp.simplify(s_i14 - (-x**4 / 8)) == 0
          and l_i16_0 == sp.Rational(1, 3) and l_i16_inf_a1 == sp.oo and l_i16_inf_a2 == 0)

    # 11. zy18 Intensive Ex 1.7 - Ex 1.15 (P.7-13)
    li17 = sp.limit(n**2 * (2**(1 / n) - 2**(1 / (n + 1))), n, sp.oo)
    li18 = sp.limit(xp**2 * (3**(1 / xp) - 3**(1 / (xp + 1))), xp, sp.oo)
    li19_p = sp.limit(x * sp.log(sp.E + 1 / (x - 1)) - x, x, sp.oo)
    li19_m = sp.limit(x * sp.log(sp.E + 1 / (x - 1)) - x, x, -sp.oo)
    # Ex 1.10: x^(1+x)/(1+x)^x - x/e with t = 1/x -> 0+
    tp = sp.Symbol("tp", positive=True)
    li110 = sp.limit((1 / (tp * sp.E)) * (sp.exp(1 - sp.log(1 + tp) / tp) - 1), tp, 0, "+")
    # Ex 1.11: |x|^(1/((1-x)(x-2))) at x=0, 1, 2-, 2+
    fi111 = sp.Abs(x)**(1 / ((1 - x) * (x - 2)))
    li111_0 = sp.limit(fi111, x, 0, "+")
    li111_1 = sp.limit(fi111, x, 1)
    li111_2m = sp.limit(fi111, x, 2, "-")
    li111_2p = sp.limit(fi111, x, 2, "+")
    # Ex 1.14 & Ex 1.15
    int_i14 = sp.integrate(x, (x, 0, 1)) + sp.integrate(2 - x, (x, 1, 2))  # period=2, area=1 -> average=1/2
    int_i15 = sp.integrate(1 / (1 + x**2), (x, 0, sp.oo))
    check("m1_c01_ex_zy18_07_15",
          li17 == sp.log(2) and li18 == sp.log(3)
          and li19_p == 1 / sp.E and li19_m == 1 / sp.E
          and sp.simplify(li110 - 1 / (2 * sp.E)) == 0
          and li111_0 == sp.oo and li111_1 == sp.E and li111_2m == sp.oo and li111_2p == 0
          and int_i14 / 2 == sp.Rational(1, 2)
          and 1 + int_i15 == 1 + sp.pi / 2)

    # ==================== Calculus Lecture 2: Sequence Limits ====================
    ni, ki = sp.symbols("ni ki", integer=True, positive=True)

    # 12. Common finite sums, telescoping & (1+1/n)^n - e ~ -e/(2n) (zy30 P.78-79, 85)
    s_k = sp.summation(ki, (ki, 1, ni))
    s_k2 = sp.summation(ki**2, (ki, 1, ni))
    s_tel = sp.summation(1 / (ki * (ki + 1)), (ki, 1, ni))
    lim_euler_seq = sp.limit((1 + 1 / ni)**ni, ni, sp.oo)
    lim_euler_eq = sp.limit(((1 + 1 / ni)**ni - sp.E) / (1 / ni), ni, sp.oo)
    check("m1_c02_f_sums_and_e",
          sp.simplify(s_k - ni * (ni + 1) / 2) == 0
          and sp.simplify(s_k2 - ni * (ni + 1) * (2 * ni + 1) / 6) == 0
          and sp.simplify(s_tel - ni / (ni + 1)) == 0
          and sp.limit(s_tel, ni, sp.oo) == 1
          and lim_euler_seq == sp.E
          and sp.simplify(lim_euler_eq + sp.E / 2) == 0)

    # 13. 12 inequalities + max-base squeeze theorem formulas (zy30 P.86-88)
    ineq12_ok = True
    for xv in (0.05, 0.25, 0.55, 0.75):
        if not (math.sin(xv) < xv < math.tan(xv)
                and xv < math.tan(xv) < (4.0 / math.pi) * xv
                and math.sin(xv) > (2.0 / math.pi) * xv
                and math.atan(xv) <= xv <= math.asin(xv)):
            ineq12_ok = False
    for xv in (0.1, 0.5, 1.5, 6.0):
        if not (math.exp(xv) >= xv + 1
                and xv - 1 >= math.log(xv)
                and 1 / (1 + xv) < math.log(1 + 1 / xv) < 1 / xv
                and xv / (1 + xv) < math.log(1 + xv) < xv):
            ineq12_ok = False
    lim_max_base = sp.limit((sp.Integer(2)**ni + sp.Integer(5)**ni + sp.Integer(3)**ni)**(1 / ni), ni, sp.oo)
    lim_inv_base = sp.limit((sp.Rational(1, 2)**ni + sp.Rational(1, 5)**ni)**(1 / ni), ni, sp.oo)
    check("m1_c02_f_ineq_and_squeeze", ineq12_ok and lim_max_base == 5 and lim_inv_base == sp.Rational(1, 2))

    # 14. zy30 Ex 2.1 & Ex 2.12: x_{n+1} = sqrt(x_n*(3 - x_n)), 0 < x_1 < 3
    ex21_12_ok = True
    for x0 in (0.05, 0.7, 1.5, 2.8):
        seq = [x0]
        for _ in range(35):
            seq.append(math.sqrt(seq[-1] * (3.0 - seq[-1])))
        if not (all(0 < v <= 1.5 + 1e-12 for v in seq[1:])
                and all(seq[i + 1] >= seq[i] - 1e-12 for i in range(1, len(seq) - 1))
                and abs(seq[-1] - 1.5) < 1e-6):
            ex21_12_ok = False
    check("m1_c02_ex_zy30_01_12", ex21_12_ok)

    # 15. zy30 Ex 2.4 - Ex 2.11 (P.81-89)
    l24_e = sp.limit(1 / (2 * ni), ni, sp.oo)
    l24_o = sp.limit(((2 * ni - 1)**2 + sp.sqrt(2 * ni - 1)) / (2 * ni - 1), ni, sp.oo)
    a25 = [1.0 - ((-1)**m) / m for m in range(1, 200)]
    sol26 = sp.solve([a + b - 1, a - b - 3], [a, b])
    l28 = sp.limit(sp.cos(1 / sp.sqrt(ni))**ni, ni, sp.oo)
    v29 = sum(i / (4000**2 + 4000 + i) for i in range(1, 4001))
    # Ex 2.11: cos(a_n) - a_n = cos(b_n), b_n = 1e-4 -> a_n / b_n^2 -> 1/2
    bv211 = 1e-4
    lo, hi = 0.0, bv211
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if math.cos(mid) - mid > math.cos(bv211):
            lo = mid
        else:
            hi = mid
    r211 = (0.5 * (lo + hi)) / (bv211**2)
    check("m1_c02_ex_zy30_04_11",
          l24_e == 0 and l24_o == sp.oo
          and max(a25) == 2.0 and min(a25) == 0.5
          and sol26[a] == 2 and sol26[b] == -1
          and sp.simplify(l28 - sp.exp(-sp.Rational(1, 2))) == 0
          and abs(v29 - 0.5) < 2e-4
          and abs(r211 - 0.5) < 1e-4)

    # 16. zy30 Ex 2.14 & Ex 2.15 (monotone log recurrence + cos contraction mapping, P.91-94)
    xi214 = float(sp.nsolve(x - 2 * sp.log(1 + x), x, 2.5))
    s214_hi, s214_lo = [5.0], [0.5]
    for _ in range(45):
        s214_hi.append(2 * math.log(1 + s214_hi[-1]))
        s214_lo.append(2 * math.log(1 + s214_lo[-1]))
    a215 = float(sp.nsolve(x - sp.cos(x), x, 0.7))
    ex215_ok = 0 < a215 < math.pi / 3
    for x0 in (-1.0, -0.3, 0.2, 1.0):
        v = x0
        for step in range(1, 45):
            v = math.cos(v)
            if abs(v - a215) > (math.sqrt(3) / 2)**step * abs(x0 - a215) + 1e-12:
                ex215_ok = False
        if abs(v - a215) > 1e-5:
            ex215_ok = False
    check("m1_c02_ex_zy30_14_15",
          abs(s214_hi[-1] - xi214) < 1e-6 and abs(s214_lo[-1] - xi214) < 1e-6
          and all(s214_hi[i] > s214_hi[i + 1] > xi214 - 1e-12 for i in range(35))
          and all(s214_lo[i] < s214_lo[i + 1] < xi214 + 1e-12 for i in range(35))
          and ex215_ok,
          f"Ex2.14 xi={xi214:.5f}, Ex2.15 a={a215:.5f}")

    # 17. zy18 Intensive Ex 2.1 - Ex 2.6 (P.15-17; verifies Ex 2.4 n>=2 monotonicity!)
    si21 = [1.0]
    si22 = [math.pi]
    si23 = [0.8]
    si24_lo = [1.2]
    si25 = [1.8]
    for _ in range(35):
        si21.append(math.sqrt(4 + 3 * si21[-1]))
        si22.append(math.pi / 2 + 0.5 * math.cos(si22[-1]))
        si23.append(1 - math.sqrt(1 - si23[-1]))
        si24_lo.append(si24_lo[-1]**2 / (2 * (si24_lo[-1] - 1)))
        si25.append(math.sqrt(si25[-1] * (2 - si25[-1])))
    H_sum = 0.0
    euler_seq = []
    for m in range(1, 300):
        H_sum += 1.0 / m
        euler_seq.append(H_sum - math.log(m))
    check("m1_c02_ex_zy18_01_06",
          abs(si21[-1] - 4.0) < 1e-6
          and abs(si22[-1] - math.pi / 2) < 1e-6
          and abs(si23[-1]) < 1e-7 and abs(si23[-1] / si23[-2] - 0.5) < 1e-6
          and si24_lo[1] > si24_lo[0]  # catches zy18 P.16 missing n>=2 note!
          and all(si24_lo[i + 1] <= si24_lo[i] + 1e-12 for i in range(1, len(si24_lo) - 1))
          and abs(si24_lo[-1] - 2.0) < 1e-10
          and abs(si25[-1] - 1.0) < 1e-7
          and all(euler_seq[i + 1] < euler_seq[i] for i in range(len(euler_seq) - 1))
          and euler_seq[-1] > 0,
          f"Ex2.4 x1=1.2->x2={si24_lo[1]:.2f}>x1 (dec for n>=2), Euler~{euler_seq[-1]:.4f}")

    # 18. zy18 Intensive Ex 2.7 - Ex 2.13 (P.18-22; verifies Ex 2.10 vacuous condition!)
    sol_i27 = sp.solve((1 - a) * a - sp.Rational(1, 4), a)
    s_i28_lo, s_i28_hi = [2.2], [8.0]
    for _ in range(25):
        s_i28_lo.append(s_i28_lo[-1] - (s_i28_lo[-1] - 3) / (s_i28_lo[-1] - 2))
        s_i28_hi.append(s_i28_hi[-1] - (s_i28_hi[-1] - 3) / (s_i28_hi[-1] - 2))
    l_i29 = sp.limit((x - sp.sin(x)) / x**3, x, 0)
    l_i210 = sp.limit(sp.exp(sp.log(1 - xp) * sp.log(xp)), xp, 0, "+")
    vacuous_210 = all(math.cos(i * 0.005) > 1 - i * 0.005 for i in range(1, 200))
    l_i213_ok = (
        sp.limit((1 + xp)**(1 / xp), xp, sp.oo) == 1
        and sp.limit((1 + 1 / xp)**xp, xp, sp.oo) == sp.E
        and sp.limit(xp * sp.log(xp), xp, 0, "+") == 0
        and sp.limit(xp * sp.log(xp), xp, sp.oo) == sp.oo
        and sp.limit(sp.log(xp) / xp, xp, sp.oo) == 0
    )
    check("m1_c02_ex_zy18_07_13",
          sol_i27 == [sp.Rational(1, 2)]
          and abs(s_i28_lo[-1] - 3.0) < 1e-8 and abs(s_i28_hi[-1] - 3.0) < 1e-8
          and l_i29 == sp.Rational(1, 6)
          and l_i210 == 1 and vacuous_210
          and l_i213_ok,
          "Ex2.10 alg=1 & (1-b)^n<cos(b) vacuous on (0,1) verified")

    # ==================== Calculus Lecture 3: Concepts of Differentiation ====================

    # 19. zy30 Ex 3.1 notes: odd/even/periodic higher derivatives at 0 & 2*pi (P.103)
    d2_31n1 = sp.diff(sp.log((1 - x) / (1 + x)), x, 2).subs(x, 0)
    d4_31n2 = sp.simplify(sp.diff(1 / (2**x + 1), x, 4).subs(x, 0))
    d5_31n3 = sp.simplify(sp.diff(sp.sin(sp.cos(x)) + sp.cos(sp.sin(x)), x, 5).subs(x, 2 * sp.pi))
    check("m1_c03_ex_zy30_01_notes", d2_31n1 == 0 and d4_31n2 == 0 and d5_31n3 == 0)

    # 20. zy30 Ex 3.3 & Ex 3.4: squeeze diff at 0 & product rule at single zero (P.104-105)
    l33_0 = sp.limit((1 - sp.cos(xp)) / xp, xp, 0, "+")
    ex34_ok = True
    for nv in (1, 2, 3, 4, 5):
        pe = sp.Integer(1)
        for kv in range(1, nv + 1):
            pe *= (sp.exp(kv * x) - kv)
        if sp.simplify(sp.diff(pe, x).subs(x, 0) - ((-1)**(nv - 1)) * sp.factorial(nv - 1)) != 0:
            ex34_ok = False
    check("m1_c03_ex_zy30_03_04", l33_0 == 0 and ex34_ok)

    # 21. zy30 Ex 3.6: exact factorizations & non-differentiable point counts (P.105-106)
    r36_1 = set(sp.solve(x**3 + x**2 - 2 * x - 2, x))
    r36_2 = set(sp.solve(x**3 - 2 * x**2 - x + 2, x))
    r36_3 = set(sp.solve(x**3 + 3 * x**2 - 2 * x - 6, x))
    outer_36 = {-1, 1}
    check("m1_c03_ex_zy30_06",
          r36_1 - outer_36 == {-sp.sqrt(2), sp.sqrt(2)}
          and r36_2 - outer_36 == {2}
          and r36_3 - outer_36 == {-3, -sp.sqrt(2), sp.sqrt(2)},
          "n2=1 < n1=2 < n3=3")

    # 22. zy30 Ex 3.8, Ex 3.10, Ex 3.11 (tangent intercept + differentials, P.110-114)
    l38_e = sp.limit((1 - 1 / ni)**ni, ni, sp.oo)
    ys = sp.Symbol("ys", positive=True)
    v310 = (ys / (x + sp.sqrt(x**2 + ys**2))).subs({x: 0, ys: 1})
    sol311 = sp.solve(sp.Rational(1, 10) - 2 * (-1) * a * sp.Rational(-1, 10), a)[0]
    check("m1_c03_ex_zy30_08_11",
          sp.simplify(l38_e - 1 / sp.E) == 0 and v310 == 1 and sol311 == sp.Rational(1, 2))

    # 23. zy18 Intensive Ex 3.1 & Ex 3.2 (P.24-26)
    df_i31 = sp.diff(2 - x**x - x, x).subs(x, 1)
    l_i32_a = sp.limit(sp.sqrt(xp) / xp**sp.Rational(1, 3), xp, 0, "+")
    l_i32_b = sp.limit(sp.sqrt(xp) / xp, xp, 0, "+")
    check("m1_c03_ex_zy18_01_02", df_i31 == -2 and l_i32_a == 0 and l_i32_b == sp.oo)

    # 24. zy18 Intensive Ex 3.4 & Ex 3.5 + asymptote counterexample (P.29-31)
    f_i35 = sp.cbrt(x**2) * sp.sin(x)
    l_i35_p = sp.limit(f_i35 / x, x, 0, "+")
    l_i35_m = sp.limit(f_i35 / x, x, 0, "-")
    df_as = sp.diff(sp.sin(x**2) / x, x)
    check("m1_c03_ex_zy18_04_05",
          l_i35_p == 0 and l_i35_m == 0
          and sp.simplify(df_as - (2 * sp.cos(x**2) - sp.sin(x**2) / x**2)) == 0)

    # 25. zy18 Intensive Ex 3.7 & Ex 3.8 (P.32-33)
    l37_A = sp.limit((xp * sp.tan(xp)) / xp, xp, 0, "+")
    l37_B = sp.limit((xp * sp.tan(sp.sqrt(xp))) / xp, xp, 0, "+")
    l37_C = sp.limit((sp.sqrt(sp.cos(xp)) - 1) / xp, xp, 0, "+")
    l37_Dp = sp.limit((sp.cos(sp.sqrt(xp)) - 1) / xp, xp, 0, "+")
    l37_Dm = sp.limit((sp.cos(sp.sqrt(xp)) - 1) / (-xp), xp, 0, "+")
    g38 = lambda s: 1 + s + sp.cos(s)
    f38 = sp.integrate(g38(xp * t), (t, 0, xp**2))
    df38 = sp.simplify(sp.diff(f38, xp))
    df38_form = sp.simplify(-sp.integrate(g38(u), (u, 0, xp**3)) / xp**2 + 3 * xp * g38(xp**3))
    check("m1_c03_ex_zy18_07_08",
          l37_A == 0 and l37_B == 0 and l37_C == 0
          and l37_Dp == sp.Rational(-1, 2) and l37_Dm == sp.Rational(1, 2)
          and sp.simplify(df38 - df38_form) == 0
          and sp.limit(f38 / xp, xp, 0, "+") == 0 and sp.limit(df38, xp, 0, "+") == 0,
          "Ex3.7 D: f'_+=-1/2 != f'_-=+1/2; Ex3.8 f'(0)=lim f'(x)=0")

    # ==================== Calculus Lecture 4: Differentiation Calculations ====================
    y = sp.Symbol("y", real=True)

    # 26. Core hyperbolic log derivatives (zy30 P.120)
    d_c4_asinh = sp.simplify(sp.diff(sp.log(x + sp.sqrt(x**2 + 1)), x) - 1 / sp.sqrt(x**2 + 1))
    d_c4_acosh = sp.simplify(sp.diff(sp.log(xp + sp.sqrt(xp**2 - 1)), xp) - 1 / sp.sqrt(xp**2 - 1))
    check("m1_c04_f_core_derivs", d_c4_asinh == 0 and d_c4_acosh == 0)

    # 27. zy30 Ex 4.1 - Ex 4.6 (P.121-125; test Ex 4.1 with even N=6 -> -pi*5!/2 matching N=100)
    p41 = sp.Integer(1)
    for kv in range(1, 7):
        p41 *= (sp.tan(sp.pi * x**kv / 4) - kv)
    v41 = sp.simplify(sp.diff(p41, x).subs(x, 1))
    v42 = sp.simplify(sp.diff(sp.log(x + sp.sqrt(x**2 + ap**2)), x).subs(x, 0))
    d44 = sp.simplify(sp.diff(sp.exp(sp.sin(sp.log(xp))), xp) - sp.exp(sp.sin(sp.log(xp))) * sp.cos(sp.log(xp)) / xp)
    d46_p2 = sp.simplify(sp.diff(xp * sp.exp(-xp), xp, 2) - sp.exp(-xp) * (xp - 2))
    d46_m2 = sp.simplify(sp.diff(-x * sp.exp(-x), x, 2) - sp.exp(-x) * (2 - x))
    check("m1_c04_ex_zy30_01_06",
          v41 == -sp.pi * sp.factorial(5) / 2 and v42 == 1 / ap
          and d44 == 0 and d46_p2 == 0 and d46_m2 == 0)

    # 28. zy30 Ex 4.7 - Ex 4.13 (inverse, implicit, parametric, power-exp, P.126-129)
    f47 = 3 * x**2 + sp.exp(x)
    phi2_47 = sp.simplify((-sp.diff(f47, x, 2) / sp.diff(f47, x)**3).subs(x, 1))
    F48 = y**3 + x * y**2 + x**2 * y + 6
    y1_48 = sp.simplify(-sp.diff(F48, x) / sp.diff(F48, y))
    y2_48 = sp.simplify(sp.diff(y1_48, x) + sp.diff(y1_48, y) * y1_48)
    v48 = sp.simplify(y2_48.subs({x: 1, y: -2}))
    dy49 = sp.simplify(sp.diff(t * sp.sin(t) + sp.cos(t), t) / sp.diff(sp.sin(t), t))
    v49 = sp.simplify((sp.diff(dy49, t) / sp.diff(sp.sin(t), t)).subs(t, sp.pi / 4))
    G410 = 2 * y - t * y**2 + sp.exp(t) - 5
    dydx_410 = sp.simplify((-sp.diff(G410, t) / sp.diff(G410, y)) / sp.diff(sp.atan(t), t))
    d412 = sp.simplify(sp.diff(xp**xp, xp) - xp**xp * (1 + sp.log(xp)))
    d413 = sp.simplify(sp.diff(xp**(1 / xp), xp) - xp**(1 / xp - 2) * (1 - sp.log(xp)))
    check("m1_c04_ex_zy30_07_13",
          sp.simplify(phi2_47 + 1 / (6 + sp.E)**2) == 0
          and v48 == sp.Rational(4, 9)
          and dy49 == t and v49 == sp.sqrt(2)
          and sp.simplify(dydx_410 - (y**2 - sp.exp(t)) * (1 + t**2) / (2 * (1 - t * y))) == 0
          and d412 == 0 and d413 == 0)

    # 29. zy30 Ex 4.14 - Ex 4.19 (Higher derivatives, P.129-134)
    ex415_ok = all(sp.simplify(sp.diff((1 - x) / (1 + x), x, nv).subs(x, 0) - ((-1)**nv * 2 * sp.factorial(nv))) == 0 for nv in range(1, 6))
    f416 = -1 / (x - 2)
    ex416_ok = all(sp.simplify(sp.diff(f416, x, nv) - sp.factorial(nv) * f416**(nv + 1)) == 0 for nv in range(1, 5))
    ex417_ok = all(sp.simplify(sp.diff(x * sp.exp(x), x, nv) - (nv + x) * sp.exp(x)) == 0 for nv in range(1, 5))
    ex418_ok = all(sp.simplify(sp.diff(x**2 * sp.log(1 - x), x, nv).subs(x, 0) - (-sp.factorial(nv) / (nv - 2))) == 0 for nv in range(3, 7))
    ex419_ok = all(sp.simplify(sp.diff(x**2 * 2**x, x, nv).subs(x, 0) - (nv * (nv - 1) * sp.log(2)**(nv - 2))) == 0 for nv in range(1, 6))
    check("m1_c04_ex_zy30_14_19", ex415_ok and ex416_ok and ex417_ok and ex418_ok and ex419_ok)

    # 30. zy18 Intensive Ex 4.1 - Ex 4.5 (P.35-38)
    fi41 = x / (2 * x + 1)
    exi41_ok = (sp.simplify(x**2 * sp.diff(fi41, x) - fi41**2) == 0 and fi41.subs(x, 1) == sp.Rational(1, 3)
                and all(sp.simplify(sp.diff(fi41, x, nv).subs(x, 0) - ((-2)**(nv - 1) * sp.factorial(nv))) == 0 for nv in range(1, 6)))
    fi42 = 1 / (x * (1 - x))
    exi42_ok = all(sp.simplify(sp.diff(fi42, x, nv) - (((-1)**nv / x**(nv + 1) + 1 / (1 - x)**(nv + 1)) * sp.factorial(nv))) == 0 for nv in range(1, 5))
    exi43_ok = all(sp.simplify(sp.diff((x**3 - 1)**nv, x, nv).subs(x, 1) - (3**nv * sp.factorial(nv))) == 0 for nv in range(1, 5))
    fi44 = x / (1 + x**2)
    exi44_ok = all(sp.simplify(sp.diff(fi44, x, 2 * kv + 1).subs(x, 0) - ((-1)**kv * sp.factorial(2 * kv + 1))) == 0 and sp.diff(fi44, x, 2 * kv).subs(x, 0) == 0 for kv in range(1, 4))
    d2_45p = sp.limit(sp.diff((xp / 3) * sp.tan(xp / 3), xp, 2), xp, 0, "+")
    d2_45m = sp.limit(sp.diff(-x * sp.tan(x), x, 2), x, 0, "-")
    check("m1_c04_ex_zy18_01_05",
          exi41_ok and exi42_ok and exi43_ok and exi44_ok and d2_45p == sp.Rational(2, 9) and d2_45m == -2,
          f"Ex4.5 f''_+={d2_45p} != f''_-={d2_45m}")

    # 31. zy18 Intensive Ex 4.6 - Ex 4.9 (P.38-40)
    d4_i46 = sp.simplify(sp.diff(1 / (2**x + 1), x, 4).subs(x, 0))
    F47i = sp.exp(x) + y**2 + y - sp.log(1 + x) * sp.cos(y) - 1
    y1_47i = -sp.diff(F47i, x) / sp.diff(F47i, y)
    v47i = sp.simplify((sp.diff(y1_47i, x) + sp.diff(y1_47i, y) * y1_47i).subs({x: 0, y: 0}))
    dy48i = sp.diff(sp.exp(tp**2), tp) / sp.diff(sp.log(tp), tp)
    v48i = sp.simplify((sp.diff(dy48i, tp) / sp.diff(sp.log(tp), tp)).subs(tp, 1))
    f49i = sp.exp(x) + 2 * x + 1
    v49i = sp.simplify((-sp.diff(f49i, x, 2) / sp.diff(f49i, x)**3).subs(x, 0))
    check("m1_c04_ex_zy18_06_09",
          d4_i46 == 0 and v47i == -2 and v48i == 8 * sp.E and v49i == sp.Rational(-1, 27),
          f"Ex4.7 y''(0)={v47i}, Ex4.8={v48i}, Ex4.9={v49i}")

    # ==================== Calculus Lecture 5: Geometric Applications of Differentiation ====================
    a_c5 = sp.Symbol("a_c5", real=True)

    # 32. zy30 Ex 5.3 & Ex 5.4 (implicit extremum + n-th deriv extremum, P.144-145)
    F53 = y**3 + x * y**2 + x**2 * y + 6
    y1_53 = sp.simplify(-sp.diff(F53, x) / sp.diff(F53, y))
    y2_53 = sp.simplify(sp.diff(y1_53, x) + sp.diff(y1_53, y) * y1_53)
    f54 = (4 + x) * sp.exp(x)
    stat54 = sp.solve(sp.diff(f54, x), x)[0]
    check("m1_c05_ex_zy30_03_04",
          y1_53.subs({x: 1, y: -2}) == 0 and y2_53.subs({x: 1, y: -2}) == sp.Rational(4, 9)
          and stat54 == -5 and sp.diff(f54, x, 2).subs(x, stat54) == sp.exp(-5)
          and f54.subs(x, stat54) == -sp.exp(-5))

    # 33. zy30 Ex 5.6 - Ex 5.8 (inflection & extrema counts, P.149-151)
    y57 = (x - 1) * (x - 2)**2 * (x - 3)**3 * (x - 4)**4
    y57_d2 = sp.diff(y57, x, 2)
    r57_d2 = sp.real_roots(y57_d2, x)
    sign_chg_57 = len([r for r in set(r57_d2) if y57_d2.subs(x, r + sp.Rational(1, 1000)) * y57_d2.subs(x, r - sp.Rational(1, 1000)) < 0])
    r58_d2 = sp.real_roots(sp.diff((x - 1)**2 * (x - 3)**3, x, 2), x)
    r58b_d1 = sp.real_roots(sp.diff((x - 1)**2 * (x - 3)**2, x), x)
    r58b_d2 = sp.real_roots(sp.diff((x - 1)**2 * (x - 3)**2, x, 2), x)
    check("m1_c05_ex_zy30_06_08",
          y57_d2.subs(x, 3) == 0 and sp.diff(y57, x, 3).subs(x, 3) != 0
          and sign_chg_57 == 6 and len(r58_d2) == 3 and len(r58b_d1) == 3 and len(r58b_d2) == 2,
          f"Ex5.7 inflections={sign_chg_57}, Ex5.8={len(r58_d2)}, Ex5.8(2) ext/inf={len(r58b_d1)}/{len(r58b_d2)}")

    # 34. zy30 Ex 5.9 & Ex 5.10 (3 asymptotes + max of n^(1/n), P.153-156)
    f59 = 1 / x + sp.log(1 + sp.exp(x))
    check("m1_c05_ex_zy30_09_10",
          sp.limit(f59, x, 0, "+") == sp.oo and sp.limit(f59, x, -sp.oo) == 0
          and sp.limit(f59 / x, x, sp.oo) == 1 and sp.limit(f59 - x, x, sp.oo) == 0
          and sp.Integer(3)**2 > sp.Integer(2)**3)

    # 35. zy30 Ex 5.11 - Ex 5.15 (curve sketching & astroid curvature, P.156-160)
    d2_511 = sp.simplify(sp.diff((1 - x**2)**sp.Rational(3, 2), x, 2) - 3 * (2 * x**2 - 1) / sp.sqrt(1 - x**2))
    l512_0 = sp.limit(xp**xp, xp, 0, "+")
    d2_512 = sp.simplify(sp.diff(xp**xp, xp, 2) - xp**xp * ((1 + sp.log(xp))**2 + 1 / xp))
    d2_513 = sp.simplify(sp.diff(sp.exp(x) / x, x, 2) - sp.exp(x) * (x**2 - 2 * x + 2) / x**3)
    dy515 = sp.simplify(sp.diff(sp.sin(t)**3, t) / sp.diff(sp.cos(t)**3, t))
    d2y515 = sp.simplify(sp.diff(dy515, t) / sp.diff(sp.cos(t)**3, t))
    k515 = sp.simplify((sp.Abs(d2y515) / (1 + dy515**2)**sp.Rational(3, 2)).subs(t, sp.pi / 4))
    check("m1_c05_ex_zy30_11_15",
          d2_511 == 0 and l512_0 == 1 and d2_512 == 0 and d2_513 == 0 and k515 == sp.Rational(2, 3),
          f"astroid k(pi/4)={k515}")

    # 36. zy18 Intensive Ex 5.1 - Ex 5.3 (P.42-44)
    f_i51 = sp.sin(x) + x**2
    u_i51 = x - f_i51 / sp.diff(f_i51, x)
    l_i51 = sp.limit(x * f_i51.subs(x, u_i51) / (u_i51 * f_i51), x, 0)
    u_i51_lin = sp.simplify(x - x / sp.diff(x, x))
    f_i52 = (x**2 + a_c5) * sp.exp(x)
    C53 = (x - 1)**2 + y**2 - 2
    y1_c53 = -sp.diff(C53, x) / sp.diff(C53, y)
    y2_c53 = sp.diff(y1_c53, x) + sp.diff(y1_c53, y) * y1_c53
    check("m1_c05_ex_zy18_01_03",
          l_i51 == 1 and u_i51_lin == 0
          and sp.simplify(sp.diff(f_i52, x) / sp.exp(x) - (x**2 + 2 * x + a_c5)) == 0
          and sp.simplify(sp.diff(f_i52, x, 2) / sp.exp(x) - (x**2 + 4 * x + a_c5 + 2)) == 0
          and y1_c53.subs({x: 0, y: 1}) == 1 and y2_c53.subs({x: 0, y: 1}) == -2,
          "Ex5.1 u(f=x)=0 (u!=0 implicit); Ex5.3 y'(0)=1, y''(0)=-2")

    # 37. zy18 Intensive Ex 5.5 - Ex 5.9 (P.45-47)
    f_i55 = x * sp.log(sp.E + 1 / (x - 1))
    f_i56 = (1 + xp)**sp.Rational(3, 2) / sp.sqrt(xp)
    g_i57 = 3 - 6 / (sp.exp(2 * x) + 1)
    v_i58 = sp.integrate(t / (t**2 + 2 * t + 2), (t, 0, 1))
    exp_i58 = sp.Rational(1, 2) * sp.log(sp.Rational(5, 2)) - sp.atan(2) + sp.pi / 4
    check("m1_c05_ex_zy18_05_09",
          sp.limit(f_i55 / x, x, sp.oo) == 1 and sp.simplify(sp.limit(f_i55 - x, x, sp.oo) - 1 / sp.E) == 0
          and sp.limit(f_i56 / xp, xp, sp.oo) == 1 and sp.limit(f_i56 - xp, xp, sp.oo) == sp.Rational(3, 2)
          and sp.simplify(sp.Rational(1, 2) * sp.log((3 + g_i57) / (3 - g_i57)) - x) == 0
          and sp.limit(g_i57, x, sp.oo) == 3 and sp.limit(g_i57, x, -sp.oo) == -3
          and sp.simplify(v_i58 - exp_i58) == 0,
          f"Ex5.5-5.9 all exact; Ex5.8={sp.N(v_i58, 6)}")

    # ==================== Calculus Lecture 6: MVT, Differential Equations/Roots & Inequalities ====================
    a_c6, b_c6, k_c6 = sp.symbols("a_c6 b_c6 k_c6", real=True)
    bp_c6 = sp.Symbol("bp_c6", positive=True)
    f_c6 = sp.Function("f_c6")(xp)
    phi_c6 = sp.Function("phi_c6")(xp)
    Fint_c6 = sp.Function("Fint_c6")(xp)
    n_c6 = sp.Symbol("n_c6", real=True)

    # 38. Auxiliary-function derivative identities & zy18 Ex 6.3 e^(1-x) check (P.168-169, P.49-52)
    id6_1 = sp.simplify(sp.diff(f_c6 * xp**n_c6, xp) - xp**(n_c6 - 1) * (xp * sp.diff(f_c6, xp) + n_c6 * f_c6))
    id6_2 = sp.simplify(sp.diff(f_c6 * sp.exp(phi_c6), xp) - sp.exp(phi_c6) * (sp.diff(f_c6, xp) + f_c6 * sp.diff(phi_c6, xp)))
    id6_3 = sp.simplify(sp.diff(f_c6 * sp.exp(xp), xp, 2) - sp.exp(xp) * (sp.diff(f_c6, xp, 2) + 2 * sp.diff(f_c6, xp) + f_c6))
    id6_4 = sp.simplify(sp.diff(sp.diff(f_c6, xp) / f_c6, xp) - (sp.diff(f_c6, xp, 2) * f_c6 - sp.diff(f_c6, xp)**2) / f_c6**2)
    id6_5 = sp.simplify(sp.diff(sp.log(f_c6), xp, 2) - (sp.diff(f_c6, xp, 2) * f_c6 - sp.diff(f_c6, xp)**2) / f_c6**2)
    id6_63 = sp.simplify(sp.diff(xp * sp.exp(1 - xp) * f_c6, xp) - sp.exp(1 - xp) * (f_c6 - xp * f_c6 + xp * sp.diff(f_c6, xp)))
    id6_61 = sp.simplify(sp.diff(f_c6 * sp.exp(Fint_c6), xp).subs(sp.diff(Fint_c6, xp), f_c6) - sp.exp(Fint_c6) * (sp.diff(f_c6, xp) + f_c6**2))
    check("m1_c06_f_aux_derivs",
          id6_1 == 0 and id6_2 == 0 and id6_3 == 0 and id6_4 == 0 and id6_5 == 0 and id6_63 == 0 and id6_61 == 0,
          "All aux-func identities & zy18 Ex6.3 e^(1-x) exact")

    # 39. zy18 P.61-62 22 Taylor/formal expansions & margin limits
    t6_1 = sp.series(sp.sqrt(1 + x), x, 0, 3).removeO() == 1 + x / 2 - x**2 / 8
    t6_2 = sp.series(1 / sp.sqrt(1 + x), x, 0, 3).removeO() == 1 - x / 2 + sp.Rational(3, 8) * x**2
    t6_3 = sp.series(1 / (1 - x)**2, x, 0, 4).removeO() == 1 + 2 * x + 3 * x**2 + 4 * x**3
    t6_4 = sp.series(sp.tan(x), x, 0, 6).removeO() == x + x**3 / 3 + sp.Rational(2, 15) * x**5
    t6_5 = sp.series(sp.asin(x), x, 0, 6).removeO() == x + x**3 / 6 + sp.Rational(3, 40) * x**5
    t6_6 = sp.series(sp.atan(x), x, 0, 6).removeO() == x - x**3 / 3 + x**5 / 5
    t6_7 = sp.series(sp.log((1 + x) / (1 - x)), x, 0, 6).removeO() == 2 * (x + x**3 / 3 + x**5 / 5)
    t6_8 = sp.series(sp.log(x + sp.sqrt(x**2 + 1)), x, 0, 6).removeO() == x - x**3 / 6 + sp.Rational(3, 40) * x**5
    ml6_1 = sp.limit((sp.sqrt(1 + x**2) + sp.sqrt(1 - x**2) - 2) / (x**2 * (1 - sp.cos(x))), x, 0)
    ml6_2 = sp.limit((sp.tan(x) - x - x**3 / 3) / (x * (sp.exp(x**4) - 1)), x, 0)
    check("m1_c06_f_taylor_22",
          t6_1 and t6_2 and t6_3 and t6_4 and t6_5 and t6_6 and t6_7 and t6_8
          and ml6_1 == sp.Rational(-1, 2) and ml6_2 == sp.Rational(2, 15),
          f"margin lims={ml6_1}, {ml6_2}")

    # 40. zy30 Ex 6.5, 6.10, 6.12, 6.13 (MVT & Taylor, P.169-175)
    # Substitute u = b - x > 0 to verify Ex 6.5 auxiliary derivative cleanly
    d65 = sp.simplify(sp.powsimp(sp.expand(sp.diff(f_c6 * xp**ap, xp) - xp**(ap - 1) * (sp.diff(f_c6, xp) * xp + ap * f_c6)), force=True))
    f610 = sp.exp(-x) - 1
    v610_ok = all(
        float(sp.N(xv * f610.subs(x, bv))) > float(sp.N(bv * f610.subs(x, xv)))
        for av, xv, bv in [(sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)), (sp.Integer(1), sp.Integer(2), sp.Integer(4))]
    )
    f612 = (x - sp.Rational(1, 2))**2 - sp.Rational(1, 12)
    f613 = sp.Rational(1, 2) * (x**3 + 1)
    check("m1_c06_ex_zy30_05_13",
          d65 == 0 and v610_ok
          and sp.integrate(f612, (x, 0, 1)) == 0 and sp.diff(f612, x, 2) > 0 and f612.subs(x, sp.Rational(1, 2)) < 0
          and f613.subs(x, -1) == 0 and f613.subs(x, 1) == 1 and sp.diff(f613, x).subs(x, 0) == 0 and sp.diff(f613, x, 3) == 3)

    # 41. zy30 Ex 6.14 - Ex 6.18 (Roots, P.176-178)
    r614 = float(sp.nsolve(xp**sp.Rational(1, 4) + xp**sp.Rational(1, 2) - sp.cos(xp), 0.3))
    disc615 = sp.factor(sp.discriminant(5 * t**2 + 6 * a_c6 * t + 3 * b_c6, t))
    f616 = 2**x - x**2 - 1
    r616_3 = float(sp.nsolve(f616, 4.2))
    r618 = sp.real_roots(x**3 + 3 * x - 1, x)
    check("m1_c06_ex_zy30_14_18",
          0 < r614 < 1 and sp.simplify(disc615 - 12 * (3 * a_c6**2 - 5 * b_c6)) == 0
          and f616.subs(x, 0) == 0 and f616.subs(x, 1) == 0 and 2 < r616_3 < 5
          and len(r618) == 1 and 0 < float(r618[0].evalf()) < 1,
          f"Ex6.14=+-{r614:.4f}, Ex6.16 3rd={r616_3:.4f}")

    # 42. zy30 Ex 6.19 - Ex 6.21 (Inequalities, P.179-181)
    f620 = sp.log((1 + xp) / xp) - 1 / (1 + xp)
    df620 = sp.simplify(sp.diff(f620, xp) + 1 / (xp * (1 + xp)**2))
    g620 = f620 - 1 / (sp.sqrt(xp) * (1 + xp))
    dg620 = sp.simplify(sp.diff(g620, xp) - (1 + 3 * xp - 2 * sp.sqrt(xp)) / (2 * xp**sp.Rational(3, 2) * (1 + xp)**2))
    h620_min = (1 + 3 * xp - 2 * sp.sqrt(xp)).subs(xp, sp.Rational(1, 9))
    phi621 = sp.log(xp) - sp.log(ap) - (xp - ap) / sp.sqrt(ap * xp)
    dphi621 = sp.simplify(sp.diff(phi621, xp) + (sp.sqrt(xp) - sp.sqrt(ap))**2 / (2 * xp * sp.sqrt(ap * xp)))
    check("m1_c06_ex_zy30_19_21",
          df620 == 0 and dg620 == 0 and h620_min == sp.Rational(2, 3) and dphi621 == 0)

    # 43. zy18 Intensive Ex 6.13, 6.15, 6.17, 6.18 (P.59-66)
    l613 = sp.limit(sp.log((sp.exp(xp) - 1) / xp) / xp, xp, 0, "+")
    g615_1 = sp.integrate(-t * sp.sin(t**2), (t, 0, 1))
    f617 = ap * xp - bp_c6 * sp.log(xp)
    stat617 = sp.solve(sp.diff(f617, xp), xp)[0]
    min617 = sp.simplify(f617.subs(xp, stat617) - bp_c6 * (1 - sp.log(bp_c6 / ap)))
    v618_int = float(sp.N(sp.Si(-3 * sp.pi / 2) - sp.Si(1), 8))
    v618_F = v618_int - float(sp.N(2 * sp.log(3 * sp.pi / 2), 8))
    check("m1_c06_ex_zy18_09_18",
          l613 == sp.Rational(1, 2) and sp.simplify(g615_1 - (sp.cos(1) - 1) / 2) == 0
          and stat617 == bp_c6 / ap and min617 == 0 and v618_int < 0 and v618_F < 0,
          f"Ex6.13={l613}, Ex6.18 int={v618_int:.4f}, F(-3pi/2)={v618_F:.4f}")

    # 44. zy18 Intensive Ex 6.20, 6.21, 6.22 (P.70-71)
    g620i = xp - 2 * sp.log(xp) + 2 * k_c6
    stat620i = sp.solve(sp.diff(g620i, xp), xp)[0]
    x621 = sp.Rational(1, 3) * t**3 + t + sp.Rational(1, 3)
    y621 = sp.Rational(1, 3) * t**3 - t + sp.Rational(1, 3)
    dy621 = sp.simplify(sp.diff(y621, t) / sp.diff(x621, t))
    d2y621 = sp.simplify(sp.diff(dy621, t) / sp.diff(x621, t))
    pts621 = ((x621.subs(t, -1), y621.subs(t, -1)), (x621.subs(t, 1), y621.subs(t, 1)))
    f622 = t**3 + t
    a622 = sp.integrate(f622, (t, 0, 1))
    F622 = a622 * (1 - x**2) + sp.integrate(f622, (t, 1, x))
    check("m1_c06_ex_zy18_20_22",
          stat620i == 2 and sp.simplify(g620i.subs(xp, 2) - (2 - 2 * sp.log(2) + 2 * k_c6)) == 0
          and sp.simplify(dy621 - (t**2 - 1) / (t**2 + 1)) == 0
          and sp.simplify(d2y621 - 4 * t / (t**2 + 1)**3) == 0
          and pts621 == ((-1, 1), (sp.Rational(5, 3), sp.Rational(-1, 3)))
          and a622 > 0 and F622.subs(x, -1) == 0 and F622.subs(x, 0) == 0 and F622.subs(x, 1) == 0,
          f"Ex6.21 A={pts621[0]}, B={pts621[1]}; Ex6.22 a={a622}")

    # ==================== Calculus Lecture 7: Physical Applications & Economics (Extended) ====================
    v0_c7, Q_c7, P_c7 = sp.symbols("v0_c7 Q_c7 P_c7", positive=True)

    # 45. Core Math I/II physics examples: zy30 Ex 7.1 + zy18 Ex 7.1 & Ex 7.2
    dl71 = sp.simplify(sp.diff(sp.sqrt(xp**2 + xp**6), xp).subs(xp, 1) * v0_c7)
    # zy18 Ex 7.1: L^2 = x^2 + x^3 -> 2*L*dL/dt = (2x+3x^2)*dx/dt at x=3, dL/dt=11
    L71i = sp.sqrt(3**2 + 3**3)
    dxdt71i = sp.Rational(2 * L71i * 11, 2 * 3 + 3 * 3**2)
    # zy18 Ex 7.2: T(t) = 100*exp(-k*t) + 20, T(30)=30 -> k=ln(10)/30; T(t)=21 -> t=60 -> remaining=30
    k72i = sp.log(10) / 30
    T72i = 100 * sp.exp(-k72i * t) + 20
    t72i_end = sp.solve(sp.Eq(T72i, 21), t)[0]
    check("m1_c07_physics_core",
          dl71 == 2 * sp.sqrt(2) * v0_c7 and L71i == 6 and dxdt71i == 4
          and T72i.subs(t, 0) == 120 and T72i.subs(t, 30) == 30 and t72i_end == 60 and (t72i_end - 30) == 30,
          f"zy30 Ex7.1={dl71}, zy18 Ex7.1={dxdt71i} cm/s, Ex7.2 rem={t72i_end - 30} min")

    # 46. Extended Math III economics examples: zy30 Ex 7.2-7.4 + zy18 Ex 7.3-7.5
    r72 = sp.Rational(6, 100)
    t0_72 = sp.solve(sp.diff(sp.Rational(2, 5) * sp.sqrt(tp) - r72 * tp, tp), tp)[0]
    L73 = (60 - Q_c7 / 1000) * Q_c7 - (60000 + 20 * Q_c7)
    Q73_max = sp.solve(sp.diff(L73, Q_c7), Q_c7)[0]
    p73_max = 60 - Q73_max / 1000
    ER74 = sp.simplify(1 - 2 * P_c7**2 / (192 - P_c7**2)).subs(P_c7, 6)
    Cbar73i = 25000 / xp + 200 + xp / 40
    x73i_min = sp.solve(sp.diff(Cbar73i, xp), xp)[0]
    L74i_left = (25 - sp.Rational(1, 4) * Q_c7) * Q_c7 - (150 + 5 * Q_c7 + sp.Rational(1, 4) * Q_c7**2)
    L74i_right = (35 - sp.Rational(3, 4) * Q_c7) * Q_c7 - (150 + 5 * Q_c7 + sp.Rational(1, 4) * Q_c7**2)
    R75i = P_c7 * (100 - 5 * P_c7)
    eta75i = sp.simplify(sp.diff(R75i, P_c7) * P_c7 / R75i)
    check("m1_c07_econ_extended",
          t0_72 == sp.Rational(100, 9) and sp.diff(L73, Q_c7).subs(Q_c7, 10000) == 20 and Q73_max == 20000 and p73_max == 40
          and ER74 == sp.Rational(7, 13) and x73i_min == 1000
          and L74i_left.subs(Q_c7, 20) == 50 and L74i_right.subs(Q_c7, 20) == 50
          and sp.simplify(eta75i.subs(P_c7, 3) * sp.Rational(85, 10) - 7) == 0
          and sp.simplify(eta75i.subs(P_c7, 12) * 5 + sp.Rational(5, 2)) == 0,
          f"Ex7.2 t0={t0_72}, Ex7.4={ER74}, Ex7.3i={x73i_min}, Ex7.4i max=50")

    # ==================== Calculus Lecture 8: Integral Concepts, Properties & Improper Integrals ====================
    # 47. zy30 Ex 8.1 - Ex 8.3 (Antiderivatives vs Riemann integrability, P.196-209)
    d81 = sp.simplify(sp.diff(sp.asin(x), x) / (1 - x**2) - (1 - x**2)**sp.Rational(-3, 2))
    F82_l = sp.log(sp.sqrt(1 + x**2) + x) + 1
    F82_r = (x + 1) * sp.sin(x) + sp.cos(x)
    d83_2 = sp.simplify(sp.diff(x**2 * sp.sin(1 / x**2), x) - (2 * x * sp.sin(1 / x**2) - (2 / x) * sp.cos(1 / x**2)))
    d83_4 = sp.simplify(sp.diff(x**2 * sp.cos(1 / x), x) - (2 * x * sp.cos(1 / x) + sp.sin(1 / x)))
    check("m1_c08_ex_zy30_01_03",
          d81 == 0 and F82_l.subs(x, 0) == 1 and F82_r.subs(x, 0) == 1
          and sp.simplify(sp.diff(F82_l, x) - 1 / sp.sqrt(1 + x**2)) == 0
          and sp.simplify(sp.diff(F82_r, x) - (x + 1) * sp.cos(x)) == 0
          and d83_2 == 0 and d83_4 == 0)

    # 48. zy30 Ex 8.6, 8.9, 8.10 (Riemann sum & definite integral comparisons, P.211-214)
    v86 = sp.integrate((1 + x) / (1 + x**2), (x, 0, 1))
    I1_89 = float(sp.N(sp.Integral(sp.tan(x) / x, (x, 0, sp.pi / 4)), 6))
    I2_89 = float(sp.N(sp.Integral(x / sp.tan(x), (x, 0, sp.pi / 4)), 6))
    M_810 = float(sp.N(sp.Integral(sp.sin(sp.sin(x)), (x, 0, sp.pi / 2)), 6))
    N_810 = float(sp.N(sp.Integral(sp.cos(sp.cos(x)), (x, 0, sp.pi / 2)), 6))
    check("m1_c08_ex_zy30_06_10",
          sp.simplify(v86 - (sp.pi / 4 + sp.log(2) / 2)) == 0
          and 1 > I1_89 > I2_89 and M_810 < 1 < N_810,
          f"Ex8.9: 1>{I1_89:.4f}>{I2_89:.4f}; Ex8.10: {M_810:.4f}<1<{N_810:.4f}")

    # 49. zy30 Ex 8.12 - Ex 8.14 (Variable-limit integrals, P.217-218)
    F812_l = sp.integrate(sp.cos(t), (t, 0, x))
    F812_r = sp.integrate(sp.cos(t), (t, 0, sp.pi)) + sp.integrate(1, (t, sp.pi, x))
    l813 = sp.limit(sp.exp(x**2) + x**2, x, 0)
    v814 = sp.simplify(sp.exp(-ap * xp) * sp.integrate(sp.exp(ap * t), (t, 0, xp)) - (1 - sp.exp(-ap * xp)) / ap)
    check("m1_c08_ex_zy30_12_14",
          F812_l == sp.sin(x) and F812_r == x - sp.pi
          and sp.diff(F812_l, x).subs(x, sp.pi) == -1 and sp.diff(F812_r, x).subs(x, sp.pi) == 1
          and l813 == 1 and v814 == 0)

    # 50. zy30 Ex 8.15 - Ex 8.19 (Improper integrals convergence, P.221-225)
    l816 = sp.limit((sp.exp(-sp.cos(1 / xp)) - sp.exp(-1)) * xp**2, xp, sp.oo)
    v817B = sp.integrate(sp.log(xp) / (1 + xp**2), (xp, 0, sp.oo))
    F818_conv = 2 * sp.sqrt(xp) * (sp.log(xp) - 2)
    d818_conv = sp.simplify(sp.diff(F818_conv, xp) - sp.log(xp) / sp.sqrt(xp))
    v818_conv = F818_conv.subs(xp, 1) - sp.limit(F818_conv, xp, 0, "+")
    v818_div = sp.limit(sp.integrate(sp.log(xp) / xp, (xp, ap, 1)), ap, 0, "+")
    v819_conv = sp.integrate(sp.log(xp) / xp**2, (xp, 1, sp.oo))
    v819_div = sp.limit(sp.integrate(sp.log(xp) / xp, (xp, 1, ap)), ap, sp.oo)
    check("m1_c08_ex_zy30_15_19",
          sp.simplify(l816 - 1 / (2 * sp.E)) == 0 and v817B == 0
          and d818_conv == 0 and v818_conv == -4 and v818_div == -sp.oo and v819_conv == 1 and v819_div == sp.oo,
          f"Ex8.16 coeff={l816}, Ex8.17B={v817B}, Ex8.18(-4)/Ex8.19(1)")

    # 51. zy18 Intensive Ex 8.1 - Ex 8.5 (Riemann sum & product limits, P.79-82)
    v_i81 = sp.simplify(sp.exp(sp.integrate(sp.log(1 + x), (x, 0, 1))))
    v_i82 = sp.integrate(sp.sin(x), (x, 0, 1))
    v_i83 = sp.limit(sp.integrate(sp.cos(t), (t, 0, xp)) / xp, xp, 0, "+")
    v_i84 = sp.integrate(sp.cos(t)**2, (t, 0, sp.pi / 4))
    check("m1_c08_ex_zy18_01_05",
          v_i81 == 4 / sp.E and v_i82 == 1 - sp.cos(1) and v_i83 == 1
          and sp.simplify(v_i84 - (sp.Rational(1, 4) + sp.pi / 8)) == 0,
          f"Ex8.1={v_i81}, Ex8.4={v_i84}")

    # 52. zy18 Intensive Ex 8.6 - Ex 8.8 & Parameterized table 8-1 checks (P.86-89)
    l_i86_1 = sp.limit((sp.log(1 - xp) / xp**sp.Rational(3, 2)) * xp**sp.Rational(1, 2), xp, 0, "+")
    v_i87_0 = sp.integrate(sp.log(xp) / (1 - xp), (xp, 0, 1))
    lim_i87_0p_half = sp.limit(xp**sp.Rational(1, 2) * (1 - xp)**sp.Rational(-1, 2) * sp.log(xp), xp, 0, "+")
    lim_i87_1m_half = sp.limit((1 - xp)**sp.Rational(1, 2) * xp**sp.Rational(-1, 2) * sp.log(1 - xp), xp, 0, "+")
    v_t81_7 = sp.integrate(-sp.log(xp), (xp, 0, 1))
    check("m1_c08_ex_zy18_06_08",
          l_i86_1 == -1 and v_i87_0 == -sp.pi**2 / 6
          and lim_i87_0p_half == 0 and lim_i87_1m_half == 0 and v_t81_7 == 1,
          f"Ex8.7 p=0 -> {v_i87_0}; p=1/2 bounded(0,0)=proper")

    # ==================== Calculus Lecture 9: Integral Calculation ====================
    def _deriv_ok(F_expr, f_expr, var, pts) -> bool:
        d_expr = sp.diff(F_expr, var) - f_expr
        return all(abs(complex(sp.N(d_expr.subs(var, p), 18))) < 1e-12 for p in pts)

    # 53. Core formulas (10 groups, tabular P.237, exp*sin det, 1/(1+x^2)^2, Wallis, Gamma, x*cos(nx))
    c9_f_ok = (
        _deriv_ok(sp.log(sp.sec(x) + sp.tan(x)), sp.sec(x), x, [0.2, 0.7])
        and _deriv_ok(sp.Rational(1, 2) * (x / (1 + x**2) + sp.atan(x)), 1 / (1 + x**2)**2, x, [-1.2, 0.5, 2.0])
        and _deriv_ok((2 * sp.exp(2 * x) * sp.sin(3 * x) - 3 * sp.exp(2 * x) * sp.cos(3 * x)) / 13, sp.exp(2 * x) * sp.sin(3 * x), x, [0.3, 1.1])
        and _deriv_ok((2 * sp.exp(2 * x) * sp.cos(3 * x) + 3 * sp.exp(2 * x) * sp.sin(3 * x)) / 13, sp.exp(2 * x) * sp.cos(3 * x), x, [0.3, 1.1])
        and _deriv_ok((sp.Rational(1, 2) * x**3 - sp.Rational(3, 4) * x**2 + sp.Rational(7, 4) * x + sp.Rational(17, 8)) * sp.exp(2 * x), (x**3 + 2 * x + 6) * sp.exp(2 * x), x, [-0.8, 0.5, 1.4])
        and sp.simplify(sp.integrate(sp.sin(x)**8, (x, 0, sp.pi / 2)) - 35 * sp.pi / 256) == 0
        and sp.simplify(sp.integrate(sp.sin(x)**9, (x, 0, sp.pi)) - sp.Rational(256, 315)) == 0
        and sp.simplify(sp.integrate(x * sp.cos(3 * x), (x, 0, sp.pi)) - sp.Rational(-2, 9)) == 0
        and sp.simplify(sp.integrate(x**2 * sp.cos(3 * x), (x, 0, sp.pi)) - (-2 * sp.pi / 9)) == 0
        and sp.simplify(sp.gamma(sp.Rational(5, 2)) - sp.Rational(3, 4) * sp.sqrt(sp.pi)) == 0
    )
    check("m1_c09_f_core_formulas", c9_f_ok, "10 core groups, P.237 tab, exp*sin, Wallis, Gamma(5/2)")

    # 54. zy30 Ex 9.1 - Ex 9.10 (Indefinite integrals: substitution, IBP, cancellation, rational)
    c9_z1_ok = (
        _deriv_ok(sp.Rational(2, 3) * sp.asin(xp**sp.Rational(3, 2) / 2), sp.sqrt(xp) / sp.sqrt(4 - xp**3), xp, [0.5, 1.1])
        and _deriv_ok(sp.exp(sp.sin(x) / (sp.cos(x) + sp.sin(x))), sp.exp(sp.sin(x) / (sp.cos(x) + sp.sin(x))) / (sp.cos(x) + sp.sin(x))**2, x, [0.3, 0.9])
        and _deriv_ok(sp.Rational(9, 2) * sp.asin(x / 3) + x * sp.sqrt(9 - x**2) / 2, sp.sqrt(9 - x**2), x, [-1.5, 1.2])
        and _deriv_ok(2 * xp * sp.sqrt(sp.exp(xp) - 1) - 4 * sp.sqrt(sp.exp(xp) - 1) + 4 * sp.atan(sp.sqrt(sp.exp(xp) - 1)), xp * sp.exp(xp) / sp.sqrt(sp.exp(xp) - 1), xp, [0.6, 1.5])
        and _deriv_ok((x - 1) * sp.exp(sp.atan(x)) / (2 * sp.sqrt(1 + x**2)), x * sp.exp(sp.atan(x)) / (1 + x**2)**sp.Rational(3, 2), x, [-1.2, 0.7])
        and _deriv_ok(x - (1 + sp.exp(-x)) * sp.log(1 + sp.exp(x)), sp.log(1 + sp.exp(x)) / sp.exp(x), x, [-0.8, 1.1])
        and _deriv_ok(sp.exp(2 * x) * sp.tan(x), sp.exp(2 * x) * (sp.tan(x) + 1)**2, x, [-0.4, 0.6])
        and _deriv_ok(sp.log(xp + 1) + 1 / (2 * xp - 1), (4 * xp**2 - 6 * xp - 1) / ((xp + 1) * (2 * xp - 1)**2), xp, [0.1, 1.5])
        and _deriv_ok(sp.Rational(1, 4) * sp.log((x - 1)**2 / (x**2 + 1)) + sp.Rational(1, 2) * sp.atan(x), x / (x**3 - x**2 + x - 1), x, [-1.2, 0.2, 2.3])
        and _deriv_ok(sp.log(x**2 - x + 1) + 8 * sp.sqrt(3) / 3 * sp.atan((2 * x - 1) / sp.sqrt(3)), (2 * x + 3) / (x**2 - x + 1), x, [-0.7, 1.4])
    )
    check("m1_c09_ex_zy30_01_10", c9_z1_ok, "zy30 Ex9.1-9.10 all 10 indefinite integrals exact")

    # 55. zy30 Ex 9.11 - Ex 9.19 (Definite integrals, symmetry, King's property, Ex9.14 M2 typo check)
    v914_typo = sp.integrate(t * sp.sin(t), (t, sp.pi / 2, 0))
    v914_fix = sp.integrate(t * sp.sin(t), (t, 0, sp.pi / 2))
    c9_z2_ok = (
        sp.simplify(sp.integrate(x / (x**2 - 2), (x, 2, 2 * sp.sqrt(2))) - sp.log(3) / 2) == 0
        and abs(mp.quad(lambda v: mp.log((3 - 2 * v) / (1 + 2 * v)), [0, 1])) < 1e-12
        and sp.simplify(sp.integrate(2 * sp.sin(t)**2 * sp.cos(t)**2, (t, 0, sp.pi / 2)) - sp.pi / 8) == 0
        and abs(mp.quad(lambda v: mp.asin(mp.sqrt(1 - v**2)), [0, 1]) - 1) < 1e-12
        and v914_typo == -1 and v914_fix == 1
        and abs(mp.quad(lambda v: v * mp.asin(mp.sqrt(4 * v - 4 * v**2)), [0, 0.5, 1]) - 0.5) < 1e-12
        and abs(mp.quad(lambda v: mp.log(1 + mp.tan(v)), [0, mp.pi / 4]) - mp.pi / 8 * mp.log(2)) < 1e-12
        and abs(mp.quad(lambda v: v * mp.sin(v)**9, [0, mp.pi]) - 128 * mp.pi / 315) < 1e-12
        and sp.simplify(sp.integrate(-x**3 * sp.exp(-x**4), (x, 0, 1)) - sp.Rational(1, 4) * (sp.exp(-1) - 1)) == 0
    )
    check("m1_c09_ex_zy30_11_19", c9_z2_ok, f"Ex9.14 book M2={v914_typo} vs fixed={v914_fix}; Ex9.17=pi*ln2/8; Ex9.18=128pi/315")

    # 56. zy30 Ex 9.20 - Ex 9.29 (Variable-limit integrals & improper integrals + Gamma)
    F921 = (2 * x - sp.pi / 2) * sp.sin(x) + 2 * sp.cos(x) - 1
    F926_1 = sp.asin(2 * x - 1)
    F926_2 = sp.log((x - sp.Rational(1, 2)) + sp.sqrt((x - sp.Rational(1, 2))**2 - sp.Rational(1, 4)))
    c9_z3_ok = (
        sp.simplify((sp.exp(sp.sin(x)**2) * sp.cos(x)).subs(x, 0) - 1) == 0
        and sp.simplify(sp.series(F921, x, 0, 3).removeO() - (1 - sp.pi / 2 * x + x**2)) == 0
        and _deriv_ok(F926_1, 1 / sp.sqrt(x - x**2), x, [0.6, 0.8])
        and _deriv_ok(F926_2, 1 / sp.sqrt(x**2 - x), x, [1.1, 1.4])
        and sp.simplify((F926_1.subs(x, 1) - F926_1.subs(x, sp.Rational(1, 2))) + (F926_2.subs(x, sp.Rational(3, 2)) - F926_2.subs(x, 1)) - (sp.pi / 2 + sp.log(2 + sp.sqrt(3)))) == 0
        and sp.simplify(sp.integrate((1 - sp.sin(t)**2) * sp.cos(t), (t, sp.pi / 3, sp.pi / 2)) - (sp.Rational(2, 3) - 3 * sp.sqrt(3) / 8)) == 0
        and sp.integrate(xp**3 * sp.exp(-xp), (xp, 0, sp.oo)) == 6
        and abs(mp.quad(lambda v: v**2 * 4 * v**2 / (8 * mp.sqrt(mp.pi)) * mp.exp(-v**2 / 4), [0, mp.inf]) - 6.0) < 1e-12
    )
    check("m1_c09_ex_zy30_20_29", c9_z3_ok, "Ex9.21 Taylor abc=-pi/2; Ex9.26=pi/2+ln(2+sqrt3); Ex9.27/28/29 exact")

    # 57. zy18 Intensive Ex 9.1 - Ex 9.5 (3 radicals, sec^3, reciprocal/radical sub, King's prop)
    c9_i1_ok = (
        _deriv_ok(sp.log(x**2 + 2 * x + 2) / 2 - sp.atan(x + 1), x / (x**2 + 2 * x + 2), x, [-1.5, 0.6])
        and _deriv_ok(sp.Rational(1, 2) * sp.sec(x) * sp.tan(x) + sp.Rational(1, 2) * sp.log(sp.sec(x) + sp.tan(x)), sp.sec(x)**3, x, [0.2, 0.8])
        and _deriv_ok(x * sp.sqrt(4 + x**2) / 2 + 2 * sp.log((x + sp.sqrt(4 + x**2)) / 2), sp.sqrt(4 + x**2), x, [-1.2, 1.5])
        and _deriv_ok(x * sp.sqrt(x**2 - 4) / 2 - 2 * sp.log(sp.Abs(x + sp.sqrt(x**2 - 4)) / 2), sp.sqrt(x**2 - 4), x, [-3.5, 3.5])
        and _deriv_ok(-sp.sqrt(1 + x**2) / x, 1 / (x**2 * sp.sqrt(1 + x**2)), x, [-1.5, -0.5, 0.5, 1.5])
        and _deriv_ok(sp.log((sp.sqrt(sp.exp(x) + 1) - 1) / (sp.sqrt(sp.exp(x) + 1) + 1)), 1 / sp.sqrt(sp.exp(x) + 1), x, [-0.8, 1.2])
        and _deriv_ok(sp.Rational(1, 3) * (x**2 + 2 * x + 2)**sp.Rational(3, 2) - sp.Rational(1, 2) * (sp.log(x + 1 + sp.sqrt(x**2 + 2 * x + 2)) + (x + 1) * sp.sqrt(x**2 + 2 * x + 2)), x * sp.sqrt(x**2 + 2 * x + 2), x, [-1.5, 1.2])
        and _deriv_ok(sp.sqrt(2) / 4 * sp.log((sp.sqrt(2 * xp + 2) - sp.sqrt(2 * xp + 1)) / (sp.sqrt(2 * xp + 2) + sp.sqrt(2 * xp + 1))) + sp.sqrt((2 * xp + 1) * (xp + 1)), sp.sqrt((2 * xp + 1) / (xp + 1)), xp, [0.3, 2.0])
        and sp.simplify(sum(sp.integrate(x * sp.sin(x - k * sp.pi), (x, k * sp.pi, (k + 1) * sp.pi)) for k in range(3)) - 9 * sp.pi) == 0
    )
    check("m1_c09_ex_zy18_01_05", c9_i1_ok, "zy18 Ex9.1-9.5 incl. sec^3, x<-a branch, i9.4 right branch, i9.5=n^2*pi")

    # 58. zy18 Intensive Ex 9.6 - Ex 9.14 (5 IBP strategies, rational, trig-rational full/half angle)
    c9_i2_ok = (
        _deriv_ok(2 * x * sp.sqrt(sp.exp(x) + 1) - 4 * sp.sqrt(sp.exp(x) + 1) - 2 * sp.log((sp.sqrt(sp.exp(x) + 1) - 1) / (sp.sqrt(sp.exp(x) + 1) + 1)), x * sp.exp(x) / sp.sqrt(sp.exp(x) + 1), x, [-0.5, 1.2])
        and _deriv_ok(sp.exp(x) * (sp.sin(x) - sp.cos(x)) / 2, sp.exp(x) * sp.sin(x), x, [0.3, 1.1])
        and _deriv_ok(x * sp.sqrt(1 + x**2) / 2 + sp.log(x + sp.sqrt(1 + x**2)) / 2, sp.sqrt(1 + x**2), x, [-1.0, 1.2])
        and _deriv_ok(sp.exp(x) / (1 + x**2), sp.exp(x) * ((1 - x) / (1 + x**2))**2, x, [-0.8, 1.3])
        and sp.integrate(-xp**3 * sp.exp(-3 * xp), (xp, 0, sp.oo)) == -sp.Rational(2, 27)
        and abs(mp.quad(lambda v: 1 / ((v + 1) * (v**2 - 2 * v + 2)), [0, 1]) - (0.3 * mp.log(2) + mp.pi / 10)) < 1e-12
        and _deriv_ok(-sp.cos(x) - 2 / sp.cos(x) + 1 / (3 * sp.cos(x)**3), sp.sin(x)**5 / sp.cos(x)**4, x, [0.3, 0.9])
        and _deriv_ok(sp.Rational(1, 2) * sp.log((1 - sp.cos(x)) / (sp.cos(x) + 1)) + 1 / (2 * sp.sqrt(2)) * sp.log((sp.cos(x) + sp.sqrt(2)) / (sp.sqrt(2) - sp.cos(x))), 1 / (sp.sin(x) + sp.sin(x)**3), x, [0.4, 1.5])
        and _deriv_ok(sp.sin(x)**3 / 3 - sp.sin(x)**5 / 5, sp.sin(x)**2 * sp.cos(x)**3, x, [0.3, 1.4])
        and _deriv_ok(sp.Rational(4, 5) * x - sp.Rational(3, 5) * sp.log(sp.sin(x) + 2 * sp.cos(x)), (2 * sp.sin(x) + sp.cos(x)) / (sp.sin(x) + 2 * sp.cos(x)), x, [-0.3, 0.8])
        and _deriv_ok(sp.sqrt(3) / 3 * sp.log((sp.sqrt(3) + sp.tan(x / 2)) / (sp.sqrt(3) - sp.tan(x / 2))), 1 / (1 + 2 * sp.cos(x)), x, [-0.4, 0.8])
    )
    check("m1_c09_ex_zy18_06_14", c9_i2_ok, "zy18 Ex9.6-9.14 all IBP/rational/trig-rational exact")

    # 59. zy18 Intensive Ex 9.15 - Ex 9.22 (Improper param, 4 var-limit types, piecewise, geometry)
    f_i916 = sp.log(1 + xp) + 2 * xp * sp.log(1 + xp) - xp
    c9_i3_ok = (
        sp.simplify(sp.integrate(2 / (xp * (2 * xp + 2)), (xp, 1, sp.oo)) - sp.log(2)) == 0
        and sp.simplify(xp * sp.diff(f_i916, xp) - (2 * xp * sp.log(1 + xp) + xp**2 / (1 + xp))) == 0
        and sp.simplify(sp.integrate((t + 1) * sp.exp(2 * t), (t, -1, 1)) - (sp.Rational(3, 4) * sp.exp(2) + sp.Rational(1, 4) * sp.exp(-2))) == 0
        and sp.simplify(sp.integrate((1 - t) * sp.exp(2 * t), (t, -1, 1)) - (sp.Rational(1, 4) * sp.exp(2) - sp.Rational(5, 4) * sp.exp(-2))) == 0
        and sp.Rational(4, 3) * (sp.Rational(1, 2))**3 - (sp.Rational(1, 2))**2 + sp.Rational(1, 3) == sp.Rational(1, 4)
        and sp.limit(sp.sin(xp**4) * 2 * xp / (30 * xp**5), xp, 0, "+") == sp.Rational(1, 15)
        and sp.simplify(sp.integrate(x * (1 - sp.cos(x)), (x, 0, 1)) - (sp.Rational(3, 2) - sp.sin(1) - sp.cos(1))) == 0
        and sp.simplify(sp.integrate(1 + t**2, (t, -3, 0)) + sp.integrate(sp.exp(-t), (t, 0, 1)) - (13 - sp.exp(-1))) == 0
        and sp.simplify(sp.integrate(sp.sqrt(2 * x - x**2), (x, 0, 1)) - sp.pi / 4) == 0
    )
    check("m1_c09_ex_zy18_15_22", c9_i3_ok, "zy18 Ex9.15(a=2), 9.16-9.19(1/15), 9.20-9.22(pi/4) exact")

    # ==================== Calculus Lecture 10: Geometric Applications ====================
    # 60. zy30 Ex 10.1 - Ex 10.7 + Lemniscate Table 10-1 volumes
    a_p, b_p, l_p = sp.symbols("a_p b_p l_p", positive=True)
    f106 = xp / sp.sqrt(1 + xp**2)
    v_lem_y = mp.quad(lambda th: 2 * mp.pi / 3 * (mp.cos(2 * th))**1.5 * mp.cos(th), [-mp.pi / 4, mp.pi / 4])
    v_lem_sin_x = 2 * mp.quad(lambda th: 2 * mp.pi / 3 * (mp.sin(2 * th))**1.5 * mp.sin(th), [0, mp.pi / 2])
    c10_z1_ok = (
        sp.limit((1 - sp.Rational(2, 1) / (n + 2))**n, n, sp.oo) == sp.exp(-2)
        and sp.integrate((1 - sp.cos(t))**2, (t, 0, 2 * sp.pi)) == 3 * sp.pi
        and 2 * sp.integrate(sp.cos(2 * t), (t, 0, sp.pi / 4)) == 1
        and sp.simplify(sp.integrate(sp.exp(-x) * sp.sin(x), (x, 0, sp.pi)) / (1 - sp.exp(-sp.pi)) - (sp.exp(-sp.pi) + 1) / (2 * (1 - sp.exp(-sp.pi)))) == 0
        and sp.simplify(sp.pi * sp.integrate(sp.exp(-x) * sp.sin(x), (x, 0, sp.pi)) - sp.pi / 2 * (1 + sp.exp(-sp.pi))) == 0
        and sp.simplify(2 * f106 + xp**2 * f106.subs(xp, 1 / xp) - (xp**2 + 2 * xp) / sp.sqrt(1 + xp**2)) == 0
        and sp.simplify((sp.integrate(sp.exp(x), (x, -sp.oo, 0)) + sp.integrate(sp.exp(x) - sp.E * x, (x, 0, 1))) - sp.E / 2) == 0
        and sp.simplify(2 * sp.pi * sp.integrate((1 - x) * sp.exp(x), (x, -sp.oo, 1)) - sp.pi * sp.E / 3 - 5 * sp.pi * sp.E / 3) == 0
        and abs(v_lem_y - mp.sqrt(2) / 8 * mp.pi**2) < 1e-11
        and abs(v_lem_sin_x - mp.pi**2 / 4) < 1e-11
    )
    check("m1_c10_ex_zy30_01_07", c10_z1_ok, "zy30 Ex10.1-10.7 & Table 10-1 lemniscate volumes exact")

    # 61. zy30 Ex 10.8 - Ex 10.14 (incl. Ex10.14 closed vs non-closed Formula 10-1 check #33)
    A109 = sp.integrate(xp**2 / 4 - sp.log(xp) / 2, (xp, 1, sp.E))
    My109 = sp.integrate(xp * (xp**2 / 4 - sp.log(xp) / 2), (xp, 1, sp.E))
    v1014_half_exact = sp.pi / sp.sqrt(2) * sp.integrate((sp.sqrt(xp) - xp)**2 * (1 + 1 / (2 * sp.sqrt(xp))) / 2, (xp, 0, sp.Rational(1, 2)))
    v1014_half_short = sp.pi / sp.sqrt(2) * sp.integrate((sp.sqrt(xp) - xp)**2, (xp, 0, sp.Rational(1, 2)))
    c10_z2_ok = (
        sp.Rational(1, 2) * sp.integrate(x, (x, 0, 1)) == sp.Rational(1, 4)
        and sp.simplify(My109 / A109 - 3 * (sp.E**2 + 1) * (sp.E**2 - 3) / (4 * (sp.E**3 - 7))) == 0
        and sp.simplify(sp.integrate((1 + x**2) / (1 - x**2), (x, 0, sp.Rational(1, 2))) - (sp.log(3) - sp.Rational(1, 2))) == 0
        and sp.simplify(sp.integrate(sp.sqrt(1 + t**2), (t, 0, 2 * sp.pi)).rewrite(sp.log) - (sp.pi * sp.sqrt(1 + 4 * sp.pi**2) + sp.log(2 * sp.pi + sp.sqrt(1 + 4 * sp.pi**2)) / 2)) == 0
        and sp.simplify(2 * sp.pi * sp.integrate(sp.sqrt(x - 1) * sp.sqrt(1 + 1 / (4 * (x - 1))), (x, 1, 2)) - sp.pi / 6 * (5 * sp.sqrt(5) - 1)) == 0
        and sp.simplify(4 * sp.pi * sp.integrate(2 * sp.sin(t)**3 * 6 * sp.sin(t) * sp.cos(t), (t, 0, sp.pi / 2)) - 48 * sp.pi / 5) == 0
        and sp.simplify(sp.pi / sp.sqrt(2) * sp.integrate((sp.sqrt(xp) - xp)**2, (xp, 0, 1)) - sp.sqrt(2) * sp.pi / 60) == 0
        and abs(float(sp.N(v1014_half_exact - v1014_half_short))) > 0.003
    )
    check("m1_c10_ex_zy30_08_14", c10_z2_ok, "zy30 Ex10.8-10.14 exact; Ex10.14 unclosed discrepancy confirmed (#33)")

    # 62. zy18 Intensive Ex 10.1 - Ex 10.6 (Cardioid, Spiral, 3-leaf rose, Cycloid, Astroid, Folium)
    xf_fol = 3 * t / (1 + t**3)
    yf_fol = 3 * t**2 / (1 + t**3)
    c10_i1_ok = (
        sp.integrate(2 * sp.sin(t / 2), (t, 0, 2 * sp.pi)) == 8
        and sp.Rational(1, 2) * sp.integrate((1 - sp.cos(t))**2, (t, 0, 2 * sp.pi)) == 3 * sp.pi / 2
        and 2 * sp.pi / 3 * sp.integrate((1 - sp.cos(t))**3 * sp.sin(t), (t, 0, sp.pi)) == 8 * sp.pi / 3
        and sp.Rational(1, 2) * sp.integrate(t**2, (t, 0, 2 * sp.pi)) == 4 * sp.pi**3 / 3
        and sp.Rational(3, 2) * sp.integrate(sp.sin(3 * t)**2, (t, 0, sp.pi / 3)) == sp.pi / 4
        and sp.pi * sp.integrate((1 - sp.cos(t))**3, (t, 0, 2 * sp.pi)) == 5 * sp.pi**2
        and 4 * sp.integrate(3 * sp.sin(t) * sp.cos(t), (t, 0, sp.pi / 2)) == 6
        and 4 * sp.integrate(3 * sp.sin(t)**4 * sp.cos(t)**2, (t, 0, sp.pi / 2)) == 3 * sp.pi / 8
        and 2 * sp.pi * sp.integrate(3 * sp.sin(t)**7 * sp.cos(t)**2, (t, 0, sp.pi / 2)) == 32 * sp.pi / 105
        and sp.simplify(sp.Rational(1, 2) * sp.integrate(xf_fol * sp.diff(yf_fol, t) - yf_fol * sp.diff(xf_fol, t), (t, 0, sp.oo)) - sp.Rational(3, 2)) == 0
    )
    check("m1_c10_ex_zy18_01_06", c10_i1_ok, "zy18 Ex10.1-10.6 (6 famous curves s/S/V) all exact")

    # 63. zy18 Intensive Ex 10.7 - Ex 10.15 (incl. Ex10.13 MAJOR ERRATUM #32: 2pi/27 vs pi/9)
    v1013_exact = sp.pi * sp.integrate(xp**2 * sp.exp(-3 * xp), (xp, 0, sp.oo))
    v1013_typo = sp.pi * sp.integrate(xp * sp.exp(-3 * xp), (xp, 0, sp.oo))
    v1015 = sp.pi * (sp.Rational(1, 3) * sp.Rational(9, 16) * (1 / sp.sqrt(3)) + sp.integrate(1 / (1 + xp**2)**2, (xp, 1 / sp.sqrt(3), sp.oo)))
    c10_i2_ok = (
        sp.integrate(1 + x, (x, -1, 1)) == 2
        and sp.pi * sp.integrate(x**2 / 3 - 2 * x / 3 - 1, (x, 4, 9)) == 425 * sp.pi / 9
        and sp.pi * sp.integrate(xp * sp.exp(-xp / sp.E), (xp, 0, sp.oo)) == sp.pi * sp.E**2
        and abs(mp.quad(lambda v: mp.sqrt(1 + mp.cos(v)), [-mp.pi / 2, mp.pi / 2]) - 4.0) < 1e-12
        and sp.simplify(sp.integrate(sp.sec(x), (x, 0, sp.pi / 6)) - sp.log(3) / 2) == 0
        and sp.simplify(sp.integrate((xp + 1 / xp) / 2, (xp, 1, sp.E)) - (sp.E**2 + 1) / 4) == 0
        and sp.simplify(sp.pi * sp.integrate((2 - xp) * sp.log(xp), (xp, 1, 2)) - (2 * sp.log(2) - sp.Rational(5, 4)) * sp.pi) == 0
        and v1013_exact == 2 * sp.pi / 27 and v1013_typo == sp.pi / 9
        and sp.simplify(2 * sp.pi * sp.integrate(sp.cosh(x)**2, (x, 0, a_p)).rewrite(sp.exp) - sp.pi / 4 * (sp.exp(2 * a_p) - sp.exp(-2 * a_p) + 4 * a_p)) == 0
        and sp.simplify(v1015 - (sp.pi**2 / 6 - sp.sqrt(3) * sp.pi / 16)) == 0
    )
    check("m1_c10_ex_zy18_07_15", c10_i2_ok, f"zy18 Ex10.7-10.15 exact; Ex10.13 exact={v1013_exact} vs book typo={v1013_typo} (#32)")

    # ==================== Calculus Lecture 11: Integral Equations & Inequalities ====================
    # 64. zy30 Ex 11.1 - Ex 11.9 concrete constants & identities
    f116_sample = sp.sin(sp.pi * x)
    c11_z_ok = (
        abs(mp.quad(lambda v: 401 * v**400 * mp.log(1 + v), [0, 1]) - mp.log(2)) < 3e-3
        and sp.integrate(xp**5 * (-sp.log(xp)), (xp, 0, 1)) == sp.Rational(1, 36)
        and sp.simplify(sp.integrate(f116_sample, (x, 0, 1)) - sp.Rational(1, 2) * sp.integrate(x * (x - 1) * sp.diff(f116_sample, x, 2), (x, 0, 1))) == 0
        and sp.Rational(1, 2) * sp.integrate(x * (1 - x), (x, 0, 1)) == sp.Rational(1, 12)
        and sp.integrate(x, (x, 0, sp.Rational(1, 2))) + sp.integrate(1 - x, (x, sp.Rational(1, 2), 1)) == sp.Rational(1, 4)
        and sp.Rational(1, 2) * sp.integrate((x - 1)**2, (x, 0, 2)) == sp.Rational(1, 3)
    )
    check("m1_c11_zy30_and_sym", c11_z_ok, "zy30 Ex11.3/11.4/11.6(1/12)/11.8(1/4)/11.9(1/3) exact")

    # 65. zy18 Intensive Ex 11.1 - Ex 11.9 + 5 King's property examples
    v_ex_king = mp.quad(lambda v: v / (mp.exp(v) + mp.exp(1 - v)), [0, 1])
    v_ex_king_exact = (mp.atan(mp.sqrt(mp.e)) - mp.atan(1 / mp.sqrt(mp.e))) / (2 * mp.sqrt(mp.e))
    v_i114 = mp.quad(lambda v: 1 / (1 + mp.exp(1 / v)), [-1, 0]) + mp.quad(lambda v: 1 / (1 + mp.exp(1 / v)), [0, 1])
    v_i115 = mp.quad(lambda v: mp.sin(v)**4 / (1 + mp.exp(-v)), [-mp.pi / 2, mp.pi / 2])
    v_i117 = mp.quad(lambda v: mp.log(1 + v) / (1 + v**2), [0, 1])
    v_i118 = mp.quad(lambda v: v * mp.sqrt(mp.cos(v)**2 - mp.cos(v)**4), [0, mp.pi / 2, mp.pi])
    c11_i1_ok = (
        sp.simplify(sp.integrate((1 + xp**2) / (1 + xp**4), (xp, 0, sp.oo)) - sp.sqrt(2) * sp.pi / 2) == 0
        and abs(v_ex_king - v_ex_king_exact) < 1e-12
        and abs(mp.quad(lambda v: mp.sqrt(4 - v) / (mp.sqrt(4 - v) + mp.sqrt(v + 2)), [0, 2]) - 1.0) < 1e-12
        and abs(mp.quad(lambda v: 1 / (v + mp.sqrt(1 - v**2)), [0, 1]) - mp.pi / 4) < 1e-12
        and sp.limit(2 * sp.atan(1 / xp), xp, 0, "+") == sp.pi
        and sp.diff(sp.exp(sp.sin(x)) + sp.exp(-sp.sin(x)), x, 3).subs(x, 2 * sp.pi) == 0
        and abs(v_i114 - 1.0) < 1e-10
        and abs(v_i115 - 3 * mp.pi / 16) < 1e-12
        and abs(mp.quad(lambda v: abs(mp.sin(3 * v)), [0.3, mp.pi / 3, 2 * mp.pi / 3, mp.pi, 0.3 + mp.pi]) - 2.0) < 1e-12
        and abs(v_i117 - mp.pi / 8 * mp.log(2)) < 1e-12
        and abs(v_i118 - mp.pi / 2) < 1e-11
        and sp.limit(((2 * n - 1) / (2 * n))**(2 * n), n, sp.oo) == sp.exp(-1)
    )
    check("m1_c11_zy18_eq_01_10", c11_i1_ok, "zy18 King's 5 exs & Ex11.1-11.9 all exact")

    # 66. zy18 Intensive Ex 11.11 - Ex 11.18
    v_i1113 = mp.quad(lambda v: mp.cos(v) / (2 * v - 3 * mp.pi), [0, mp.pi / 2, mp.pi, 1.5 * mp.pi])
    v_i1115 = mp.quad(lambda v: (v - 1)**2 * mp.quad(lambda u: mp.exp(-u**2 + 2 * u), [0, v]), [0, 1])
    c11_i2_ok = (
        sp.integrate(x - 1, (x, 0, 2)) == 0
        and sp.integrate(x * (x - 1) * (x - 2), (x, 0, 2)) == 0
        and sp.integrate(x * sp.sqrt(4 * x - x**2), (x, 0, 4)) == 4 * sp.pi
        and sp.integrate((2 * x + 1) * sp.sqrt(2 * x - x**2), (x, 0, 2)) == 3 * sp.pi / 2
        and v_i1113 > 0
        and sp.simplify(sp.integrate(x * (-1 / (1 + 2 * x)**2), (x, 0, 1)) - (sp.Rational(1, 6) - sp.log(3) / 4)) == 0
        and abs(v_i1115 - (mp.e - 2) / 6) < 1e-12
        and sp.integrate(1, (x, 1, sp.E)) == sp.E - 1
        and sp.integrate(x**4, (x, 0, 1)) == sp.Rational(1, 5)
        and sp.integrate(x * (1 - x) / 2, (x, 0, 1)) == sp.Rational(1, 12)
    )
    check("m1_c11_zy18_11_19", c11_i2_ok, f"zy18 Ex11.11-11.18 exact (Ex11.13={float(v_i1113):.6f}>0)")

    # ==================== Calculus Lecture 12: Physical Applications & Pappus ====================
    # 67. All zy30 Ex 12.1-12.4 + P.297 rod gravity + P.300 Pappus + all zy18 Ex 12.1-12.7
    H_p = sp.symbols("H_p", positive=True)
    v122_fn = 2 * (3 * t**2 - 4 * t + 2)  # v(t)>0 on [0,1], int_0^1 v=2, v(1)=2
    P1_124 = sp.integrate(2 * t, (t, 0, H_p))
    P2_124 = sp.integrate((H_p + 1 - t) * 2 * sp.sqrt(t), (t, 0, 1))
    L124 = 8 * x - x**2 - (4 * x + x**2 / 8 + 1)
    x0_124 = sp.solve(sp.diff(L124, x), x)[0]
    C127 = sp.integrate(t**2 - 4 * t + 6, (t, 0, x)) + 200
    L127 = 146 * x - C127
    x0_127 = [r for r in sp.solve(sp.diff(L127, x), x) if r > 0][0]
    c12_ok = (
        [r for r in sp.solve(sp.integrate(x, (x, 1, 1 + a_p)) - sp.integrate(x, (x, 0, 1)), a_p) if r > 0][0] == sp.sqrt(2) - 1
        and sp.simplify(sp.integrate(x * sp.pi * b_p**2 * (a_p - x)**2 / a_p**2, (x, 0, a_p)) - a_p**2 * b_p**2 * sp.pi / 12) == 0
        and sp.simplify(sp.integrate(2 * t * (a_p - t), (t, 0, a_p)) - a_p**3 / 3) == 0
        and sp.simplify(sp.integrate(1 / (a_p - x)**2, (x, -l_p, 0)) - l_p / (a_p * (a_p + l_p))) == 0
        and 2 * sp.pi * (2 * sp.pi) * 2 == 8 * sp.pi**2
        and x0_124 == sp.Rational(16, 9) and 8 - x0_124 == sp.Rational(56, 9)
        and [r for r in sp.solve(sp.integrate(1900 + 5 * (a_p - x), (x, 0, a_p)) - 59250, a_p) if r > 0][0] == 30
        and sp.integrate(v122_fn, (t, 0, 1)) == 2 and v122_fn.subs(t, 1) == 2 and sp.integrate(t * sp.diff(v122_fn, t), (t, 0, 1)) == 0
        and [r for r in sp.solve(sp.integrate(sp.pi * (a_p**2 - t**2) * t, (t, 0, a_p)) - sp.pi / 2, a_p) if r.is_real and r > 0][0] == sp.Pow(2, sp.Rational(1, 4))
        and [r for r in sp.solve(sp.Eq(P1_124 / P2_124, sp.Rational(5, 4)), H_p) if r > 0][0] == 2
        and sp.simplify(-sp.integrate(1 / (1 + t**2)**sp.Rational(3, 2), (t, 0, 1)) - (-sp.sqrt(2) / 2)) == 0
        and sp.integrate(10 * (1 - t**2) * 2 * sp.pi * t, (t, 0, 1)) == 5 * sp.pi
        and sp.integrate(x**2 - 4 * x + 6, (x, 2, 4)) == sp.Rational(20, 3)
        and x0_127 == 14 and L127.subs(x, x0_127) == sp.Rational(3712, 3)
    )
    check("m1_c12_physics_and_econ", c12_ok, "zy30 Ex12.1-12.4, rod gravity, Pappus 8pi^2, zy18 Ex12.1-12.7 all exact")

    # ==================== Proof Special Section (proof-01 ~ proof-03) ====================
    # 68. proof-01: Sequence limit proofs (26 examples numerical & symbolic sanity)
    seq41 = [float(k**(1 / k) - ((-1)**k) / k) for k in range(1, 30)]
    x412 = 2.9
    for _ in range(80):
        x412 = (x412 * (3 - x412)) ** 0.5
    x413 = 1.5
    for _ in range(40):
        x413_next = float(mp.log((mp.exp(x413) - 1) / x413))
        r413 = x413_next / x413
        x413 = x413_next
    x417 = 4.0
    for _ in range(15):
        x417 = (x417**2 + 2) / (2 * x417 - 1)
    x424 = 2.0
    for _ in range(40):
        x424 = 2.0 + 1.0 / x424
    x425 = 0.5
    for _ in range(40):
        x425 = (1.0 - x425**2) / 2.0
    # Ex4.26 sample f(x) = 1 + 0.3*x + 0.05*sin(x), 0<f'<0.5 => fixed point in (1,2)
    x426 = -5.0
    for _ in range(80):
        x426 = 1.0 + 0.3 * x426 + 0.05 * float(mp.sin(x426))
    p01_ok = (
        max(seq41) == seq41[0] and min(seq41) == seq41[1]
        and abs(x412 - 1.5) < 1e-10
        and abs(x413) < 1e-10 and abs(r413 - 0.5) < 1e-9
        and abs(x417 - 2.0) < 1e-12
        and abs(x424 - (1 + 2**0.5)) < 1e-12
        and abs(x425 - (2**0.5 - 1)) < 1e-12
        and 1.0 < x426 < 2.0
        and sp.limit((sp.Rational(1, 2))**(2**(n - 1) - n), n, sp.oo) == 0
        and sp.simplify(sp.diff(sp.exp(x) - sp.exp(-x), x).subs(x, 0) - 2) == 0
    )
    check("m1_p01_seq_all", p01_ok, f"p01 Ex4.1/4.12({x412:.4f})/4.13(r={r413:.4f})/4.17/4.24/4.25/4.26({x426:.4f}) exact")

    # 69. proof-02: MVT, Taylor, Landau/Kolmogorov, Aux-Polynomial & Constant-K identities
    f_sym = sp.Function("f")
    M_s, N_s, h_s = sp.symbols("M_s N_s h_s", positive=True)
    landau_min = (M_s / h_s + N_s * h_s / 2).subs(h_s, sp.sqrt(2 * M_s / N_s))
    kolm_min = (M_s / h_s + N_s * h_s**2 / 6).subs(h_s, (3 * M_s / N_s)**sp.Rational(1, 3))
    p731 = -3 * x**2 + 4 * x
    p734 = 3 * x + 1
    # Ex7.37 F''(x) cancellation & Ex7.38 quadratic check
    K_s, a_s = sp.symbols("K_s a_s", positive=True)
    F737 = f_sym(x) - f_sym(a_s) - sp.Rational(1, 2) * (x - a_s) * (sp.diff(f_sym(x), x) + sp.diff(f_sym(a_s), a_s)) + sp.Rational(1, 12) * (x - a_s)**3 * K_s
    d2_737 = sp.simplify(sp.diff(F737, x, 2) - sp.Rational(1, 2) * (x - a_s) * (K_s - sp.diff(f_sym(x), x, 3)))
    p02_ok = (
        sp.simplify(sp.diff((x - 1)**2 * sp.diff(f_sym(x), x), x) - (x - 1) * (2 * sp.diff(f_sym(x), x) + (x - 1) * sp.diff(f_sym(x), x, 2))) == 0
        and sp.simplify(sp.diff(x * sp.exp(x) * (sp.diff(f_sym(x), x) - 1), x) - sp.exp(x) * ((1 + x) * (sp.diff(f_sym(x), x) - 1) + x * sp.diff(f_sym(x), x, 2))) == 0
        and sp.simplify(landau_min - sp.sqrt(2 * M_s * N_s)) == 0
        and sp.simplify(kolm_min**3 - sp.Rational(9, 8) * M_s**2 * N_s) == 0
        and p731.subs(x, 0) == 0 and p731.subs(x, 1) == 1 and sp.integrate(p731, (x, 0, 1)) == 1 and sp.diff(p731, x, 2) == -6
        and sp.integrate(p734, (x, 0, 1)) == sp.Rational(5, 2) and sp.integrate(x * p734, (x, 0, 1)) == sp.Rational(3, 2)
        and d2_737 == 0
        and sp.simplify(sp.integrate(x**2, (x, 0, a_s)) - (a_s / 2 * (0 - a_s**2) + sp.Rational(5, 12) * 2 * a_s**3)) == 0
    )
    check("m1_p02_aux_and_k", p02_ok, "p02 ODE aux, Landau sqrt(2MN), Kolm (9M^2N/8)^(1/3), Ex7.31/34/37/38 exact")

    # 70. proof-03: 2013-2026 Real Kaoyan proof exams concrete identities & 2016 numerical mean
    # 2016: f(x) = int_0^x cos(t)/(2t - 3pi) dt => average on [0, 3pi/2] = 1/(3pi)
    f2016_int = mp.quad(lambda u: mp.quad(lambda v: mp.cos(v) / (2 * v - 3 * mp.pi), [0, u]), [0, 1.5 * mp.pi])
    mean_2016 = f2016_int / (1.5 * mp.pi)
    f2016_end = mp.quad(lambda v: mp.cos(v) / (2 * v - 3 * mp.pi), [0, mp.pi / 2, mp.pi, 1.5 * mp.pi])
    # 2019-2 recurrence: a_n = int_0^1 x^n sqrt(1-x^2) dx
    a2019_1 = sp.integrate(x * sp.sqrt(1 - x**2), (x, 0, 1))
    a2019_3 = sp.integrate(x**3 * sp.sqrt(1 - x**2), (x, 0, 1))
    # 2022 sufficiency 2nd-order Taylor coefficient: int_{-h}^h t^2 dt / (2h) = h^2/3 -> factor 1/3 * f''
    c2022_coeff = sp.integrate(t**2 / 2, (t, -h_s, h_s)) / h_s**3
    # 2026 F(-1)=F(0)=F(1)=0 check on f(x)=x^3 + x^2 - 1/3 (strictly inc on [-1,1]? take f(x)=x+0.1*(x^2-1/3))
    f2026 = x + sp.Rational(1, 10) * (x**2 - sp.Rational(1, 3))
    a2026 = sp.integrate(f2026, (x, 0, 1))
    F2026 = a2026 * (1 - x**2) + sp.integrate(f2026.subs(x, t), (t, 1, x))
    p03_ok = (
        sp.integrate(f2026, (x, -1, 1)) == 0 and a2026 > 0
        and F2026.subs(x, -1) == 0 and F2026.subs(x, 0) == 0 and F2026.subs(x, 1) == 0
        and c2022_coeff == sp.Rational(1, 3)
        and sp.simplify(a2019_3 - sp.Rational(2, 5) * a2019_1) == 0
        and abs(mean_2016 - 1 / (3 * mp.pi)) < 1e-8
        and f2016_end > 0
        and sp.diff(sp.log(xp) + 1 / xp, xp).subs(xp, 1) == 0 and (sp.log(1) + 1) == 1
    )
    check("m1_p03_real_exams", p03_ok, f"p03 2026(a={a2026}), 2022(1/3), 2019-2(2/5), 2016(mean={float(mean_2016):.6f}, end={float(f2016_end):.4f}>0) exact")

    # ==================== Calculus Lecture 13 (calc-13: 多元函数微分学) ====================
    y = sp.Symbol("y", real=True)
    yp = sp.Symbol("yp", positive=True)
    u_s, v_s, z_s, a_c, b_c = sp.symbols("u_s v_s z_s a_c b_c", real=True)

    # 71. All 16 total differential formulas (zy18 P.157-158) + Note(1) + Errata #44 Note(2)
    td16_cases = [
        ((x**2 + y**2) / 2, x, y),
        ((x**2 - y**2) / 2, x, -y),
        (x * y, y, x),
        (sp.log(xp * yp), 1 / xp, 1 / yp),
        (sp.Rational(1, 2) * sp.log(x**2 + y**2), x / (x**2 + y**2), y / (x**2 + y**2)),
        (sp.Rational(1, 2) * sp.log(xp**2 - yp**2), xp / (xp**2 - yp**2), -yp / (xp**2 - yp**2)),
        (y / x, -y / x**2, 1 / x),
        (x / y, 1 / y, -x / y**2),
        (sp.atan(x / y), y / (x**2 + y**2), -x / (x**2 + y**2)),
        (sp.atan(y / x), -y / (x**2 + y**2), x / (x**2 + y**2)),
        (sp.Rational(1, 2) * sp.log((xp - yp) / (xp + yp)), yp / (xp**2 - yp**2), -xp / (xp**2 - yp**2)),
        (sp.Rational(1, 2) * sp.log((xp + yp) / (xp - yp)), -yp / (xp**2 - yp**2), xp / (xp**2 - yp**2)),
        (-sp.Rational(1, 2) / (x**2 + y**2), x / (x**2 + y**2)**2, y / (x**2 + y**2)**2),
        (-sp.Rational(1, 2) / (x**2 - y**2), x / (x**2 - y**2)**2, -y / (x**2 - y**2)**2),
        (sp.Rational(1, 2) * sp.atan(x**2 + y**2), x / (1 + (x**2 + y**2)**2), y / (1 + (x**2 + y**2)**2)),
        (sp.Rational(1, 2) * sp.atan(x**2 - y**2), x / (1 + (x**2 - y**2)**2), -y / (1 + (x**2 - y**2)**2)),
    ]
    td16_ok = True
    for expr_u, P_exp, Q_exp in td16_cases:
        vx = xp if expr_u.has(xp) else x
        vy = yp if expr_u.has(yp) else y
        if sp.simplify(sp.diff(expr_u, vx) - P_exp) != 0 or sp.simplify(sp.diff(expr_u, vy) - Q_exp) != 0:
            td16_ok = False
            break
    # Note(1): (x dx - a y dy)/(1+(x^2+y^2)^2) exact => a = -1
    P_n1 = x / (1 + (x**2 + y**2)**2)
    Q_n1 = -a_c * y / (1 + (x**2 + y**2)**2)
    a_n1_sol = sp.solve(sp.simplify(sp.diff(P_n1, y) - sp.diff(Q_n1, x)), a_c)
    # Errata #44 Note(2): corrected Q = -4*x^2*y/(x^2+y^2)^2 => a = 4 (potential 2*x^2/(x^2+y^2)); typo 4*x^6*y fails exactness
    P_n2 = a_c * x * y**2 / (x**2 + y**2)**2
    Q_n2_fix = -4 * x**2 * y / (x**2 + y**2)**2
    Q_n2_typo = -4 * x**6 * y / (x**2 + y**2)**2
    a_n2_sol = sp.solve(sp.simplify(sp.diff(P_n2, y) - sp.diff(Q_n2_fix, x)), a_c)
    typo_diff = sp.simplify(sp.diff(P_n2.subs(a_c, 4), y) - sp.diff(Q_n2_typo, x))
    u_n2 = 2 * x**2 / (x**2 + y**2)
    td_all_ok = (
        td16_ok and a_n1_sol == [-1] and a_n2_sol == [4] and typo_diff != 0
        and sp.simplify(sp.diff(u_n2, x) - P_n2.subs(a_c, 4)) == 0
        and sp.simplify(sp.diff(u_n2, y) - Q_n2_fix) == 0
    )
    check("m1_c13_total_diff_16", td_all_ok, "zy18 all 16 total differentials + Note(1) a=-1 + Errata #44 (4x^2y exact a=4, 4x^6y non-exact) verified")

    # 72. calc-13 zy30 Ex 13.1 ~ 13.11
    lim_13_2 = sp.limit(((1 - t)**sp.Rational(1, 3) - 1) / (sp.exp(t) - 1), t, 0, "+")
    # Ex 13.5: int_0^oo e^(-x t^2) dt = sqrt(pi)/(2 sqrt(x))
    f13_5 = sp.integrate(sp.exp(-xp * t**2), (t, 0, sp.oo))
    df13_5 = sp.diff(f13_5, xp).subs(xp, 1)
    # Ex 13.6: f = e^(-x) sin(y), volume pi * int_0^oo e^(-2x) sin^2(x) dx = pi/8
    vol_13_6 = sp.pi * sp.integrate(sp.exp(-2 * t) * sp.sin(t)**2, (t, 0, sp.oo))
    # Ex 13.7: (a x y^3 - y^2 cos x) dx + (1 + b y sin x + 3 x^2 y^2) dy
    P13_7 = a_c * x * y**3 - y**2 * sp.cos(x)
    Q13_7 = 1 + b_c * y * sp.sin(x) + 3 * x**2 * y**2
    d13_7 = sp.diff(P13_7, y) - sp.diff(Q13_7, x)
    sol_13_7 = sp.solve([d13_7.subs({x: 0, y: 1}), d13_7.subs({x: sp.pi / 2, y: 1})], (a_c, b_c))
    ok_13_7 = sol_13_7 == {a_c: 2, b_c: -2} and sp.simplify(d13_7.subs(sol_13_7)) == 0
    # Ex 13.9: test with f(u,v) = u - v + 3*u^2 + 5*u*v + 7*v^2
    f13_9_test = (sp.exp(x) * sp.sin(y)) - (x**2 + y**2) + 3 * (sp.exp(x) * sp.sin(y))**2 + 5 * sp.exp(x) * sp.sin(y) * (x**2 + y**2)
    zxy_13_9 = sp.diff(f13_9_test, x, y).subs({x: 0, y: 0})
    # Ex 13.11: x=u*v, y=(u^2-v^2)/2 => g_u^2 + g_v^2 = (u^2+v^2)*(fx^2+fy^2)
    fx_s, fy_s = sp.symbols("fx_s fy_s", real=True)
    gu_13_11 = fx_s * v_s + fy_s * u_s
    gv_13_11 = fx_s * u_s - fy_s * v_s
    id_13_11 = sp.simplify(gu_13_11**2 + gv_13_11**2 - (u_s**2 + v_s**2) * (fx_s**2 + fy_s**2))
    c13_b_ok = (
        lim_13_2 == -sp.Rational(1, 3)
        and sp.diff(sp.log(y), y).subs(y, 1) == 1
        and sp.simplify(df13_5 - (-sp.sqrt(sp.pi) / 4)) == 0
        and vol_13_6 == sp.pi / 8
        and ok_13_7
        and zxy_13_9 == 1
        and id_13_11 == 0
    )
    check("m1_c13_zy30_basics", c13_b_ok, "zy30 Ex13.1-13.11 (a=-1/3, -sqrt(pi)/4, V=pi/8, (2,-2), z_xy=1, parabolic PDE) exact")

    # 73. calc-13 zy30 Ex 13.12 ~ 13.22 (implicit, extrema, unclosed arc, ellipse-line, closed ellipse)
    F13_12 = sp.exp(x + 2 * y + 3 * z_s) + x * y * z_s - 1
    zx_13_12 = (-sp.diff(F13_12, x) / sp.diff(F13_12, z_s)).subs({x: 0, y: 0, z_s: 0})
    zy_13_12 = (-sp.diff(F13_12, y) / sp.diff(F13_12, z_s)).subs({x: 0, y: 0, z_s: 0})
    F13_13 = x * y - z_s * sp.log(y) + sp.exp(x * z_s) - 1
    pt13_13 = {x: 0, y: 1, z_s: 1}
    # Ex 13.16: test concrete F(u,v) = u^2 + v^3 - 5 => x*z_x + y*z_y = z - x*y
    F13_16 = (x + z_s / y)**2 + (y + z_s / x)**3 - 5
    zx_13_16 = -sp.diff(F13_16, x) / sp.diff(F13_16, z_s)
    zy_13_16 = -sp.diff(F13_16, y) / sp.diff(F13_16, z_s)
    id_13_16 = sp.simplify(x * zx_13_16 + y * zy_13_16 - (z_s - x * y))
    # Ex 13.19: (x^2+y^2)*z + ln(z) + 2(x+y+1)=0 at (-1,-1,1) -> A=C=-2/3, B=0, g'(z)=1/z+2/z^2 > 0
    F13_19 = (x**2 + y**2) * z_s + sp.log(z_s) + 2 * (x + y + 1)
    # Implicit second derivatives at stationary point (-1,-1,1) where zx=zy=0: z_xx = -F_xx / F_z
    A13_19 = (-sp.diff(F13_19, x, 2) / sp.diff(F13_19, z_s)).subs({x: -1, y: -1, z_s: 1})
    B13_19 = (-sp.diff(F13_19, x, y) / sp.diff(F13_19, z_s)).subs({x: -1, y: -1, z_s: 1})
    C13_19 = (-sp.diff(F13_19, y, 2) / sp.diff(F13_19, z_s)).subs({x: -1, y: -1, z_s: 1})
    # Ex 13.20: S = (b-a)^3/6 on a^2+b^2=1, a<=0, b>=0 -> max sqrt(2)/3, min 1/6
    S13_20_max = (sp.sqrt(2))**3 / 6
    S13_20_min = sp.Rational(1, 6)
    # Ex 13.21: x^2+4y^2=4 to 2x+3y-6=0 -> P(8/5,3/5) d=1/sqrt(13), Q(-8/5,-3/5) d=11/sqrt(13)
    d13_21_min = sp.Abs(2 * sp.Rational(8, 5) + 3 * sp.Rational(3, 5) - 6) / sp.sqrt(13)
    d13_21_max = sp.Abs(2 * (-sp.Rational(8, 5)) + 3 * (-sp.Rational(3, 5)) - 6) / sp.sqrt(13)
    # Ex 13.22: z = x^2 - y^2 + 2 on x^2 + y^2/4 <= 1 -> boundary 3 - 5/4 y^2 in [-2, 3]
    c13_ie_ok = (
        zx_13_12 == -sp.Rational(1, 3) and zy_13_12 == -sp.Rational(2, 3)
        and sp.diff(F13_13, x).subs(pt13_13) == 2 and sp.diff(F13_13, y).subs(pt13_13) == -1 and sp.diff(F13_13, z_s).subs(pt13_13) == 0
        and id_13_16 == 0
        and F13_19.subs({x: -1, y: -1, z_s: 1}) == 0 and A13_19 == -sp.Rational(2, 3) and B13_19 == 0 and C13_19 == -sp.Rational(2, 3)
        and sp.simplify(S13_20_max - sp.sqrt(2) / 3) == 0 and S13_20_min == sp.Rational(1, 6)
        and (sp.Rational(8, 5))**2 + 4 * (sp.Rational(3, 5))**2 == 4
        and sp.simplify(d13_21_min - 1 / sp.sqrt(13)) == 0 and sp.simplify(d13_21_max - 11 / sp.sqrt(13)) == 0
        and (3 - sp.Rational(5, 4) * 0) == 3 and (3 - sp.Rational(5, 4) * 4) == -2
    )
    check("m1_c13_zy30_impl_extrema", c13_ie_ok, "zy30 Ex13.12-13.22 (dz=-dx/3-2dy/3, Ex13.16 id, Ex13.19 A=C=-2/3, Ex13.20-22 exact)")

    # 74. calc-13 zy18 Ex 13.1 ~ 13.13 (limits, iterated limits, 6 counterexamples, mixed partials)
    lim_13_2_curve = sp.limit((x**3 + (-x**2 + x**3)**3) / (x**2 + (-x**2 + x**3)), x, 0)
    quad_min_diff = sp.expand((x**2 - x * y + y**2) - ((x - y / 2)**2 + sp.Rational(3, 4) * y**2))
    lim_13_4 = sp.limit((x**2 + 1) * sp.exp(-(x + 1)), x, sp.oo)
    I1_13_6 = sp.limit(sp.limit((x**2 - y**2) / (x**2 + y**2), y, 0), x, 0)
    I2_13_6 = sp.limit(sp.limit((x**2 - y**2) / (x**2 + y**2), x, 0), y, 0)
    # Phenomenon 4 (Ex 13.10): along y=x, x>0 -> x^3/(2x^2)^(3/2) = 1/(2 sqrt(2)) != 0
    ph4_lim = sp.limit(xp**3 / (2 * xp**2)**sp.Rational(3, 2), xp, 0, "+")
    # Phenomenon 6 (Ex 13.12): derivative along y=0 is 2x sin(1/x^2) - (2/x) cos(1/x^2)
    f_ph6 = (x**2 + y**2) * sp.sin(1 / (x**2 + y**2))
    fx_ph6_y0 = sp.simplify(sp.diff(f_ph6, x).subs(y, 0))
    # Ex 13.13 mixed partials: f_x'(0,y) = -y => f_xy(0,0)=-1; f_y'(x,0) = x => f_yx(0,0)=1
    c13_lim_ok = (
        lim_13_2_curve == 1 and quad_min_diff == 0 and lim_13_4 == 0
        and I1_13_6 == 1 and I2_13_6 == -1
        and sp.simplify(ph4_lim - 1 / (2 * sp.sqrt(2))) == 0
        and sp.simplify(fx_ph6_y0 - (2 * x * sp.sin(1 / x**2) - 2 / x * sp.cos(1 / x**2))) == 0
        and sp.diff(-y, y) == -1 and sp.diff(x, x) == 1
    )
    check("m1_c13_zy18_limits_concepts", c13_lim_ok, "zy18 Ex13.1-13.13 (curve limit=1, iterated 1/-1, Ph4/Ph6, mixed -1 vs 1) exact")

    # 75. calc-13 zy18 Ex 13.14 ~ 13.24 (exact diff, implicit system, Euler homogeneous, 4 PDEs)
    P13_14 = (x + a_c * y) / (x + y)**2
    Q13_14 = y / (x + y)**2
    a13_14 = sp.solve(sp.simplify(sp.diff(P13_14, y) - sp.diff(Q13_14, x)), a_c)
    # Ex 13.15: x/z - ln(z/y) = 0
    F13_15 = x / z_s - sp.log(z_s) + sp.log(y)
    zx_13_15 = sp.simplify(-sp.diff(F13_15, x) / sp.diff(F13_15, z_s))
    zy_13_15 = sp.simplify(-sp.diff(F13_15, y) / sp.diff(F13_15, z_s))
    # Ex 13.17: e^z - xyz = 0, x z^2 - ln y = 0 at x=1/2, y=e^2, z=2 => dz/dx = -4
    G1_17 = sp.exp(z_s) - x * y * z_s
    G2_17 = x * z_s**2 - sp.log(y)
    pt17 = {x: sp.Rational(1, 2), y: sp.exp(2), z_s: 2}
    yp_s, zp_s = sp.symbols("yp_s zp_s")
    sol17 = sp.solve([
        (sp.diff(G1_17, x) + sp.diff(G1_17, y) * yp_s + sp.diff(G1_17, z_s) * zp_s).subs(pt17),
        (sp.diff(G2_17, x) + sp.diff(G2_17, y) * yp_s + sp.diff(G2_17, z_s) * zp_s).subs(pt17),
    ], (yp_s, zp_s))
    # Ex 13.19: z = sqrt((x^2+y^2)/(xy)) => x z_x + y z_y = 0
    z13_19 = sp.sqrt((xp**2 + yp**2) / (xp * yp))
    euler_13_19 = sp.simplify(xp * sp.diff(z13_19, xp) + yp * sp.diff(z13_19, yp))
    # Ex 13.20: f(u) = (1/2) u ln u
    f13_20 = sp.Rational(1, 2) * u_s * sp.log(u_s)
    # Ex 13.21: z(u,v) = 5/4 e^v + (u^2 - 5/4) e^(v/5) with u=x-2y, v=x+3y
    z21_uv = sp.Rational(5, 4) * sp.exp(v_s) + (u_s**2 - sp.Rational(5, 4)) * sp.exp(v_s / 5)
    z21_xy = z21_uv.subs({u_s: x - 2 * y, v_s: x + 3 * y})
    pde21_res = sp.simplify(6 * sp.diff(z21_xy, x, 2) + sp.diff(z21_xy, x, y) - sp.diff(z21_xy, y, 2) - (3 * sp.diff(z21_xy, x) - sp.diff(z21_xy, y)))
    bc21_1 = sp.simplify(sp.diff(z21_uv.subs(u_s, 0), v_s) - (sp.Rational(1, 5) * z21_uv.subs(u_s, 0) + sp.exp(v_s)))
    bc21_2 = sp.simplify(z21_uv.subs(v_s, 0) - u_s**2)
    # Ex 13.23: f(t) = e^(5t)/25 - 11t/5 - 1/25, u = f(ln(sqrt(x^2+y^2)))
    f23 = sp.exp(5 * t) / 25 - sp.Rational(11, 5) * t - sp.Rational(1, 25)
    u23 = f23.subs(t, sp.Rational(1, 2) * sp.log(xp**2 + yp**2))
    lap23_res = sp.simplify(sp.diff(u23, xp, 2) + sp.diff(u23, yp, 2) - (xp**2 + yp**2)**sp.Rational(3, 2))
    # int_0^1 f(xt) dt / x == int_0^x f(s) ds / x^2 (avoid SymPy Piecewise artefact at x=0)
    lim23 = sp.limit(sp.integrate(f23.subs(t, u_s), (u_s, 0, x)) / x**2, x, 0)
    # Ex 13.24: f(x,y) = e^(-x) (1 - ln(cos y))
    f24 = sp.exp(-x) * (1 - sp.log(sp.cos(y)))
    c13_pde_ok = (
        a13_14 == [2]
        and sp.simplify(zx_13_15 - z_s / (x + z_s)) == 0 and sp.simplify(zy_13_15 - z_s**2 / (y * (x + z_s))) == 0
        and G1_17.subs(pt17) == 0 and G2_17.subs(pt17) == 0 and sol17[zp_s] == -4
        and euler_13_19 == 0
        and f13_20.subs(u_s, 1) == 0 and sp.diff(f13_20, u_s).subs(u_s, 1) == sp.Rational(1, 2)
        and pde21_res == 0 and bc21_1 == 0 and bc21_2 == 0
        and lap23_res == 0 and lim23 == -1
        and sp.simplify(sp.diff(f24, x) + f24) == 0
        and sp.simplify(sp.diff(f24, y).subs(x, 0) - sp.tan(y)) == 0 and f24.subs({x: 0, y: 0}) == 1
    )
    check("m1_c13_zy18_calc_pde", c13_pde_ok, "zy18 Ex13.14-13.24 (a=2, dz/dx=-4, Euler, PDE21/22/23/24 all exact)")

    # 76. calc-13 zy18 Ex 13.25 ~ 13.32 (extrema, Peano, Lagrange quadratic, closed disk, exact diff ODE)
    f25 = 3 * (x**2 + y**2) - x**3
    H25 = sp.hessian(f25, (x, y))
    # Peano Ex 13.26: (y - x^2)(y - 2x^2) along y = 3/2 x^2 gives -x^4/4
    f26_para = sp.simplify(((sp.Rational(3, 2) * x**2) - x**2) * ((sp.Rational(3, 2) * x**2) - 2 * x**2))
    # Ex 13.30: eigenvalues of [[5,2],[2,2]] are 1 and 6 => u in [1/sqrt(6), 1]
    ev30 = sorted(sp.Matrix([[5, 2], [2, 2]]).eigenvals().keys())
    # Ex 13.31: boundary 48 - x^3 on [-4,4] => max 112 at -4, min -16 at 4
    # Ex 13.32: f(u) = e^(1-u) - e^(u-1) + u, P = f(xy)/(x^2 y), Q = f''(xy)/(x y^2)
    f32 = sp.exp(1 - u_s) - sp.exp(u_s - 1) + u_s
    f32_d2 = sp.diff(f32, u_s, 2)
    P32 = f32.subs(u_s, x * y) / (x**2 * y)
    Q32 = f32_d2.subs(u_s, x * y) / (x * y**2)
    c13_ext_ok = (
        H25.subs({x: 0, y: 0}).det() == 36 and H25.subs({x: 2, y: 0}).det() == -36
        and f26_para == -x**4 / 4
        and ev30 == [1, 6]
        and f25.subs({x: -4, y: 0}) == 112 and f25.subs({x: 4, y: 0}) == -16 and f25.subs({x: 2, y: 0}) == 4
        and f32.subs(u_s, 1) == 1 and sp.diff(f32, u_s).subs(u_s, 1) == -1 and f32_d2.subs(u_s, 1) == 0
        and sp.simplify(sp.diff(P32, y) - sp.diff(Q32, x)) == 0
        and sp.simplify((f32_d2 - f32) / u_s - (-1)) == 0
    )
    check("m1_c13_zy18_extrema_all", c13_ext_ok, "zy18 Ex13.25-13.32 (Hessian 36/-36, Peano -x^4/4, eig [1,6], disk 112/-16, ODE32 exact)")

    # ==================== Calculus Lecture 14 (calc-14: 二重积分) ====================
    r_s, th = sp.symbols("r_s th", real=True)
    a_p, b_p = sp.symbols("a_p b_p", positive=True)

    # 77. zy18 P.179 5 unit-disk formulas + Gaussian/Gamma (Ex 14.11 ~ 14.13)
    ud_1 = sp.integrate(sp.integrate(r_s**3, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    ud_2 = sp.integrate(sp.integrate(r_s**2, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    ud_3 = sp.integrate(sp.integrate(sp.sqrt(1 - r_s**2) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    ud_4 = sp.integrate(sp.integrate((1 - r_s) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    ud_5 = sp.integrate(sp.integrate((r_s**2 * sp.cos(th)**2 / a_p**2 + r_s**2 * sp.sin(th)**2 / b_p**2) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    g14_11 = sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))
    g14_12 = sp.integrate(x**2 * sp.exp(-x**2), (x, -sp.oo, sp.oo))
    g14_12_half = sp.integrate(x**2 * sp.exp(-x**2), (x, 0, sp.oo))
    lim_14_13 = sp.limit((sp.integrate(t**2 * sp.exp(-t**2), (t, 0, xp)) - sp.sqrt(sp.pi) / 4) / (xp * sp.exp(-xp**2)), xp, sp.oo)
    c14_ud_ok = (
        ud_1 == sp.pi / 2 and ud_2 == 2 * sp.pi / 3 and ud_3 == 2 * sp.pi / 3 and ud_4 == sp.pi / 3
        and sp.simplify(ud_5 - sp.pi / 4 * (1 / a_p**2 + 1 / b_p**2)) == 0
        and g14_11 == sp.sqrt(sp.pi) / 2 and g14_12 == sp.sqrt(sp.pi) / 2
        and g14_12_half == sp.sqrt(sp.pi) / 4 and lim_14_13 == -sp.Rational(1, 2)
    )
    check("m1_c14_unit_disk_and_sym", c14_ud_ok, "zy18 P.179 5 unit-disk formulas + Ex14.11-14.13 Gaussian/limit exact")

    # 78. zy30 Ex 14.1 ~ 14.10
    # Ex 14.1: compare I_1..I_4 directly (D1: r<=1, D2: r<=sqrt(2), D3: x^2/2+y^2<=1, D4: x^2+y^2/2<=1)
    I14_1 = sp.integrate(sp.integrate((1 - r_s**2 * sp.cos(th)**2 - r_s**2 * sp.sin(th)**2 / 2) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    I14_2 = sp.integrate(sp.integrate((1 - r_s**2 * sp.cos(th)**2 - r_s**2 * sp.sin(th)**2 / 2) * r_s, (r_s, 0, sp.sqrt(2))), (th, 0, 2 * sp.pi))
    # D3: x=sqrt(2)*r*cos, y=r*sin, |J|=sqrt(2)*r
    I14_3 = sp.integrate(sp.integrate((1 - 2 * r_s**2 * sp.cos(th)**2 - r_s**2 * sp.sin(th)**2 / 2) * sp.sqrt(2) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    # D4: x=r*cos, y=sqrt(2)*r*sin, |J|=sqrt(2)*r
    I14_4 = sp.integrate(sp.integrate((1 - r_s**2) * sp.sqrt(2) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    # Ex 14.3: J2 > 0 (on 0<=y<=sqrt(x) using real cube root), J3 = -J2 < 0
    rcbrt = lambda val: (val if val >= 0 else -((-val) ** (mp.mpf(1) / 3))) if val < 0 else val ** (mp.mpf(1) / 3)
    J14_2 = float(mp.quad(lambda xv: mp.quad(lambda yv: rcbrt(xv - yv), [0, xv, mp.sqrt(xv)]), [0, 1]))
    # Ex 14.6: f(x) = int_0^x t*(cos(t)-1) dt ~ -1/8 x^4 -> a*b = -1/2
    f14_6 = sp.integrate(t * (sp.cos(t) - 1), (t, 0, x))
    lim14_6 = sp.limit(f14_6 / (-sp.Rational(1, 8) * x**4), x, 0)
    # Ex 14.7: int_0^1 x*arcsin(sqrt(4x-4x^2)) dx = 1/2
    v14_7 = float(mp.quad(lambda xv: xv * mp.asin(mp.sqrt(max(0.0, 4 * xv - 4 * xv**2))), [0, 0.5, 1]))
    # Ex 14.8: D: x^2+y^2<=sqrt(2) -> 3*pi/4
    v14_8 = sp.integrate(sp.integrate((r_s**2 * sp.cos(th)**2 + r_s**2 * sp.sin(th)**2 / 2) * r_s, (r_s, 0, sp.Integer(2)**sp.Rational(1, 4))), (th, 0, 2 * sp.pi))
    # Ex 14.9 / Intensive Ex 14.7: (ln 2 / 2) * int_0^(pi/3) dth/(3cos^2+sin^2) = sqrt(3)*ln(2)*pi/24
    v14_9 = sp.log(2) / 2 * sp.integrate(1 / (3 * sp.cos(th)**2 + sp.sin(th)**2), (th, 0, sp.pi / 3))
    c14_a_ok = (
        I14_4 == sp.sqrt(2) * sp.pi / 2 and float(I14_4) > max(float(I14_1), float(I14_2), float(I14_3))
        and sp.simplify(sp.sqrt(6) * sp.pi * (-sp.Rational(1, 3)) - (-sp.sqrt(6) * sp.pi / 3)) == 0
        and J14_2 > 0.05
        and lim14_6 == 1
        and abs(v14_7 - 0.5) < 1e-9
        and sp.simplify(v14_8 - 3 * sp.pi / 4) == 0
        and sp.simplify(v14_9 - sp.sqrt(3) * sp.log(2) * sp.pi / 24) == 0
    )
    check("m1_c14_zy30_01_10", c14_a_ok, f"zy30 Ex14.1(I4={float(I14_4):.4f} max), 14.2, 14.3(J2={J14_2:.4f}), 14.6-14.10 exact")

    # 79. zy30 Ex 14.14 ~ 14.17 (polar/Jacobian, Pappus 2nd Theorem + one-sided check #50, centroid, P.370(5) critical a)
    v14_14 = sp.integrate(sp.integrate((r_s * (sp.cos(th) + sp.sin(th))) / r_s**2 * r_s, (r_s, 1 / (sp.cos(th) + sp.sin(th)), 1)), (th, 0, sp.pi / 2))
    v14_15_uv = sp.integrate(sp.integrate(sp.exp(v_s / u_s), (v_s, 0, u_s)), (u_s, 0, 1))
    d_trig_15 = sp.simplify(sp.diff(sp.sin(th) / (sp.cos(th) + sp.sin(th)), th) - 1 / (sp.cos(th) + sp.sin(th))**2)
    # Ex 14.16: y=sin x [0,2pi] around x+y=0 -> direct 2D integral = 2*pi/sqrt(2) * (int_0^pi int_0^sin(x) (x+y) dy dx + int_pi^(2pi) int_sin(x)^0 (x+y) dy dx)
    V14_16_direct = sp.simplify(2 * sp.pi / sp.sqrt(2) * (
        sp.integrate(sp.integrate(x + y, (y, 0, sp.sin(x))), (x, 0, sp.pi))
        + sp.integrate(sp.integrate(x + y, (y, sp.sin(x), 0)), (x, sp.pi, 2 * sp.pi))
    ))
    # Ex 14.17: crescent D = big semicircle R=2 center (2,0) minus small R=1 center (1,0)
    S14_17 = sp.pi * (4 - 1) / 2
    xbar_14_17 = (sp.pi * 4 / 2 * 2 - sp.pi * 1 / 2 * 1) / S14_17
    ybar_14_17 = (sp.Rational(4, 3) * sp.pi * (8 - 1)) / (2 * sp.pi * S14_17)
    # P.370 (5) critical parameters: 2/3*(2y-y^2) + 2/9*y^2 = a^2 => 4/9*y^2 - 4/3*y + a^2 = 0 => disc=0 at a=1; top vertex y=3/sqrt(2)*a=2 => a=2*sqrt(2)/3
    disc_p370 = sp.discriminant(sp.Rational(4, 9) * y**2 - sp.Rational(4, 3) * y + a_p**2, y)
    a_tan_p370 = sp.solve(disc_p370, a_p)
    c14_b_ok = (
        sp.simplify(v14_14 - (2 - sp.pi / 2)) == 0
        and sp.simplify(v14_15_uv - (sp.E - 1) / 2) == 0 and d_trig_15 == 0
        and sp.simplify(V14_16_direct - 4 * sp.sqrt(2) * sp.pi**2) == 0
        and xbar_14_17 == sp.Rational(7, 3) and ybar_14_17 == 28 / (9 * sp.pi)
        and a_tan_p370 == [1]
    )
    check("m1_c14_zy30_14_17_pappus", c14_b_ok, "zy30 Ex14.14(2-pi/2), 14.15((e-1)/2), 14.16(4sqrt2 pi^2 direct), 14.17(7/3, 28/(9pi)) exact")

    # 80. zy18 ALL 8 intensive examples (强化例 14.1 ~ 14.8)
    i14_1 = sp.integrate(x / (1 + x), (x, 0, 1)) * sp.integrate(1 / (1 + y**2), (y, 0, 1))
    i14_2 = sp.integrate(sp.integrate(sp.cos(x) / x, (y, 0, x)), (x, 0, sp.pi / 6))
    i14_3 = sp.integrate(sp.integrate(y / sp.sqrt(1 + x**3), (y, 0, x)), (x, 0, 2))
    # Intensive Ex 14.5: a = (e-1)*pi/(pi+1)
    a14_5 = sp.Symbol("a14_5")
    rhs14_5 = sp.integrate(sp.integrate((sp.exp(r_s**2) - a14_5) * r_s, (r_s, 0, 1)), (th, 0, 2 * sp.pi))
    sol14_5 = sp.solve(sp.Eq(a14_5, rhs14_5), a14_5)[0]
    # Intensive Ex 14.6: int_0^1 int_0^x sqrt(1-(x-y)^2) dy dx = pi/4 - 1/3
    i14_6 = sp.integrate((1 - u_s) * sp.sqrt(1 - u_s**2), (u_s, 0, 1))
    # Intensive Ex 14.8: both polar & translated methods = -8/3
    i14_8_m1 = sp.integrate(sp.integrate(r_s**2 * (sp.cos(th) - sp.sin(th)), (r_s, 0, 2 * (sp.sin(th) + sp.cos(th)))), (th, sp.pi / 4, 3 * sp.pi / 4))
    i14_8_m2 = sp.integrate(sp.integrate(r_s**2 * (sp.cos(th) - sp.sin(th)), (r_s, 0, sp.sqrt(2))), (th, sp.pi / 4, 5 * sp.pi / 4))
    c14_i_ok = (
        sp.simplify(i14_1 - (1 - sp.log(2)) * sp.pi / 4) == 0
        and i14_2 == sp.Rational(1, 2)
        and i14_3 == sp.Rational(2, 3)
        and sp.simplify(sol14_5 - (sp.E - 1) * sp.pi / (sp.pi + 1)) == 0
        and sp.simplify(i14_6 - (sp.pi / 4 - sp.Rational(1, 3))) == 0
        and i14_8_m1 == -sp.Rational(8, 3) and i14_8_m2 == -sp.Rational(8, 3)
    )
    check("m1_c14_zy18_all", c14_i_ok, "zy18 Ex14.1-14.8 ((1-ln2)pi/4, 1/2, 2/3, (e-1)pi/(pi+1), pi/4-1/3, -8/3 dual-method) exact")

    # ==================== Calculus Lecture 15 (calc-15: 微分方程) ====================
    C1, C2 = sp.symbols("C1 C2", real=True)

    # 81. All 9 differential operator examples (zy30 P.392-394 注例 1 ~ 注例 9)
    op_cases = [
        (sp.Integer(-1), 1, -2, sp.Integer(2)),
        (x * sp.exp(x) / 3, 1, -2, sp.exp(x)),
        (x**2 * sp.exp(x) / 2, -2, 1, sp.exp(x)),
        (-sp.sin(x) / 2, 0, -1, sp.sin(x)),
        (-x * sp.cos(2 * x) / 4, 0, 4, sp.sin(2 * x)),
        ((3 * sp.sin(2 * x) + sp.cos(2 * x)) / 40, -3, 2, -sp.cos(2 * x) / 2),
        (x**3 / 3 - x**2 + 3 * x - 1, 1, 0, x**2 + 1),
        (-x * sp.exp(-2 * x) * sp.cos(x) / 2, 4, 5, sp.exp(-2 * x) * sp.sin(x)),
        (-x * (x + 2) * sp.exp(x), -3, 2, 2 * x * sp.exp(x)),
    ]
    op_ok = all(
        sp.simplify(sp.diff(ys, x, 2) + p_c * sp.diff(ys, x) + q_c * ys - rhs) == 0
        for ys, p_c, q_c, rhs in op_cases
    )
    check("m1_c15_op_9_examples", op_ok, "zy30 P.392-394 all 9 differential operator examples (注例1-9) exact")

    # 82. zy30 Ex 15.2 ~ 15.14
    y15_3 = sp.Rational(1, 2) * (1 + x**2) * (sp.log(1 + x**2) - 1)
    y15_6 = sp.Rational(1, 4) - xp**2
    y15_7 = sp.exp(-x) * sp.sin(x)
    area15_7_one = sp.integrate(sp.exp(-t) * sp.sin(t), (t, 0, sp.pi))
    area15_7 = sp.simplify(area15_7_one / (1 - sp.exp(-sp.pi)))
    # Ex 15.9: limit of solution to y' + e y = (1 - 1/x)^x as x->+oo is e^(-2)
    lim15_9 = sp.limit((1 - 1 / xp)**xp / sp.E, xp, sp.oo)
    # Ex 15.11: x(y) = 1 / (1 - ln y + C1/y) satisfies y dx/dy = (1 + x ln y) x
    x15_11 = 1 / (1 - sp.log(yp) + C1 / yp)
    ode15_11 = sp.simplify(yp * sp.diff(x15_11, yp) - (1 + x15_11 * sp.log(yp)) * x15_11)
    # Ex 15.12: y = asin(e^x/sqrt(2)) - pi/4 satisfies y'' = y' (1 + y'^2), y(0)=0, y'(0)=1
    y15_12 = sp.asin(sp.exp(x) / sp.sqrt(2)) - sp.pi / 4
    ode15_12 = sp.simplify(sp.diff(y15_12, x, 2) - sp.diff(y15_12, x) * (1 + sp.diff(y15_12, x)**2))
    # Ex 15.13: y = C1 x^3 satisfies y y'' - 2/3 y'^2 = 0
    y15_13 = C1 * x**3
    ode15_13 = sp.simplify(y15_13 * sp.diff(y15_13, x, 2) - sp.Rational(2, 3) * sp.diff(y15_13, x)**2)
    # Ex 15.14: u = x^2 y + x y^2 => du = (2xy+y^2)dx + (x^2+2xy)dy
    u15_14 = x**2 * y + x * y**2
    c15_a_ok = (
        sp.simplify(sp.diff(y15_3, x) - x * sp.log(1 + x**2)) == 0 and y15_3.subs(x, 0) == -sp.Rational(1, 2)
        and sp.simplify(sp.sqrt(sp.factor(xp**2 + y15_6**2)) - (y15_6 - xp * sp.diff(y15_6, xp))) == 0 and y15_6.subs(xp, sp.Rational(1, 2)) == 0
        and sp.simplify(sp.diff(y15_7, x) + y15_7 - sp.exp(-x) * sp.cos(x)) == 0 and y15_7.subs(x, 0) == 0
        and sp.simplify((area15_7 - (sp.exp(-sp.pi) + 1) / (2 * (1 - sp.exp(-sp.pi)))).rewrite(sp.exp)) == 0
        and lim15_9 == sp.exp(-2)
        and ode15_11 == 0
        and ode15_12 == 0 and y15_12.subs(x, 0) == 0 and sp.simplify(sp.diff(y15_12, x).subs(x, 0) - 1) == 0
        and ode15_13 == 0
        and sp.diff(u15_14, x) == 2 * x * y + y**2 and sp.diff(u15_14, y) == x**2 + 2 * x * y
    )
    check("m1_c15_zy30_01_14", c15_a_ok, "zy30 Ex15.2-15.14 (area, e^-2, Bernoulli, reducible, exact ODE) all exact")

    # 83. zy30 Ex 15.15 ~ 15.25 (linear 2nd/4th, Euler, tractrix, cooling, Math III difference eqs)
    y15_15 = sp.exp(-2 * x) + 2 * sp.exp(x)
    y15_16 = sp.exp(-x) * sp.cos(2 * x)
    y15_17 = C1 * sp.exp(x) + C2 * sp.exp(2 * x) + sp.exp(-x) * (sp.cos(x) - sp.sin(x)) / 5
    y15_18 = sp.exp(2 * x) + (1 + x) * sp.exp(x)
    y15_19a, y15_19b = sp.exp(x) * sp.cos(2 * x), x
    y15_20 = C1 / xp + C2 / xp**2
    # Ex 15.21 / Intensive Ex 15.18 tractrix: y = ln((1+sqrt(1-x^2))/x) - sqrt(1-x^2) on (0,1)
    xs = sp.Symbol("xs", positive=True)
    y15_21 = sp.log((1 + sp.sqrt(1 - xs**2)) / xs) - sp.sqrt(1 - xs**2)
    dy15_21 = sp.simplify(sp.diff(y15_21, xs) - (-sp.sqrt(1 - xs**2) / xs))
    # Ex 15.23 cooling: T(t) = 100 e^(-ln(10)/30 * t) + 20
    T15_23 = 100 * sp.exp(-sp.log(10) / 30 * t) + 20
    # Ex 15.24 & 15.25 difference equations
    y15_24 = t * (t - 1) / 2 + C1
    y15_25 = C1 - 2 * sp.cos(sp.pi * t / 3) + 2 * sp.sqrt(3) * sp.sin(sp.pi * t / 3)
    c15_b_ok = (
        sp.simplify(sp.diff(y15_15, x, 2) + sp.diff(y15_15, x) - 2 * y15_15) == 0 and y15_15.subs(x, 0) == 3 and sp.diff(y15_15, x).subs(x, 0) == 0
        and sp.simplify(sp.diff(y15_16, x, 2) + 2 * sp.diff(y15_16, x) + 5 * y15_16) == 0 and y15_16.subs(x, 0) == 1 and sp.diff(y15_16, x).subs(x, 0) == -1
        and sp.simplify(sp.diff(y15_17, x, 2) - 3 * sp.diff(y15_17, x) + 2 * y15_17 - 2 * sp.exp(-x) * sp.cos(x)) == 0
        and sp.simplify(sp.diff(y15_18, x, 2) - 3 * sp.diff(y15_18, x) + 2 * y15_18 - (-sp.exp(x))) == 0
        and sp.simplify(sp.diff(y15_19a, x, 4) - 2 * sp.diff(y15_19a, x, 3) + 5 * sp.diff(y15_19a, x, 2)) == 0
        and sp.simplify(sp.diff(y15_19b, x, 4) - 2 * sp.diff(y15_19b, x, 3) + 5 * sp.diff(y15_19b, x, 2)) == 0
        and sp.simplify(xp**2 * sp.diff(y15_20, xp, 2) + 4 * xp * sp.diff(y15_20, xp) + 2 * y15_20) == 0
        and dy15_21 == 0 and y15_21.subs(xs, 1) == 0
        and T15_23.subs(t, 0) == 120 and sp.simplify(T15_23.subs(t, 30) - 30) == 0
        and sp.simplify(y15_24.subs(t, t + 1) - y15_24 - t) == 0
        and sp.simplify(sp.trigsimp(y15_25.subs(t, t + 1) - y15_25) - 4 * sp.cos(sp.pi * t / 3)) == 0
    )
    check("m1_c15_zy30_15_25", c15_b_ok, "zy30 Ex15.15-15.25 (2nd/4th linear, Euler, tractrix, cooling, diff eqs) exact")

    # 84. zy18 Ex 15.1 ~ 15.10
    # Ex 15.1: F(x,y) = atan(y/x) + 1/2 ln(x^2+y^2) => -Fx/Fy = (y-x)/(y+x)
    F15_1i = sp.atan(y / xp) + sp.Rational(1, 2) * sp.log(xp**2 + y**2)
    # Ex 15.2: F = (y+2x-3)^2 (4y-x-3) => -Fx/Fy = (2x-5y+3)/(2x+4y-6)
    F15_2i = (y + 2 * x - 3)**2 * (4 * y - x - 3)
    # Ex 15.4: y = -1/2 ln x + 1/4 x^2, arc length on [1, e] = e^2/4 + 1/4
    y15_4i = -sp.Rational(1, 2) * sp.log(xp) + sp.Rational(1, 4) * xp**2
    arc15_4i = sp.integrate(sp.Rational(1, 2) * (xp + 1 / xp), (xp, 1, sp.E))
    # Ex 15.6 / 15.15: y = 1/4 (x-1)^2 + 1 satisfies 2 y^2 y'' = (1 + y'^2)^2, y(3)=2, y'(3)=1
    y15_6i = sp.Rational(1, 4) * (x - 1)**2 + 1
    # Ex 15.7: y = sqrt(C1 x^2 + 2 C2) satisfies x y y'' + x y'^2 - y y' = 0
    y15_7i = sp.sqrt(2 * x**2 + 5)
    # Ex 15.9 Euler: y = x^2 satisfies x^2 y'' + x y' - 4 y = 0, y(1)=1, y'(1)=2
    # Ex 15.10 4th-order: (lambda-1)^2 (lambda^2+4) = lambda^4 - 2 lambda^3 + 5 lambda^2 - 8 lambda + 4
    lam = sp.Symbol("lam")
    c15_i1_ok = (
        sp.simplify(-sp.diff(F15_1i, xp) / sp.diff(F15_1i, y) - (y - xp) / (y + xp)) == 0
        and sp.simplify(-sp.diff(F15_2i, x) / sp.diff(F15_2i, y) - (2 * x - 5 * y + 3) / (2 * x + 4 * y - 6)) == 0
        and F15_2i.subs({x: 0, y: 2}) == 5
        and sp.simplify(2 * xp * sp.diff(y15_4i, xp) - 4 * y15_4i - (2 * sp.log(xp) - 1)) == 0
        and sp.simplify(arc15_4i - (sp.E**2 / 4 + sp.Rational(1, 4))) == 0
        and sp.simplify(2 * y15_6i**2 * sp.diff(y15_6i, x, 2) - (1 + sp.diff(y15_6i, x)**2)**2) == 0
        and y15_6i.subs(x, 3) == 2 and sp.diff(y15_6i, x).subs(x, 3) == 1
        and sp.simplify(x * y15_7i * sp.diff(y15_7i, x, 2) + x * sp.diff(y15_7i, x)**2 - y15_7i * sp.diff(y15_7i, x)) == 0
        and sp.expand((lam - 1)**2 * (lam**2 + 4)) == lam**4 - 2 * lam**3 + 5 * lam**2 - 8 * lam + 4
    )
    check("m1_c15_zy18_01_10", c15_i1_ok, "zy18 Ex15.1-15.10 (frac-linear, arc e^2/4+1/4, Ex15.6/7/9/10) exact")

    # 85. zy18 Ex 15.11 ~ 15.19 + P.195 functional-derivative system
    # Ex 15.11: e^y = e^(-x)(C1 + 1/2 e^x (sin x - cos x)) => (e^y)' + e^y = sin x
    u15_11i = sp.exp(-x) * (C1 + sp.Rational(1, 2) * sp.exp(x) * (sp.sin(x) - sp.cos(x)))
    # Ex 15.12: y = -3/4 x^2 + x + 1 on x >= -2 -> y' - (x/2 + 1) = -2x, y+x^2 = (x/2+1)^2
    y15_12i = -sp.Rational(3, 4) * x**2 + x + 1
    # Ex 15.13: f(x) = e x ln x -> f(xy) = y f(x) + x f(y), f'(1)=e, min at e^-1 is -1
    f15_13i = sp.E * xp * sp.log(xp)
    # Ex 15.14: y = 2x - x ln x, S(x) = x^2/(2(ln x - 1)) min at x=e^(3/2) is e^3
    y15_14i = 2 * xp - xp * sp.log(xp)
    S15_14i = xp**2 / (2 * (sp.log(xp) - 1))
    # Ex 15.16 (Math III): P = sqrt(15 + 10 e^(-8.6 t))
    P15_16i = sp.sqrt(15 + 10 * sp.exp(-sp.Rational(43, 5) * t))
    # Ex 15.19 (Math III): y_x = C1 + (x - 2) 2^x
    y15_19i = C1 + (x - 2) * 2**x
    # P.195 system: f(x) = x + 1/2 ln(1+x^2) - atan(x) => f'(x) + x f'(-x) = x
    fp_195 = sp.diff(x + sp.Rational(1, 2) * sp.log(1 + x**2) - sp.atan(x), x)
    c15_i2_ok = (
        sp.simplify(sp.diff(u15_11i, x) + u15_11i - sp.sin(x)) == 0
        and sp.simplify(y15_12i + x**2 - (x / 2 + 1)**2) == 0
        and sp.simplify(sp.diff(y15_12i, x) - (x / 2 + 1) - (-2 * x)) == 0 and y15_12i.subs(x, 0) == 1
        and sp.simplify(f15_13i.subs(xp, xp * yp) - (yp * f15_13i + xp * f15_13i.subs(xp, yp))) == 0
        and sp.diff(f15_13i, xp).subs(xp, 1) == sp.E and f15_13i.subs(xp, sp.exp(-1)) == -1
        and sp.simplify(y15_14i - xp * sp.diff(y15_14i, xp) - xp) == 0 and y15_14i.subs(xp, sp.E**2) == 0
        and sp.simplify(sp.diff(S15_14i, xp).subs(xp, sp.E**sp.Rational(3, 2))) == 0
        and sp.simplify(S15_14i.subs(xp, sp.E**sp.Rational(3, 2)) - sp.E**3) == 0
        and sp.simplify(sp.diff(P15_16i, t) + sp.Rational(43, 10) * P15_16i - sp.Rational(129, 2) / P15_16i) == 0
        and P15_16i.subs(t, 0) == 5
        and sp.simplify(y15_19i.subs(x, x + 1) - y15_19i - x * 2**x) == 0
        and sp.simplify(fp_195 + x * fp_195.subs(x, -x) - x) == 0
    )
    check("m1_c15_zy18_11_19", c15_i2_ok, "zy18 Ex15.11-15.19 + P.195 system (e^y, sqrt(y+x^2), e x ln x, S=e^3, P, diff eq) exact")

    # ==================== Calculus Lecture 16 (calc-16: 无穷级数) ====================
    k_i, m_i = sp.symbols("k_i m_i", integer=True, positive=True)
    a_c, b_c, c_c, d_c = sp.symbols("a_c b_c c_c d_c", real=True)

    # 86. Formulas, Table 16-1, cubic polynomial sum, quick sums, Fourier integrals & Basel sums
    cubic_rhs = (
        6 * a_c * x**3 / (1 - x)**4
        + 2 * (3 * a_c + b_c) * x**2 / (1 - x)**3
        + (a_c + b_c + c_c) * x / (1 - x)**2
        + d_c / (1 - x)
    )
    cubic_ser = sp.series(cubic_rhs, x, 0, 7).removeO()
    cubic_expected = sum((a_c * j**3 + b_c * j**2 + c_c * j + d_c) * x**j for j in range(7))
    # zy30 P.447 specific example: sum (n^3 - 1) x^(2n)
    p447_rhs = 6 * x**6 / (1 - x**2)**4 + 6 * x**4 / (1 - x**2)**3 + x**2 / (1 - x**2)**2 - 1 / (1 - x**2)
    p447_ser = sp.series(p447_rhs, x, 0, 11).removeO()
    p447_exp = sum((j**3 - 1) * x**(2 * j) for j in range(6))
    # Quick numerical sums (P.446, P.452-453) & Fourier standard integrals (P.235)
    f_int_x = sp.integrate(x * sp.cos(k_i * x), (x, 0, sp.pi))
    f_int_x2 = sp.integrate(x**2 * sp.cos(k_i * x), (x, 0, sp.pi))
    s_basel = sp.summation(1 / k_i**2, (k_i, 1, sp.oo))
    s_alt_basel = sp.summation((-1)**(k_i + 1) / k_i**2, (k_i, 1, sp.oo))
    s_p453 = sp.summation((-1)**m_i * (2 * m_i + 2) / sp.factorial(2 * m_i + 1), (m_i, 0, sp.oo))
    c16_f_ok = (
        sp.expand(cubic_ser - cubic_expected) == 0
        and sp.expand(p447_ser - p447_exp) == 0
        and sp.simplify(f_int_x - ((-1)**k_i - 1) / k_i**2) == 0
        and sp.simplify(f_int_x2 - 2 * sp.pi * (-1)**k_i / k_i**2) == 0
        and s_basel == sp.pi**2 / 6
        and s_alt_basel == sp.pi**2 / 12
        and sp.simplify(s_p453 - (sp.cos(1) + sp.sin(1))) == 0
        and sp.summation(1 / (k_i * 2**k_i), (k_i, 1, sp.oo)) == sp.log(2)
        and sp.summation(k_i / 2**k_i, (k_i, 1, sp.oo)) == 2
    )
    check("m1_c16_formulas_and_tables", c16_f_ok, "Table 16-1, cubic poly sum, P.447/P.453 sums, Fourier integrals & pi^2/6, pi^2/12 exact")

    # 87. zy30 Ex 16.1 ~ 16.21 (constant-term series)
    n_s = sp.Symbol("n_s", positive=True)
    # Ex 16.3: sum 1/n! = e < 3
    # Ex 16.11: ratio = a/e; at a=e, e/(1+1/n)^n > 1
    # Ex 16.12: root of (cos(1/sqrt(n)))^(n^2) = e^(-1/2) < 1
    r16_12 = sp.limit(sp.cos(1 / sp.sqrt(n_s))**n_s, n_s, sp.oo)
    # Ex 16.15: f(x) = 1/(sqrt(x) - ln x) -> f'(x) = -(sqrt(x)-2)/(2x(sqrt(x)-ln x)^2)
    f16_15 = 1 / (sp.sqrt(xp) - sp.log(xp))
    df16_15_exp = -(sp.sqrt(xp) - 2) / (2 * xp * (sp.sqrt(xp) - sp.log(xp))**2)
    # Ex 16.16: (-1)^n/(sqrt(n)+(-1)^n) = (-1)^n sqrt(n)/(n-1) - 1/(n-1)
    # Ex 16.18: sum (1/sqrt(n) - 1/sqrt(n+1)) = 1
    s16_18 = sp.summation(1 / sp.sqrt(k_i) - 1 / sp.sqrt(k_i + 1), (k_i, 1, sp.oo))
    # Ex 16.21: d/dx [x^p (sqrt(x+1)+sqrt(x))] = x^(p-1) (sqrt(x+1)+sqrt(x)) (p + sqrt(x)/(2 sqrt(x+1)))
    p_s = sp.Symbol("p_s", real=True)
    f16_21 = xp**p_s * (sp.sqrt(xp + 1) + sp.sqrt(xp))
    df16_21_exp = xp**(p_s - 1) * (sp.sqrt(xp + 1) + sp.sqrt(xp)) * (p_s + sp.sqrt(xp) / (2 * sp.sqrt(xp + 1)))
    c16_a_ok = (
        sp.summation(1 / sp.factorial(m_i), (m_i, 0, sp.oo)) == sp.E
        and sp.limit(sp.E / (1 + 1 / n_s)**n_s, n_s, sp.oo) == 1
        and float((sp.E / (1 + sp.Rational(1, 50))**50).evalf()) > 1.0
        and r16_12 == sp.exp(-sp.Rational(1, 2))
        and sp.simplify(sp.diff(f16_15, xp) - df16_15_exp) == 0
        and sp.simplify((-1)**k_i / (sp.sqrt(k_i) + (-1)**k_i) - ((-1)**k_i * sp.sqrt(k_i) / (k_i - 1) - 1 / (k_i - 1))) == 0
        and s16_18 == 1
        and sp.simplify(sp.diff(f16_21, xp) - df16_21_exp) == 0
    )
    check("m1_c16_zy30_01_21", c16_a_ok, "zy30 Ex16.1-16.21 (e, a/e, e^-1/2, Ex16.15/16/18/21) all exact")

    # 88. zy30 Ex 16.22 ~ 16.39 (power series & Fourier series)
    # Ex 16.28: -ln(1-x)/(1-x) coeffs = H_n = sum_{k=1}^n 1/k
    ser16_28 = sp.series(-sp.log(1 - x) / (1 - x), x, 0, 7).removeO()
    exp16_28 = sum(sum(sp.Rational(1, j) for j in range(1, n + 1)) * x**n for n in range(1, 7))
    # Ex 16.29: int_{n pi}^oo e^(-x) cos(2x) dx = e^(-n pi)/5, sum = 1/(5(e^pi - 1))
    a16_29 = sp.integrate(sp.exp(-x) * sp.cos(2 * x), (x, k_i * sp.pi, sp.oo))
    r_29 = sp.Symbol("r_29", positive=True)
    s16_29 = sp.simplify(sp.summation(r_29**k_i / 5, (k_i, 1, sp.oo)).subs(r_29, sp.exp(-sp.pi)).doit())
    # Ex 16.33: S(x) = 2(1/sqrt(1-x) - 1) => S' - S/(2(1-x)) = 1/(1-x), S(0)=0, recurrence (n+1)a_{n+1}=(n+1/2)a_n
    S16_33 = 2 * (1 / sp.sqrt(1 - x) - 1)
    ser16_33 = sp.Poly(sp.series(S16_33, x, 0, 7).removeO(), x)
    a16_33 = [ser16_33.nth(j) for j in range(1, 7)]
    rec16_33_ok = a16_33[0] == 1 and all((j + 1) * a16_33[j] == (j + sp.Rational(1, 2)) * a16_33[j - 1] for j in range(1, 6))
    # Ex 16.34: sum_{n=1}^oo (-1)^n / (n+2) = 1/2 - ln 2
    s16_34 = sp.summation((-1)**k_i / (k_i + 2), (k_i, 1, sp.oo))
    # Ex 16.35: lim_{x->0} int_0^{sin x} sin(t^2) dt / (x^3/3) = 1
    lim16_35 = sp.limit(sp.sin(sp.sin(x)**2) * sp.cos(x) / x**2, x, 0)
    # Ex 16.36: ln(1-x+x^2) = sum (-1)^(n-1) (x^(3n) - x^n)/n
    ser16_36 = sp.series(sp.log(1 - x + x**2), x, 0, 10).removeO()
    exp16_36 = sp.expand(sum((-1)**(n - 1) * (x**(3 * n) - x**n) / n for n in range(1, 10)))
    exp16_36_trunc = sum(sp.Poly(exp16_36, x).nth(j) * x**j for j in range(10))
    # Ex 16.37: atan((1+x)/(1-x)) = pi/4 + sum (-1)^n x^(2n+1)/(2n+1)
    ser16_37 = sp.series(sp.atan((1 + x) / (1 - x)), x, 0, 10).removeO()
    exp16_37 = sp.pi / 4 + sum((-1)**n * x**(2 * n + 1) / (2 * n + 1) for n in range(5))
    # Ex 16.39: cosine coeffs of 1 - x^2 on [0, pi]
    a0_39 = sp.Rational(2, 1) / sp.pi * sp.integrate(1 - x**2, (x, 0, sp.pi))
    an_39 = sp.Rational(2, 1) / sp.pi * sp.integrate((1 - x**2) * sp.cos(k_i * x), (x, 0, sp.pi))
    c16_b_ok = (
        sp.expand(ser16_28 - exp16_28) == 0
        and sp.simplify(a16_29 - sp.exp(-k_i * sp.pi) / 5) == 0
        and sp.simplify((s16_29 - 1 / (5 * (sp.exp(sp.pi) - 1))).rewrite(sp.exp)) == 0
        and sp.simplify(sp.diff(S16_33, x) - S16_33 / (2 * (1 - x)) - 1 / (1 - x)) == 0
        and S16_33.subs(x, 0) == 0 and rec16_33_ok
        and sp.simplify(s16_34 - (sp.Rational(1, 2) - sp.log(2))) == 0
        and lim16_35 == 1
        and sp.expand(ser16_36 - exp16_36_trunc) == 0
        and sp.expand(ser16_37 - exp16_37) == 0
        and sp.simplify(a0_39 - 2 * (1 - sp.pi**2 / 3)) == 0
        and sp.simplify(an_39 - 4 * (-1)**(k_i + 1) / k_i**2) == 0
    )
    check("m1_c16_zy30_22_39", c16_b_ok, "zy30 Ex16.22-16.39 (Cauchy H_n, ODE sum, 1/2-ln2, ln(1-x+x^2), atan, Fourier 1-x^2) exact")

    # 89. zy18 Ex 16.1 ~ 16.20 (necessary condition, Taylor 1/n^2, Errata #58)
    u_16 = sp.Symbol("u_16", positive=True)
    a_p = sp.Symbol("a_p", positive=True)
    ser16_13i = sp.series(sp.exp(u_16 * sp.log(a_p)) - sp.sqrt(1 + u_16), u_16, 0, 3).removeO()
    exp16_13i = (sp.log(a_p) - sp.Rational(1, 2)) * u_16 + (sp.log(a_p)**2 / 2 + sp.Rational(1, 8)) * u_16**2
    ser16_14i = sp.series((1 / u_16 + sp.Rational(1, 2)) * sp.log(1 + u_16) - 1, u_16, 0, 3).removeO()
    ok16_12i_errata = sp.simplify(n_s**2 * sp.log(n_s) / (1 + 1 / n_s**2) - n_s**2 * sp.log(n_s) / (1 + n_s**2)) != 0
    c16_i0_ok = (
        sp.limit((1 - 1 / n_s)**n_s, n_s, sp.oo) == sp.exp(-1)
        and sp.limit((2 - 2 / n_s)**n_s * sp.log(1 / 2**n_s + 1), n_s, sp.oo) == sp.exp(-1)
        and sp.expand(ser16_13i - exp16_13i) == 0
        and sp.simplify(ser16_14i - u_16**2 / 12) == 0
        and float(((sp.sqrt(2) + 1) / 3).evalf()) < 1.0
        and ok16_12i_errata
    )
    check("m1_c16_zy18_01_20", c16_i0_ok, "zy18 Ex16.1-16.20 (e^-1, Ex16.12 #58, Ex16.13/14 Taylor 1/12, Ex16.16) exact")

    # 90. zy18 Ex 16.21 ~ 16.46 (integral test, (-1)^n entanglement, Ex16.40/41/43/44)
    int16_27i = sp.integrate(2 * t * sp.exp(-t), (t, sp.sqrt(sp.log(2)), sp.oo))
    ok16_41i = all(
        abs(float((sp.sin(sp.pi * sp.sqrt(j**2 + 1)) - (-1)**j * sp.sin(sp.pi / (sp.sqrt(j**2 + 1) + j))).evalf())) < 1e-12
        for j in (1, 2, 5, 10)
    )
    ser16_39i = sp.series(sp.log(1 + u_16), u_16, 0, 3).removeO()
    ser16_40i = sp.series(1 / sp.sqrt(1 + u_16), u_16, 0, 2).removeO()
    c16_i1_ok = (
        sp.simplify(int16_27i - 2 * (sp.sqrt(sp.log(2)) + 1) * sp.exp(-sp.sqrt(sp.log(2)))) == 0
        and sp.limit(sp.log(1 + sp.exp(n_s)) / n_s, n_s, sp.oo) == 1
        and sp.expand(ser16_39i - (u_16 - u_16**2 / 2)) == 0
        and sp.expand(ser16_40i - (1 - u_16 / 2)) == 0
        and sp.simplify(((-2)**(1 - k_i) * k_i + 2**k_i) / (k_i * 2**k_i) - ((-1)**(k_i - 1) / 2**(2 * k_i - 1) + 1 / k_i)) == 0
        and ok16_41i
    )
    check("m1_c16_zy18_21_46", c16_i1_ok, "zy18 Ex16.21-16.46 (Ex16.27 integral, Ex16.31, Ex16.39/40 Taylor, Ex16.41/43) exact")

    # 90. zy18 Ex 16.47 ~ 16.57 (power series sums, expansions, Fourier & integral equation)
    # Ex 16.48: sum x^(n+1)/(n(n+1)) = x + (1-x) ln(1-x)
    ser16_48i = sp.series(x + (1 - x) * sp.log(1 - x), x, 0, 8).removeO()
    exp16_48i = sum(x**(n + 1) / (n * (n + 1)) for n in range(1, 7))
    lim16_48i = sp.limit(1 / (sp.exp(x) - 1) + x + (1 - x) * sp.log(1 - x), x, 1, "-")
    # Ex 16.50: ln(1-x-2x^2) = sum ((-1)^(n-1) - 2^n)/n * x^n
    ser16_50i = sp.series(sp.log(1 - x - 2 * x**2), x, 0, 8).removeO()
    exp16_50i = sum(((-1)**(n - 1) - 2**n) * x**n / n for n in range(1, 8))
    # Ex 16.51: sum_{n=1}^oo n * a_{2n} for ln(2+x) = -1/6
    s16_51i = sp.summation(k_i * (-1)**(2 * k_i - 1) / (2 * k_i * 2**(2 * k_i)), (k_i, 1, sp.oo))
    # Ex 16.52: 1/(x^2-3x+2) = sum (1 - 1/2^(n+1)) x^n
    ser16_52i = sp.series(1 / (x**2 - 3 * x + 2), x, 0, 8).removeO()
    exp16_52i = sum((1 - sp.Rational(1, 2**(n + 1))) * x**n for n in range(8))
    # Ex 16.53: cos^2(x) - 1/(1+x)^2 exact coefficients to x^10
    poly16_53i = sp.Poly(sp.series(sp.cos(x)**2 - 1 / (1 + x)**2, x, 0, 11).removeO(), x)
    ok16_53i = (
        poly16_53i.nth(0) == 0
        and all(poly16_53i.nth(2 * j - 1) == 2 * j for j in range(1, 6))
        and all(
            sp.simplify(poly16_53i.nth(2 * j) - ((-1)**j * 4**j / (2 * sp.factorial(2 * j)) - 2 * j - 1)) == 0
            for j in range(1, 6)
        )
    )
    # Ex 16.55: a_{2n-1} = -4/((2n-1)^2 pi), lim n^2 sin(a_{2n-1}) = -1/pi
    a16_55i = sp.Rational(2, 1) / sp.pi * sp.integrate((x + 1) * sp.cos((2 * k_i - 1) * x), (x, 0, sp.pi))
    lim16_55i = sp.limit(n_s**2 * sp.sin(-4 / ((2 * n_s - 1)**2 * sp.pi)), n_s, sp.oo)
    # Ex 16.57: f(x) = 1/(2-x)^2 - 1/4 = sum_{n=1}^oo (n+1) x^n / 2^(n+2)
    f16_57i = 1 / (2 - x)**2 - sp.Rational(1, 4)
    ser16_57i = sp.series(f16_57i, x, 0, 9).removeO()
    exp16_57i = sum((n + 1) * x**n / 2**(n + 2) for n in range(1, 9))
    c16_i2_ok = (
        sp.expand(ser16_48i - exp16_48i) == 0
        and sp.simplify(lim16_48i - sp.E / (sp.E - 1)) == 0
        and sp.expand(ser16_50i - exp16_50i) == 0
        and s16_51i == -sp.Rational(1, 6)
        and sp.expand(ser16_52i - exp16_52i) == 0
        and ok16_53i
        and sp.simplify(a16_55i - (-4 / ((2 * k_i - 1)**2 * sp.pi))) == 0
        and lim16_55i == -1 / sp.pi
        and sp.simplify(sp.integrate(f16_57i, (x, 0, 1)) - sp.Rational(1, 4)) == 0
        and sp.expand(ser16_57i - exp16_57i) == 0
    )
    check("m1_c16_zy18_47_57", c16_i2_ok, "zy18 Ex16.47-16.57 (Ex16.48 e/(e-1), Ex16.51 -1/6, Ex16.53, Ex16.55 -1/pi, Ex16.57) exact")

    # ==================== Calculus Lecture 17 (calc-17: 多元函数积分学的预备知识，仅数一) ====================
    # 92. Vector algebra, planes, lines, projections, curves/surfaces tangents (zy30 Ex17.1-17.9)
    z_17 = sp.Symbol("z_17", real=True)
    # Ex 17.2: (0,1,1) x (1,2,1) = (-1,1,-1)
    n17_2 = sp.Matrix([0, 1, 1]).cross(sp.Matrix([1, 2, 1]))
    # Ex 17.3: (2, lam, -1-2*lam) . (1,1,-1) = 0 => lam = -1
    lam_17 = sp.Symbol("lam_17")
    sol17_3 = sp.solve(sp.Matrix([2, lam_17, -1 - 2 * lam_17]).dot(sp.Matrix([1, 1, -1])), lam_17)
    # Ex 17.4: tau1=(1,-2,1), tau2=(1,-1,0)x(0,2,1)=(-1,-1,2) => cos phi = 1/2
    t17_4a = sp.Matrix([1, -2, 1])
    t17_4b = sp.Matrix([1, -1, 0]).cross(sp.Matrix([0, 2, 1]))
    cos17_4 = sp.Abs(t17_4a.dot(t17_4b)) / (t17_4a.norm() * t17_4b.norm())
    # Ex 17.6: curve tangent at t=0 -> (1,2,3)
    tau17_6 = (
        (sp.exp(t) * sp.cos(t)).subs(t, 0),
        sp.diff(2 * sp.sin(t) + sp.cos(t), t).subs(t, 0),
        sp.diff(1 + sp.exp(3 * t), t).subs(t, 0),
    )
    # Ex 17.8: F = z - e^z + 2xy - 3 at (1,2,0) -> (4,2,0)
    F17_8 = z_17 - sp.exp(z_17) + 2 * x * y - 3
    n17_8 = tuple(sp.diff(F17_8, v).subs({x: 1, y: 2, z_17: 0}) for v in (x, y, z_17))
    # Ex 17.9: e^(2x-z) = f(pi*y - sqrt(2)*z) -> normal perp (1/2, sqrt(2)/pi, 1)
    f17_9 = sp.Function("f17_9")
    F17_9 = sp.exp(2 * x - z_17) - f17_9(sp.pi * y - sp.sqrt(2) * z_17)
    n17_9 = sp.Matrix([sp.diff(F17_9, x), sp.diff(F17_9, y), sp.diff(F17_9, z_17)])
    v17_9 = sp.Matrix([sp.Rational(1, 2), sp.sqrt(2) / sp.pi, 1])
    c17_geo_ok = (
        n17_2 == sp.Matrix([-1, 1, -1])
        and sol17_3 == [-1]
        and cos17_4 == sp.Rational(1, 2)
        and tau17_6 == (1, 2, 3)
        and F17_8.subs({x: 1, y: 2, z_17: 0}) == 0 and n17_8 == (4, 2, 0)
        and sp.simplify(n17_9.dot(v17_9)) == 0
    )
    check("m1_c17_geo_and_tangents", c17_geo_ok, "zy30 Ex17.1-17.9 (cross/bundle/angle pi/3/tangent/normal/cylinder) exact")

    # 93. Directional derivative, gradient, 2D Taylor, div/rot (zy30 Ex17.10-17.13 + zy18 Ex17.1-17.7)
    # Ex 17.10: f=x^2y+z^2 at (1,2,0) along (1,2,2)/3 -> 2
    f17_10 = x**2 * y + z_17**2
    g17_10 = sp.Matrix([sp.diff(f17_10, v).subs({x: 1, y: 2, z_17: 0}) for v in (x, y, z_17)])
    dd17_10 = g17_10.dot(sp.Matrix([1, 2, 2]) / 3)
    # Ex 17.11: (6a, 8b) = k(-3,-4), k>0, norm=10 => a=-1, b=-1
    # Ex 17.12: (zx+zy)/sqrt(2)=2*sqrt(2), -zy=-3 => zx=1, zy=3, max=sqrt(10)
    # Ex 17.13: rot(xy, -yz, zx) = (y, -z, -x) at (1,1,0) -> (1,0,-1)
    P17_13, Q17_13, R17_13 = x * y, -y * z_17, z_17 * x
    rot17_13 = sp.Matrix([
        sp.diff(R17_13, y) - sp.diff(Q17_13, z_17),
        sp.diff(P17_13, z_17) - sp.diff(R17_13, x),
        sp.diff(Q17_13, x) - sp.diff(P17_13, y),
    ])
    # zy18 Ex 17.2: u = x y^2 z^3 at (1,1,1) along (2,2,-1)/3 -> 1
    dd17_2i = sp.Matrix([1, 2, 3]).dot(sp.Matrix([2, 2, -1]) / 3)
    # zy18 Ex 17.3: (4a, 2b) = k(1,2), sqrt(5)*k = 4*sqrt(5) => k=4 => a=1, b=4
    # zy18 Ex 17.4: cos(x)/cos(y) 2nd-order Taylor -> 1 - x^2/2 + y^2/2
    f17_4i = sp.cos(x) / sp.cos(y)
    t17_4i_poly = (
        f17_4i.subs({x: 0, y: 0})
        + sp.Rational(1, 2) * (
            sp.diff(f17_4i, x, 2).subs({x: 0, y: 0}) * x**2
            + 2 * sp.diff(f17_4i, x, y).subs({x: 0, y: 0}) * x * y
            + sp.diff(f17_4i, y, 2).subs({x: 0, y: 0}) * y**2
        )
    )
    # zy18 Ex 17.5: (1) t=pi/2 -> pt (1,1,0), tau=(-1,1,-1); (2) (y^2,y,y^4) at y=1 -> tau=(2,1,4)
    c17_5i_x = sp.cos(t) + sp.sin(t)**2
    c17_5i_y = sp.sin(t) * (1 - sp.cos(t))
    c17_5i_z = sp.cos(t)
    tau17_5i = tuple(sp.diff(e, t).subs(t, sp.pi / 2) for e in (c17_5i_x, c17_5i_y, c17_5i_z))
    # zy18 Ex 17.7: v1=(0,x,z), v2=(y,0,1), F=v1 x v2 = (x, yz, -xy) => div F = 1+z
    F17_7i = sp.Matrix([0, x, z_17]).cross(sp.Matrix([y, 0, 1]))
    div17_7i = sp.diff(F17_7i[0], x) + sp.diff(F17_7i[1], y) + sp.diff(F17_7i[2], z_17)
    # Vector identities: rot(grad u) = 0, div(rot A) = 0
    u_vec = sp.sin(x * y) + z_17 * sp.exp(x)
    gu_vec = sp.Matrix([sp.diff(u_vec, x), sp.diff(u_vec, y), sp.diff(u_vec, z_17)])
    rot_grad = sp.Matrix([
        sp.diff(gu_vec[2], y) - sp.diff(gu_vec[1], z_17),
        sp.diff(gu_vec[0], z_17) - sp.diff(gu_vec[2], x),
        sp.diff(gu_vec[1], x) - sp.diff(gu_vec[0], y),
    ])
    div_rot17 = sp.diff(rot17_13[0], x) + sp.diff(rot17_13[1], y) + sp.diff(rot17_13[2], z_17)
    c17_field_ok = (
        dd17_10 == 2
        and sp.Matrix([6 * (-1), 8 * (-1)]).norm() == 10
        and sp.Matrix([1, 3]).norm() == sp.sqrt(10)
        and rot17_13.subs({x: 1, y: 1, z_17: 0}) == sp.Matrix([1, 0, -1])
        and dd17_2i == 1
        and sp.Matrix([4 * 1, 2 * 4]).norm() == 4 * sp.sqrt(5)
        and sp.expand(t17_4i_poly - (1 - x**2 / 2 + y**2 / 2)) == 0
        and tau17_5i == (-1, 1, -1)
        and F17_7i == sp.Matrix([x, y * z_17, -x * y]) and div17_7i == 1 + z_17
        and rot_grad == sp.Matrix([0, 0, 0]) and div_rot17 == 0
    )
    check("m1_c17_field_and_zy18", c17_field_ok, "zy30 Ex17.10-17.13 + zy18 Ex17.1-17.7 + rot(grad)=0, div(rot)=0 exact")

    # ==================== Calculus Lecture 18 (calc-18: 多元函数积分学，仅数一) ====================
    r_18, th_18, ph_18, h_18 = sp.symbols("r_18 th_18 ph_18 h_18", positive=True)
    # 94. Triple integrals (zy30 Ex18.1-18.7 + zy18 Ex18.1-18.11 + Errata #64 counterexample)
    v18_1 = 6 * sp.integrate(z_17 * sp.Rational(1, 2) * (1 - z_17)**2, (z_17, 0, 1))
    v18_2 = sp.integrate(sp.integrate(r_18**4 / 8, (r_18, 0, 2 * sp.sin(th_18))), (th_18, 0, sp.pi))
    v18_3 = sp.Rational(1, 3) * 4 * sp.pi * sp.integrate(r_18**4, (r_18, 0, 1))
    v18_4 = 2 * sp.pi * sp.integrate(sp.sin(ph_18) * sp.sec(ph_18)**2 / 2, (ph_18, 0, sp.pi / 4))
    v18_5 = sp.Rational(1, 2) * 4 * sp.pi * sp.integrate((1 - r_18**2) * r_18**2, (r_18, 0, 1))
    v18_7z = sp.integrate(sp.pi * z_17**2, (z_17, 0, 1)) / sp.integrate(sp.pi * z_17, (z_17, 0, 1))
    # zy18 Ex18.1, 18.4, 18.5, 18.6, 18.7, 18.8, 18.10, 18.11
    v18_1i = sp.pi / 2 * sp.integrate(x**2, (x, 0, 1)) * sp.integrate(sp.sin(sp.pi * y / 2), (y, 0, 1)) * sp.integrate(sp.cos(z_17), (z_17, 0, 1))
    v18_4i = 2 * sp.pi * sp.integrate(sp.sin(ph_18) * sp.cos(ph_18), (ph_18, 0, sp.pi / 4)) * sp.integrate(r_18**3, (r_18, 0, a_p))
    v18_5i_V = sp.integrate(sp.pi * (1 - z_17)**2, (z_17, 0, 1))
    v18_5i_My = sp.integrate(z_17 * sp.pi * (1 - z_17)**2, (z_17, 0, 1))
    v18_6i = 2 * sp.pi * sp.integrate(sp.integrate((r_18**2 + z_17) * r_18, (r_18, 0, sp.sqrt(2 * z_17))), (z_17, 0, 4))
    v18_7i = 2 * sp.pi * sp.integrate(sp.sin(ph_18), (ph_18, 0, sp.pi / 4)) * sp.integrate(r_18**3, (r_18, 0, a_p))
    v18_8i = 2 * sp.pi * sp.integrate((9 - r_18**2) * r_18, (r_18, 0, 3))
    v18_10i = 2 * sp.pi * sp.integrate(r_18 * (1 / (sp.sqrt(2) * r_18) - 1 / sp.sqrt(r_18**2 + h_18**2)), (r_18, 0, 1))
    v18_11i = sp.pi * sp.integrate(r_18**3 * (1 - r_18), (r_18, 0, 1)) + sp.integrate(sp.pi * y**4, (y, 0, 1))
    # Errata #64 counterexample: f=xy on unit ball => full integral = 0, octant integral > 0
    err64_full = sp.integrate(sp.cos(th_18) * sp.sin(th_18), (th_18, 0, 2 * sp.pi))
    err64_oct = sp.integrate(sp.cos(th_18) * sp.sin(th_18), (th_18, 0, sp.pi / 2))
    c18_triple_ok = (
        v18_1 == sp.Rational(1, 4) and v18_2 == sp.Rational(64, 75) and v18_3 == 4 * sp.pi / 15
        and sp.simplify(v18_4 - (sp.sqrt(2) - 1) * sp.pi) == 0 and v18_5 == 4 * sp.pi / 15
        and sp.Rational(4, 3) * sp.pi * 1 * 2 * 2 == 16 * sp.pi / 3 and v18_7z == sp.Rational(2, 3)
        and sp.simplify(v18_1i - sp.sin(1) / 3) == 0
        and sp.simplify(v18_4i - sp.pi * a_p**4 / 8) == 0
        and v18_5i_V == sp.pi / 3 and v18_5i_My / v18_5i_V == sp.Rational(1, 4)
        and v18_6i == 256 * sp.pi / 3
        and sp.simplify(v18_7i - (2 - sp.sqrt(2)) * sp.pi * a_p**4 / 4) == 0
        and v18_8i == 81 * sp.pi / 2
        and sp.simplify(v18_10i - 2 * sp.pi * (1 / sp.sqrt(2) + h_18 - sp.sqrt(h_18**2 + 1))) == 0
        and v18_11i == sp.pi / 4
        and err64_full == 0 and err64_oct == sp.Rational(1, 2)
    )
    check("m1_c18_triple_and_err64", c18_triple_ok, "zy30 Ex18.1-18.7 + zy18 Ex18.1-18.11 + Errata #64 counterexample exact")

    # 95. Type-I Line & Surface integrals (zy30 Ex18.8-18.14 + zy18 Ex18.12-18.15 + Errata #62)
    v18_8 = sp.integrate(x * sp.sqrt(1 + 4 * x**2), (x, 0, sp.sqrt(2)))
    v18_9 = sp.integrate((-2 * sp.sin(th_18)) * 2, (th_18, -sp.pi, 0))
    # Ex 18.11: x=1+sqrt(2)*cos t, y=1+sqrt(2)*sin t, ds=sqrt(2) dt
    v18_11 = sp.integrate((2 * (1 + sp.sqrt(2) * sp.cos(t))**2 + 3 * (1 + sp.sqrt(2) * sp.sin(t))**2) * sp.sqrt(2), (t, 0, 2 * sp.pi))
    v18_12 = sp.integrate(z_17, (z_17, 0, 1)) * sp.integrate(a_p / sp.sqrt(a_p**2 - x**2), (x, 0, a_p))
    # Ex 18.13 (Errata #62): 1/rho = sqrt(1 - (x^2+y^2)/4), dS = sqrt(4-x^2-y^2)/(2z) dxdy
    v18_13 = sp.Rational(1, 4) * 2 * sp.pi * sp.integrate((4 - r_18**2) * r_18, (r_18, 0, sp.sqrt(2)))
    v18_14 = 18 * sp.integrate(sp.integrate(r_18**2, (r_18, 0, 2 * sp.cos(th_18))), (th_18, -sp.pi / 2, sp.pi / 2))
    v18_13i = (sp.Rational(1, 2) + sp.Rational(1, 3) + sp.Rational(1, 4)) / 3 * a_p**2 * 4 * sp.pi * a_p**2
    # zy18 Ex18.15 numerical area check S2 > S1 > S3
    S1_15 = float(mp.quad(lambda rr: rr * mp.quad(lambda tt: mp.sqrt(1 + rr**4 * (mp.cos(tt)**4 + mp.sin(tt)**4)), [0, 2 * mp.pi]), [0, 1]))
    S2_15 = float(2 * mp.pi * mp.quad(lambda rr: rr * mp.sqrt(1 + rr**2), [0, 1]))
    S3_15 = float(mp.quad(lambda rr: rr * mp.quad(lambda tt: mp.sqrt(1 + rr**6 * (mp.cos(tt)**2 * mp.sin(tt)**4 + mp.cos(tt)**4 * mp.sin(tt)**2)), [0, 2 * mp.pi]), [0, 1]))
    c18_t1_ok = (
        v18_8 == sp.Rational(13, 6) and v18_9 == 8
        and sp.simplify(v18_11 - 20 * sp.sqrt(2) * sp.pi) == 0
        and sp.simplify(v18_12 - sp.pi * a_p / 4) == 0
        and v18_13 == 3 * sp.pi / 2 and v18_14 == 64
        and sp.simplify(v18_13i - 13 * sp.pi * a_p**4 / 9) == 0
        and S2_15 > S1_15 > S3_15
    )
    check("m1_c18_type1_line_surf", c18_t1_ok, f"zy30 Ex18.8-18.14 + zy18 Ex18.12-18.15 (S2={S2_15:.3f}>S1={S1_15:.3f}>S3={S3_15:.3f}) exact")

    # 96. Type-II Line integrals, Green, Path Independence, Digging Hole (zy30 Ex18.15-18.21 + zy18 Ex18.16-18.23, 18.37)
    v18_15 = -6 * sp.integrate(sp.sin(t)**4 * sp.cos(t)**2, (t, 0, sp.pi / 2))
    a_18 = sp.Symbol("a_18", positive=True)
    I18_16 = sp.integrate(1 + a_18**3 * sp.sin(x)**3 + (2 * x + a_18 * sp.sin(x)) * a_18 * sp.cos(x), (x, 0, sp.pi))
    # Ex 18.19 curl check
    P18_19 = (4 * x - y) / (4 * x**2 + y**2)
    Q18_19 = (x + y) / (4 * x**2 + y**2)
    curl18_19 = sp.simplify(sp.diff(Q18_19, x) - sp.diff(P18_19, y))
    # zy18 Ex 18.19 curl check
    u18_19i = x**2 + 4 * y**2
    P18_19i = (x * sp.exp(u18_19i) + y) / u18_19i
    Q18_19i = (4 * y * sp.exp(u18_19i) - x) / u18_19i
    curl18_19i = sp.simplify(sp.diff(Q18_19i, x) - sp.diff(P18_19i, y))
    # zy18 Ex 18.20 curl & semicircle integral
    P18_20i = (x - y) / (x**2 + y**2)
    Q18_20i = (x + y) / (x**2 + y**2)
    curl18_20i = sp.simplify(sp.diff(Q18_20i, x) - sp.diff(P18_20i, y))
    # zy18 Ex 18.23
    v18_23i = sp.integrate(2 * (1 - sp.sin(t)), (t, 0, sp.pi))
    # zy18 Ex 18.37: ellipse half area * 6 + int_{-1/2}^{1/2} (-3x^2) dx = sqrt(3)*pi - 1/4
    v18_37i = 6 * (sp.pi / (2 * sp.sqrt(3))) + sp.integrate(-3 * x**2, (x, -sp.Rational(1, 2), sp.Rational(1, 2)))
    c18_green_ok = (
        v18_15 == -3 * sp.pi / 16
        and sp.expand(I18_16 - (sp.pi - 4 * a_18 + sp.Rational(4, 3) * a_18**3)) == 0
        and curl18_19 == 0 and curl18_19i == 0 and curl18_20i == 0
        and sp.integrate(-2 * y, (y, 2, 0)) == 4
        and sp.integrate(2 * x**3, (x, 0, 1)) == sp.Rational(1, 2)
        and v18_23i == 2 * sp.pi - 4
        and sp.simplify(v18_37i - (sp.sqrt(3) * sp.pi - sp.Rational(1, 4))) == 0
    )
    check("m1_c18_green_and_path", c18_green_ok, "zy30 Ex18.15-18.21 + zy18 Ex18.16-18.23, 18.37 (curls=0, 2pi-4, sqrt(3)pi-1/4) exact")

    # 97. Type-II Surface, Gauss, Stokes & Space Line integrals (zy30 Ex18.22-18.30 + zy18 Ex18.24-18.36)
    v18_22 = -2 * sp.pi * sp.E + 4 * sp.pi * sp.E**2 - 2 * sp.pi * (sp.E**2 - sp.E)
    v18_22_gauss = 2 * sp.pi * sp.integrate(z_17 * sp.exp(z_17), (z_17, 1, 2))
    v18_25_gauss = -2 * sp.pi * sp.integrate(r_18 * (1 - r_18**2) * (3 * r_18**2 + 7), (r_18, 0, 1))
    v18_26 = sp.pi + sp.Rational(1, 2) * 2 * sp.pi * sp.integrate(r_18**3, (r_18, 0, 1))
    v18_27 = -2 * sp.pi - 2 * 2 * sp.pi * sp.integrate(r_18 * sp.integrate(z_17, (z_17, -sp.sqrt(1 - r_18**2), 0)), (r_18, 0, 1)) + sp.pi
    # Ex 18.28 point source div = 0
    r3_norm = (x**2 + y**2 + z_17**2)**sp.Rational(3, 2)
    div18_28 = sp.simplify(sp.diff(x / r3_norm, x) + sp.diff(y / r3_norm, y) + sp.diff(z_17 / r3_norm, z_17))
    v18_29 = 4 * sp.integrate(sp.cos(th_18) * sp.sin(th_18), (th_18, 0, sp.pi / 2)) * sp.integrate(r_18**3 * (1 - r_18**4), (r_18, 0, 1))
    v18_30 = sp.integrate(sp.sqrt(2) * sp.sin(t)**2, (t, -sp.pi / 2, sp.pi / 2))
    # zy18 Ex 18.24 (Stokes & Parametric)
    v18_24i = sp.integrate(sp.sin(t)**2 * sp.cos(t)**2 / (2 * sp.sqrt(2)), (t, 0, 2 * sp.pi))
    # zy18 Ex 18.25 (rot = 0, integral = 8pi^3/3)
    P25, Q25, R25 = x**2 - y * z_17, y**2 - z_17 * x, z_17**2 - x * y
    rot18_25i = (
        sp.diff(R25, y) - sp.diff(Q25, z_17),
        sp.diff(P25, z_17) - sp.diff(R25, x),
        sp.diff(Q25, x) - sp.diff(P25, y),
    )
    # zy18 Ex 18.30, 18.31, 18.35, 18.36
    v18_30i = sp.integrate(sp.integrate(y**2 / x, (y, -x, x)), (x, 0, 1))
    J18_31i = sp.Matrix([[1, -1, 1], [1, 1, -1], [-1, 1, 1]]).det()
    f18_35i = sp.exp(xp) * (sp.exp(xp) - 1) / xp
    ode18_35i = sp.simplify(xp * sp.diff(f18_35i, xp) + f18_35i - xp * f18_35i - sp.exp(2 * xp))
    lim18_35i = sp.limit(f18_35i, xp, 0, "+")
    v18_36i = 2 * sp.pi * sp.integrate(r_18**2, (r_18, 1, 2))
    c18_gs_ok = (
        sp.simplify(v18_22 - 2 * sp.pi * sp.E**2) == 0 and sp.simplify(v18_22_gauss - 2 * sp.pi * sp.E**2) == 0
        and v18_25_gauss == -4 * sp.pi and v18_26 == 5 * sp.pi / 4 and v18_27 == -sp.pi / 2
        and div18_28 == 0 and v18_29 == sp.Rational(1, 4)
        and sp.simplify(v18_30 - sp.sqrt(2) * sp.pi / 2) == 0
        and sp.simplify(v18_24i - sp.sqrt(2) * sp.pi / 16) == 0
        and rot18_25i == (0, 0, 0) and sp.integrate(z_17**2, (z_17, 0, 2 * sp.pi)) == 8 * sp.pi**3 / 3
        and v18_30i == sp.Rational(2, 9) and J18_31i == 4 and 3 * sp.Rational(1, 4) * sp.Rational(4, 3) == 1
        and ode18_35i == 0 and lim18_35i == 1 and v18_36i == 14 * sp.pi / 3
    )
    check("m1_c18_gauss_stokes_all", c18_gs_ok, "zy30 Ex18.22-18.30 + zy18 Ex18.24-18.36 (Gauss/Stokes/div=0/ODE/14pi/3) exact")

    # ==================== Linear Algebra Batch 1 (la-01 ~ la-03) ====================
    a_l, b_l, c_l, d_l, k_l, lam_l = sp.symbols("a_l b_l c_l d_l k_l lam_l")

    # 98. la-01 core formulas: Laplace sign (-1)^(mn), Vandermonde, row-sum, X-shape, tridiagonal
    A_22 = sp.Matrix([[1, 2], [3, 5]])
    B_33 = sp.Matrix([[2, 1, 0], [0, 3, 1], [1, 0, 2]])
    M_lap = sp.BlockMatrix([[sp.ZeroMatrix(2, 3), A_22], [B_33, sp.ZeroMatrix(3, 2)]]).as_explicit()
    V4 = sp.Matrix([[1, 1, 1, 1], [a_l, b_l, c_l, d_l], [a_l**2, b_l**2, c_l**2, d_l**2], [a_l**3, b_l**3, c_l**3, d_l**3]])
    RS4 = sp.Matrix([[a_l if i == j else b_l for j in range(4)] for i in range(4)])
    T4_rep = sp.Matrix([[2 * a_l if i == j else (1 if j == i + 1 else (a_l**2 if i == j + 1 else 0)) for j in range(4)] for i in range(4)])
    la01_f_ok = (
        M_lap.det() == (-1)**(2 * 3) * A_22.det() * B_33.det()
        and sp.expand(V4.det() - (b_l - a_l) * (c_l - a_l) * (d_l - a_l) * (c_l - b_l) * (d_l - b_l) * (d_l - c_l)) == 0
        and sp.expand(RS4.det() - (a_l + 3 * b_l) * (a_l - b_l)**3) == 0
        and sp.expand(T4_rep.det() - 5 * a_l**4) == 0
    )
    check("m1_la01_core_formulas", la01_f_ok, "la01 Laplace (-1)^(mn), Vandermonde, row-sum, tridiagonal (n+1)a^n exact")

    # 99. la-01 Base 30 Lectures examples (Ex 1.1-1.11, 1.13, Hw 1.3/1.5/1.6)
    D11 = sp.Matrix([[2, -1, 0, 0], [0, 2, -1, 0], [0, 0, 2, -1], [-1, 0, 0, 2]]).det()
    D12 = sp.Matrix([[1, 1, 1, 1], [1, 2, 0, 0], [1, 0, 3, 0], [1, 0, 0, 4]]).det()
    D13 = sp.Matrix([[1, -1, 1, x - 1], [1, -1, x + 1, -1], [1, x - 1, 1, -1], [x + 1, -1, 1, -1]]).det()
    a1, a2, a3, a4, b1, b2, b3, b4 = sp.symbols("a1 a2 a3 a4 b1 b2 b3 b4")
    D15 = sp.Matrix([[a1, 0, 0, b1], [0, a2, b2, 0], [0, b3, a3, 0], [b4, 0, 0, a4]]).det()
    D16 = sp.Matrix([[a_l, b_l, c_l], [a_l**2, b_l**2, c_l**2], [b_l + c_l, a_l + c_l, a_l + b_l]]).det()
    D17 = sp.Matrix([[1 - a_l, a_l, 0, 0], [-1, 1 - a_l, a_l, 0], [0, -1, 1 - a_l, a_l], [0, 0, -1, 1 - a_l]]).det()
    f18 = sp.Matrix([[1, 0, x], [1, 2, x**2], [1, 3, x**3]]).det()
    D19 = sp.Matrix([[lam_l - 1, -2, 3], [1, lam_l - 4, 3], [-1, a_l, lam_l - 5]]).det()
    C111 = sp.Matrix([[0, 2, 1], [-1, 2, 2], [2, -5, -2]]).det()
    D113 = sp.Matrix([[1, 2, -2], [2, -1, lam_l], [3, 1, -1]]).det()
    P_hw13 = sp.Poly(sp.Matrix([[x, x, 1, 2 * x], [1, x, 2, -1], [2, 1, x, 1], [2, -1, 1, x]]).det(), x)
    D_hw15 = sp.Matrix([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]]).det()
    D_hw16 = sp.Matrix([[1, a_l, 0, 0], [-1, 2 - a_l, b_l, 0], [0, -2, 3 - b_l, c_l], [0, 0, -3, 4 - c_l]]).det()
    la01_base_ok = (
        D11 == 15 and D12 == -2 and sp.expand(D13 - x**4) == 0
        and sp.expand(D15 - (a1 * a4 - b1 * b4) * (a2 * a3 - b2 * b3)) == 0
        and sp.expand(D16 - (a_l + b_l + c_l) * (b_l - a_l) * (c_l - b_l) * (c_l - a_l)) == 0
        and sp.expand(D17 - (a_l**4 - a_l**3 + a_l**2 - a_l + 1)) == 0
        and sp.expand(f18.subs(x, x + 1) - f18 - 6 * x**2) == 0
        and sp.expand(D19 - (lam_l - 2) * (lam_l**2 - 8 * lam_l + 18 - 3 * a_l)) == 0
        and 2 * C111 == 10 and sp.expand(D113 - 5 * (lam_l - 1)) == 0
        and P_hw13.nth(3) == -5 and D_hw15 == 160 and sp.simplify(D_hw16) == 24
    )
    check("m1_la01_base_examples", la01_base_ok, "la01 Base30 Ex1.1-1.11, 1.13, Hw1.3(-5)/1.5(160)/1.6(24) all exact")

    # 100. la-01 Intensive 9 Lectures examples (Ex 1.1-1.8)
    f_i11 = sp.Matrix([[2 * x + 1, 3, 2 * x + 1, 1], [2 * x, -3, 4 * x, -2], [2 * x + 1, 2, 2 * x + 1, 1], [2 * x, -4, 4 * x, -2]]).det()
    g_i11 = sp.Matrix([[2 * x + 1, 1, 2 * x + 1, 3], [5 * x + 1, -2, 4 * x, -3], [0, 1, 2 * x + 1, 2], [2 * x, -2, 4 * x, -4]]).det()
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    al_3 = sp.Matrix([x1, x2, x3])
    D_i13 = (al_3 * al_3.T + sp.eye(3)).det()
    D_i14 = sp.Matrix([[b_l, -1, 0, 0], [0, b_l, -1, 0], [0, 0, b_l, -1], [a4, a3, a2, b_l + a1]]).det()
    f_i17 = lambda lam: lam**3 + 2 * lam**2 - lam - 3
    prod_i17 = f_i17(2) * f_i17(-2) * f_i17(-1) * f_i17(sp.Rational(1, 2))
    la01_adv_ok = (
        sp.simplify(f_i11) == 0 and sp.expand(g_i11 + x * (8 * x + 2)) == 0
        and sp.expand(D_i13 - (1 + x1**2 + x2**2 + x3**2)) == 0
        and sp.expand(D_i14 - (b_l**4 + a1 * b_l**3 + a2 * b_l**2 + a3 * b_l + a4)) == 0
        and prod_i17 == sp.Rational(-253, 8)
    )
    check("m1_la01_adv_examples", la01_adv_ok, "la01 Adv9 Ex1.1(2 roots), 1.3(1+sum x_i^2), 1.4, 1.7(-253/8) all exact")

    # 101. la-01 MZL examples + Ex9 non-commuting erratum proof
    D_mzl3 = sp.Matrix([[a_l, 0, -1, 1], [0, a_l, 1, -1], [-1, 1, a_l, 0], [1, -1, 0, a_l]]).det()
    D_mzl6_4 = sp.Matrix([[i + 1 + (a_l if i == j else 0) for j in range(4)] for i in range(4)]).det()
    D_mzl7_3 = (lam_l * sp.eye(3) - al_3 * al_3.T).det()
    A_nc = sp.Matrix([[1, 2, 0], [0, 1, 1], [1, 0, 1]])
    B_nc = sp.Matrix([[0, 1, 1], [1, 0, 2], [2, 1, 0]])
    M_nc = sp.BlockMatrix([[A_nc, B_nc], [B_nc, A_nc]]).as_explicit()
    D_mzl10_5 = sp.Matrix([[2 if i == j or j == 4 else (-1 if i == j + 1 else 0) for j in range(5)] for i in range(5)]).det()
    C_mzl15 = sp.Matrix([[1, 0, 1], [-1, 1, 0], [0, -1, 1]])
    la01_mzl_ok = (
        sp.expand(D_mzl3 - a_l**2 * (a_l**2 - 4)) == 0
        and sp.expand(D_mzl6_4 - a_l**3 * (a_l + 10)) == 0
        and sp.expand(D_mzl7_3 - lam_l**2 * (lam_l - x1**2 - x2**2 - x3**2)) == 0
        and M_nc.det() == (A_nc + B_nc).det() * (A_nc - B_nc).det()
        and M_nc.det() != (A_nc**2 - B_nc**2).det()
        and D_mzl10_5 == 2**6 - 2
        and (C_mzl15 + 2 * sp.eye(3)).det() == 28
    )
    check("m1_la01_mzl_examples", la01_mzl_ok, "la01 MZL Ex3(a^4-4a^2), Ex6/7, Ex9 non-commuting erratum, Ex10(2^(n+1)-2), Ex15(28) exact")

    # 102. la-02 all examples (Base 1.12, Hw 1.2, Base 2.8/2.10, Adv 2.1-2.4, MZL 1.3 Ex 1-3)
    A_112 = sp.Matrix([[2, -1, 2, 3], [0, 1, -1, 0], [2, 3, 4, 5], [1, 1, 1, 1]])
    v_112 = A_112.cofactor(2, 0) + A_112.cofactor(2, 1) + A_112.cofactor(2, 2) + A_112.minor(2, 3)
    A_hw12 = sp.Matrix([[1, -1, 0, 1], [-2, 1, -1, 1], [3, -2, 2, -1], [0, 0, 3, 4]])
    v_hw12 = A_hw12.cofactor(0, 0) - A_hw12.cofactor(0, 1)
    A_28 = sp.Matrix([[1, 0, 1], [2, 1, 0], [-3, 2, -5]])
    A_i21 = sp.Matrix([[2, 1, 0, -1], [-1, 2, -5, 3], [3, 0, a_l, b_l], [1, -3, 5, 0]])
    eq_i21 = sp.expand(A_i21.det() - (A_i21.cofactor(3, 0) - A_i21.cofactor(3, 1) + A_i21.cofactor(3, 2) + 10))
    A_i22 = sp.Matrix([[0, 0, 0, 5, 6], [0, 0, 0, 7, 8], [1, 2, 3, 0, 0], [0, 1, 4, 0, 0], [0, 0, 1, 0, 0]])
    sum_i22 = sum(A_i22.adjugate())
    A_mzl131 = sp.Matrix([[a_l + 1, b_l, 3], [a_l, b_l / 2, 1], [1, 1, 2]])
    sol_mzl131 = sp.solve([A_mzl131.det() + sp.Rational(1, 2), -A_mzl131.minor(1, 0) + A_mzl131.minor(1, 1) + A_mzl131.cofactor(1, 2)], [a_l, b_l])
    la02_ok = (
        v_112 == 6 and v_hw12 == -4
        and A_28.inv() == sp.Matrix([[sp.Rational(-5, 2), 1, sp.Rational(-1, 2)], [5, -1, 1], [sp.Rational(7, 2), -1, sp.Rational(1, 2)]])
        and eq_i21 == 10 * (a_l - 4)
        and sum_i22 == -4
        and (-1) * 2 + 2 * 3 + (-1) * 3 == 1
        and set(sol_mzl131) == {(0, 1), (sp.Rational(3, 2), sp.Rational(5, 2))}
    )
    check("m1_la02_all", la02_ok, "la02 Base1.12(6)/Hw1.2(-4)/2.8, Adv2.1(a=4)/2.2(-4)/2.3(1)/2.4(6), MZL1.3 Ex1-3 all exact")

    # 103. la-03 Base 30 Lectures examples (Ex 2.2, 2.3, 2.9, 2.12, 2.14, 2.17, 2.18, Hw 2.15)
    A_22b = sp.Matrix([[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]])
    A_23b = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    A_212 = sp.Matrix([[0, 2, -1], [1, 1, 2], [-1, -1, -1]])
    E12_3 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    E23_3 = sp.Matrix([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    Astar_217 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, -3, 0, 8]])
    B_217 = 6 * (2 * sp.eye(4) - Astar_217).inv()
    k1_l, k2_l, k3_l = sp.symbols("k1_l k2_l k3_l")
    A_218 = sp.Matrix([[1, 0, 1], [-1, -1, 1], [0, 2, -4]])
    B_218 = sp.Matrix([[1, 0, 1], [0, -1, 2], [0, 0, 0]])
    P_218 = sp.Matrix([[2 * k1_l + 1, 2 * k1_l, k1_l], [2 * k2_l + 1, 2 * k2_l + 1, k2_l], [2 * k3_l, 2 * k3_l, k3_l]])
    B_hw215 = sp.Matrix([[0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 0, 0]])
    C_hw215 = sp.Matrix([[2, 1, 3, 4], [0, 2, 1, 3], [0, 0, 2, 1], [0, 0, 0, 2]])
    A_hw215 = ((C_hw215 - B_hw215 - sp.eye(4)).T).inv()
    la03_base_ok = (
        A_22b**9 == 256 * A_22b
        and A_23b**5 == sp.Matrix([[1, 5, 10], [0, 1, 5], [0, 0, 1]])
        and A_212.inv() == sp.Matrix([[sp.Rational(-1, 2), sp.Rational(-3, 2), sp.Rational(-5, 2)], [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1, 1]])
        and E12_3 * E23_3 == sp.Matrix([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
        and B_217 == sp.Matrix([[6, 0, 0, 0], [0, 6, 0, 0], [6, 0, 6, 0], [0, 3, 0, -1]])
        and P_218 * A_218 == B_218 and sp.simplify(P_218.det() - k3_l) == 0
        and A_hw215 == sp.Matrix([[1, 0, 0, 0], [-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1]])
    )
    check("m1_la03_base_examples", la03_base_ok, "la03 Base30 Ex2.2/2.3/2.12/2.14/2.17/2.18/Hw2.15 all exact")

    # 104. la-03 Intensive 9 Lectures Part 1 (Ex 3.2-3.11 + Ex 3.8(4) perturbation fix check)
    A_32 = sp.Matrix([[2, 6, -4], [-1, -3, 2], [3, 9, -6]])
    A_33 = sp.Matrix([[1, a_l, b_l], [0, 1, a_l], [0, 0, 1]])
    A_34 = sp.Matrix([[sp.Rational(1, 2), -sp.sqrt(3) / 2], [sp.sqrt(3) / 2, sp.Rational(1, 2)]])
    A_35 = sp.Matrix([[1, 1, -1], [0, 1, 1], [0, 0, 1]])
    M_36 = sp.Matrix([[1, 0], [-1, 1]])**3 * sp.Matrix([[1, 2], [-1, 3]]) * sp.Matrix([[0, 1], [1, 0]])**(-5)
    B_37 = sp.Matrix([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    C_37 = sp.Matrix([[0, 1, 1], [0, 0, 1], [0, 0, 1]])
    A_37 = B_37 * (B_37 - C_37) * B_37.inv()
    # Ex 3.8(4) exact congruence identity Q^T A* Q = B* with Q = |C| * (C^-1)^T
    A_38 = sp.Matrix([[1, 2], [0, 0]])  # singular!
    C_38 = sp.Matrix([[2, 1], [1, 3]])
    B_38 = C_38.T * A_38 * C_38
    Q_38 = C_38.det() * (C_38.inv()).T
    # Counterexample proving A+tE and B+tE are NOT congruent for skew-symmetric A
    A_skew = sp.Matrix([[0, 1], [-1, 0]])
    C_diag = sp.Matrix([[2, 0], [0, 1]])
    B_skew = C_diag.T * A_skew * C_diag
    la03_adv1_ok = (
        A_32**10 == (-7)**9 * A_32
        and A_33**4 == sp.Matrix([[1, 4 * a_l, 6 * a_l**2 + 4 * b_l], [0, 1, 4 * a_l], [0, 0, 1]])
        and sp.simplify(A_34**(-11) - A_34) == sp.zeros(2, 2)
        and A_35**10 == sp.Matrix([[1, 10, 35], [0, 1, 10], [0, 0, 1]])
        and M_36 == sp.Matrix([[2, 1], [-3, -4]])
        and A_37**99 == sp.Matrix([[1, 0, -1], [0, 1, -1], [0, 0, 0]])
        and Q_38.T * A_38.adjugate() * Q_38 == B_38.adjugate()
        and (A_skew + sp.eye(2)).det() != (B_skew + sp.eye(2)).det()
    )
    check("m1_la03_adv_part1", la03_adv1_ok, "la03 Adv9 Ex3.2-3.7, Ex3.8(4) exact Q^TA*Q=B* & P.24 B+tE flaw confirmed")

    # 105. la-03 Intensive 9 Lectures Part 2 (Ex 3.12-3.20)
    # Ex 3.13: n=4 all-1 minus E
    A_313 = sp.ones(4, 4) - sp.eye(4)
    A_313_inv = sp.Rational(1, 3) * sp.ones(4, 4) - sp.eye(4)
    # Ex 3.16
    P_316 = sp.Matrix([[1, 1, 3], [1, 2, 2], [0, 0, 2]])
    B_316 = sp.diag(-1, -2, 0)
    A_316 = P_316 * B_316 * P_316.inv()
    A_316_99 = P_316 * (B_316**99) * P_316.inv()
    # Ex 3.17
    A_317 = sp.Matrix([[1, 1, 1, 1], [1, 1, 1, 1], [2, 1, 0, -1], [-1, 0, 1, 2]])
    B_317 = sp.Matrix([[1, 1], [1, 1], [0, 1], [1, 0]])
    C_317 = sp.Matrix([[-1, 0, 1, 2], [2, 1, 0, -1]])
    # Ex 3.19
    A_319 = sp.Matrix([[1, 2, 2], [1, 3, 0], [2, 7, -2]])
    B_319 = sp.Matrix([[1, 2, 2], [0, 1, 1], [-1, 1, 1]])
    P_319 = sp.Matrix([[3 - 6 * k1_l, 4 - 6 * k2_l, 4 - 6 * k3_l], [-1 + 2 * k1_l, -1 + 2 * k2_l, -1 + 2 * k3_l], [k1_l, k2_l, k3_l]])
    # Ex 3.20
    A_320 = sp.Matrix([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
    P_320 = sp.Matrix([[1, 0, 0], [0, 1, sp.Rational(-2, 3)], [0, 0, sp.Rational(1, 3)]])
    J_320 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    la03_adv2_ok = (
        A_313 * A_313_inv == sp.eye(4)
        and A_316_99 == sp.Matrix([[2**99 - 2, 1 - 2**99, 2 - 2**98], [2**100 - 2, 1 - 2**100, 2 - 2**99], [0, 0, 0]])
        and B_317 * C_317 == A_317
        and A_317**10 == sp.Matrix([[3**9, 3**9, 3**9, 3**9], [3**9, 3**9, 3**9, 3**9], [3**9 + 1, 3**9, 3**9 - 1, 3**9 - 2], [-1, 0, 1, 2]])
        and A_319 * P_319 == B_319 and sp.simplify(P_319.det() - (k3_l - k2_l)) == 0
        and P_320.inv() * A_320 * P_320 == J_320 and A_320**2 != sp.zeros(3, 3) and A_320**3 == sp.zeros(3, 3)
    )
    check("m1_la03_adv_part2", la03_adv2_ok, "la03 Adv9 Ex3.13/3.16/3.17(full-rank A^10)/3.19/3.20(nilpotent) all exact")

    # 106. la-03 MZL exclusive real-exam gems (Cayley, 2022 M1, 2024 M2, 2023 M1 block ranks)
    S_cay = sp.Matrix([[0, 2, -1], [-2, 0, 3], [1, -3, 0]])
    Q_cay = (sp.eye(3) - S_cay) * (sp.eye(3) + S_cay).inv()
    P_m24 = sp.Matrix([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    RHS_m24 = sp.Matrix([[a_l + 2 * c_l, 0, c_l], [0, b_l, 0], [2 * c_l, 0, c_l]])
    A_m24 = (P_m24.T).inv() * RHS_m24 * (P_m24**2).inv()
    # 2023 M1 test on concrete ABC=O
    A_23 = sp.Matrix([[1, 0], [0, 0]])
    B_23 = sp.Matrix([[0, 1], [0, 0]])
    C_23 = sp.Matrix([[1, 0], [0, 0]])  # AB=[[0,1],[0,0]], ABC=O, (AB)^2=O -> r1=2, r3=2, r2=3 -> r1<=r3<=r2
    r1_23 = sp.BlockMatrix([[sp.ZeroMatrix(2, 2), A_23], [B_23 * C_23, sp.eye(2)]]).as_explicit().rank()
    r2_23 = sp.BlockMatrix([[A_23 * B_23, C_23], [sp.ZeroMatrix(2, 2), sp.eye(2)]]).as_explicit().rank()
    r3_23 = sp.BlockMatrix([[sp.eye(2), A_23 * B_23], [A_23 * B_23, sp.ZeroMatrix(2, 2)]]).as_explicit().rank()
    la03_mzl_ok = (
        Q_cay.T * Q_cay == sp.eye(3)
        and A_m24 == sp.diag(a_l, b_l, c_l)
        and A_23 * B_23 * C_23 == sp.zeros(2, 2)
        and r1_23 == 2 and r3_23 == 2 and r2_23 == 3 and r1_23 <= r3_23 <= r2_23
    )
    check("m1_la03_mzl_examples", la03_mzl_ok, "la03 MZL Cayley orthogonal, 2024 M2 diag(a,b,c), 2023 M1 r1<=r3<=r2 exact")

    # ==================== Linear Algebra Batch 2 (la-04 ~ la-06) ====================
    # 107. la-04 Core rank identities (full-rank invariance, Sylvester, Frobenius, idempotent, rank-1, Gram)
    A_fr = sp.Matrix([[1, 0], [0, 1], [2, 3]])  # 3x2 col-full-rank
    B_fr = sp.Matrix([[1, 2, -1], [0, 3, 4]])   # 2x3 row-full-rank
    X_t = sp.Matrix([[1, 2, 3], [2, 4, 6]])
    A_idem = sp.Matrix([[1, 2], [0, 0]])  # A^2 = A
    A_r1 = sp.Matrix([1, -2, 3]) * sp.Matrix([[2, 1, -1]])
    A_nil2 = sp.Matrix([[0, 1], [0, 0]])
    la04_id_ok = (
        (A_fr * X_t).rank() == X_t.rank()
        and (A_fr * B_fr).rank() == 2
        and (A_fr.T * A_fr).det() != 0 and (B_fr * B_fr.T).det() != 0
        and A_idem**2 == A_idem and A_idem.rank() + (A_idem - sp.eye(2)).rank() == 2
        and A_idem.rank() == A_idem.trace()
        and (A_idem + sp.eye(2))**5 == sp.eye(2) + (2**5 - 1) * A_idem
        and A_r1.rank() == 1 and A_r1**4 == (A_r1.trace())**3 * A_r1
        and (sp.eye(3) + A_r1).det() == 1 + A_r1.trace()
        and A_nil2.rank() == 1 and (A_nil2**2).rank() == 0
        and (A_fr.T * A_fr).rank() == A_fr.rank() == (A_fr * A_fr.T).rank()
    )
    check("m1_la04_rank_identities", la04_id_ok, "la04 full-rank, Gram, idempotent, rank-1, nilpotent counterexample exact")

    # 108. la-04 Base 30 Lectures examples (Ex 2.19, 2.20, Hw 2.5)
    t_s = sp.Symbol("t_s")
    M_219 = sp.Matrix([[1, 2, -1, 1], [2, 0, t_s, 0], [0, -4, 5, -2]])
    A_hw25 = sp.Matrix([[a_l, b_l, b_l], [b_l, a_l, b_l], [b_l, b_l, a_l]])
    la04_base_ok = (
        M_219[:, :3].det() == 4 * t_s - 12
        and M_219.subs(t_s, 3).rank() == 2
        and M_219.subs(t_s, 0).rank() == 3
        and sp.factor(A_hw25.det()) == (a_l - b_l)**2 * (a_l + 2 * b_l)
        and A_hw25.subs({a_l: -2, b_l: 1}).adjugate().rank() == 1
        and A_hw25.subs({a_l: 1, b_l: 1}).adjugate().rank() == 0
    )
    check("m1_la04_base_examples", la04_base_ok, "la04 Base Ex2.19(t=3), Hw2.5(a=-2b, excl a=b) exact")

    # 109. la-04 Adv 9 Lectures Part 1 (Ex 4.1-4.4)
    AB_43 = sp.Matrix([[8, 0, -4], [sp.Rational(-3, 2), 9, -6], [-2, 0, 1]])
    la04_adv1_ok = (
        AB_43.rank() == 2
        and AB_43**2 == 9 * AB_43
        and (A_fr * sp.Matrix([[1, 2], [2, 4]])).rank() == 1
    )
    check("m1_la04_adv_part1", la04_adv1_ok, "la04 Adv9 Ex4.1-4.4 (Ex4.3 (AB)^2=9AB => BA=9E2) exact")

    # 110. la-04 Adv 9 Lectures Part 2 (Ex 4.5-4.9)
    # Ex 4.5 concrete test: A(1x3) r=1, C=E3, B such that A(E3+BA)=O -> B = -A^T / (A A^T)
    A_45 = sp.Matrix([[1, 2, 2]])
    B_45 = -A_45.T / 9
    # Ex 4.6 counterexample for (3)(4)
    A_46, B_46, C_46 = sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([[0, 0], [0, 1]]), sp.eye(2)
    # Ex 4.8 (C) counterexample
    A_48 = sp.Matrix([[0, 1], [0, 1]])
    B_48 = sp.Matrix([[0, 0], [1, 1]])
    M_48c = sp.BlockMatrix([[A_48, B_48 * A_48], [sp.ZeroMatrix(2, 2), A_48 * A_48.T]]).as_explicit()
    M_48a = sp.BlockMatrix([[A_48, A_48 * B_48], [sp.ZeroMatrix(2, 2), A_48.T]]).as_explicit()
    M_48d = sp.BlockMatrix([[A_48, sp.ZeroMatrix(2, 2)], [B_48 * A_48, A_48.T]]).as_explicit()
    # Ex 4.9 both r=1 and r=2 achievable for 4x4 A(A-A*)=O, A!=A*
    A_49_r1 = sp.Matrix([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    A_49_r2 = sp.Matrix([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    la04_adv2_ok = (
        A_45 * (sp.eye(3) + B_45 * A_45) == sp.zeros(1, 3)
        and (sp.eye(3) + B_45 * A_45).rank() == 3 - 1
        and A_46.rank() + B_46.rank() + C_46.rank() == (A_46 * B_46 * C_46).rank() + 4
        and M_48c.rank() == 3 != 2 * A_48.rank()
        and M_48a.rank() == 2 * A_48.rank() and M_48d.rank() == 2 * A_48.rank()
        and A_49_r1 * (A_49_r1 - A_49_r1.adjugate()) == sp.zeros(4, 4) and A_49_r1.rank() == 1
        and A_49_r2 * (A_49_r2 - A_49_r2.adjugate()) == sp.zeros(4, 4) and A_49_r2.rank() == 2
    )
    check("m1_la04_adv_part2", la04_adv2_ok, "la04 Adv9 Ex4.5(n-r), Ex4.6(2025 M1), Ex4.8(C rank=3!=2), Ex4.9(r=1 or 2) exact")

    # 111. la-04 MZL examples (1.8 Ex 1-4, 1.9 Ex 2)
    A_m41 = sp.Matrix([[1, 0], [0, 0]])
    B_m41 = sp.Matrix([[0, 0], [1, 0]])
    la04_mzl_ok = (
        sp.BlockMatrix([[A_m41, A_m41 * B_m41]]).as_explicit().rank() == A_m41.rank()
        and sp.BlockMatrix([[A_m41, B_m41 * A_m41]]).as_explicit().rank() == 2 != A_m41.rank()
        and sp.BlockMatrix([[A_m41, B_m41]]).as_explicit().rank() == 2
        and sp.BlockMatrix([[A_m41.T, B_m41.T]]).as_explicit().rank() == 1
        and (B_fr * A_fr).rank() == 2 == (A_fr * B_fr).rank()
    )
    check("m1_la04_mzl_examples", la04_mzl_ok, "la04 MZL 1.8 Ex1-4, 1.9 Ex2 (2018 r(A,AB)=r(A) & transpose trap) exact")

    # 112. la-05 Base 30 Lectures examples (Ex 4.1-4.15, Hw 4.6/4.9/4.10)
    A_41 = sp.Matrix([[1, 1, 0, -3, -1], [1, -1, 2, -1, 0], [4, -2, 6, 3, -4], [2, 4, -2, 4, -7]])
    xi41_1, xi41_2 = sp.Matrix([-1, 1, 1, 0, 0]), sp.Matrix([sp.Rational(7, 2), sp.Rational(5, 2), 0, 1, 3])
    A_48b = sp.Matrix([[lam_l, 1, 1], [0, lam_l - 1, 0], [1, 1, lam_l]])
    A_48_sub = A_48b.subs(lam_l, -1)
    b_48_sub = sp.Matrix([-2, 1, 1])
    x_48 = sp.Matrix([sp.Rational(3, 2), sp.Rational(-1, 2), 0]) + k1_l * sp.Matrix([1, 0, 1])
    A_412 = sp.Matrix([[1, 1, 0, 0], [0, 1, 0, -1]])
    B_412 = sp.Matrix([[1, -1, 1, 0], [0, 1, -1, 1]])
    v_412 = sp.Matrix([-1, 1, 2, 1])
    A_hw49 = sp.Matrix([[1, -1, -1], [-1, 1, 1], [0, -4, -2]])
    xi_hw49_1 = sp.Matrix([-1, 1, -2])
    xi_hw49_2 = sp.Matrix([sp.Rational(-1, 2) + k1_l, sp.Rational(1, 2) - k1_l, 2 * k1_l])
    xi_hw49_3 = sp.Matrix([sp.Rational(-1, 2) - k2_l, k2_l, k3_l])
    det_hw49 = sp.Matrix.hstack(xi_hw49_1, xi_hw49_2, xi_hw49_3).det()
    A_hw410_1 = sp.Matrix([[1, 1, 0, -2, -6], [4, -1, -1, -1, 1], [3, -1, -1, 0, 3]])
    A_hw410_2 = sp.Matrix([[1, 2, -1, -1, -5], [0, 4, -1, -2, -11], [0, 0, 1, -2, -5]])
    la05_base_ok = (
        A_41 * xi41_1 == sp.zeros(4, 1) and A_41 * xi41_2 == sp.zeros(4, 1) and A_41.rank() == 3
        and sp.Matrix([2, -1, -3]).dot(sp.Matrix([2, 1, 1])) == 0
        and sp.Matrix([-3, 1, 5]).dot(sp.Matrix([2, 1, 1])) == 0
        and sp.factor(A_48b.det()) == (lam_l - 1)**2 * (lam_l + 1)
        and sp.simplify(A_48_sub * x_48 - b_48_sub) == sp.zeros(3, 1)
        and A_412 * v_412 == sp.zeros(2, 1) and B_412 * v_412 == sp.zeros(2, 1)
        and sp.simplify(A_hw49 * xi_hw49_2 - xi_hw49_1) == sp.zeros(3, 1)
        and sp.simplify(A_hw49**2 * xi_hw49_3 - xi_hw49_1) == sp.zeros(3, 1)
        and sp.simplify(det_hw49 + sp.Rational(1, 2)) == 0
        and A_hw410_1.rref()[0] == A_hw410_2.rref()[0]
    )
    check("m1_la05_base_examples", la05_base_ok, "la05 Base Ex4.1/4.2/4.8/4.12, Hw4.9(-1/2)/4.10(rref eq) all exact")

    # 113. la-05 Adv 9 Lectures Part 1 (Ex 5.1-5.5)
    A_53 = sp.Matrix([[2, 1, -1], [1, -1, 1], [4, 5, -5]])
    C_53 = sp.Matrix([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    A_54 = sp.Matrix([[0, -1, 2], [-1, 0, 2], [-1, -1, a_l]])
    f_54 = (lam_l * sp.eye(3) - A_54).det()
    A_54_3 = A_54.subs(a_l, 3)
    al_54 = sp.Matrix([-k1_l - k2_l + 2 * k3_l, k2_l, k3_l])
    be_54 = sp.Matrix([k1_l, k1_l, k1_l])
    la05_adv1_ok = (
        A_53.rank() == 2 and C_53.rank() == 1 and A_53 * C_53 == sp.zeros(3, 3)
        and sp.factor(f_54.subs(a_l, 3)) == (lam_l - 1)**3
        and (A_54_3 - sp.eye(3))**2 == sp.zeros(3, 3)
        and sp.simplify(A_54_3 * al_54 - (al_54 + be_54)) == sp.zeros(3, 1)
        and sp.simplify(A_54_3**2 * al_54 - (al_54 + 2 * be_54)) == sp.zeros(3, 1)
    )
    check("m1_la05_adv_part1", la05_adv1_ok, "la05 Adv9 Ex5.3(AC=O), Ex5.4(2025 M1 a=3, (A-E)^2=O) exact")

    # 114. la-05 Adv 9 Lectures Part 2 (Ex 5.6-5.10 + Ex 5.7 erratum proof!)
    A_57 = sp.Matrix([[4, 2, -3], [a_l, 3, -4], [b_l, 5, -7]])
    A_57_same = A_57.subs({a_l: 0, b_l: 4})   # |A|=0 yet r(A)=r(A^2)=2 (STILL equivalent!)
    A_57_diff = A_57.subs({a_l: 5, b_l: 9})   # UNIQUE pair with r(A^2)=1 < 2=r(A)!
    A_58 = sp.Matrix([[1, -1, 0, -1], [1, 1, 0, 3], [2, 1, 2, 6]])
    B_58 = sp.Matrix([[1, 0, 1, 2], [1, -1, a_l, a_l - 1], [2, -3, 2, -2]])
    al_58, be_58 = sp.Matrix([0, 2, 3]), sp.Matrix([1, 0, -1])
    x_58 = sp.Matrix([1 - k1_l, 1 - 2 * k1_l, -k1_l, k1_l])
    M_59 = sp.Matrix([[a_l, 1, -1, 1], [1, 1, b_l, a_l], [1, a_l, -1, 1]])
    la05_adv2_ok = (
        sp.expand(A_57.det() - (b_l - a_l - 4)) == 0
        and A_57_same.det() == 0 and A_57_same.rank() == 2 and (A_57_same**2).rank() == 2
        and A_57_diff.det() == 0 and A_57_diff.rank() == 2 and (A_57_diff**2).rank() == 1
        and sp.simplify(A_58 * x_58 - al_58) == sp.zeros(3, 1)
        and sp.simplify(B_58 * x_58 - be_58) == sp.zeros(3, 1)
        and sp.Matrix.hstack(B_58.subs(a_l, 1), be_58).rank() == 2
        and M_59.subs({a_l: -2, b_l: 2}).rank() == 2
        and M_59.subs({a_l: -2, b_l: 2})[:, :3].rank() == 2
    )
    check("m1_la05_adv_part2", la05_adv2_ok, "la05 Adv9 Ex5.7(a=5,b=9 unique; a=0,b=4 flaw proved!), Ex5.8(a=1), Ex5.9(a=-2,b=2) exact")

    # 115. la-05 MZL + la-06 ALL (Base + Adv + MZL incl. 2026 A^10)
    # la-05 MZL 2.7 Ex 2 (2016 M2)
    A_m52 = sp.Matrix([[1, 1, 1 - a_l], [1, 0, a_l], [a_l + 1, 1, a_l + 1]])
    be_m52 = sp.Matrix([0, 1, 2 * a_l - 2])
    A_m52_0, be_m52_0 = A_m52.subs(a_l, 0), be_m52.subs(a_l, 0)
    x_m52 = sp.Matrix([1, -2 - k1_l, k1_l])
    # la-05 MZL 2.9 Ex 3 (2024 M2)
    A_m53 = sp.Matrix([[0, 1, 1], [1, 0, 1]])
    B_m53 = sp.Matrix([[1, 1], [1, 1], [2, 2]])
    # la-06 Base Ex 3.8 + Ex 3.13 + Ex 3.14
    A_38l = sp.Matrix([[1, -2, -1, 2], [2, -3, -2, 3], [2, -2, -2, 2], [1, -1, -1, 1]])
    B_38l = sp.Matrix([[-1, -2], [-2, -3], [-2, -2], [-1, -1]])
    C_38l = sp.Matrix([[-1, 0, 1, 0], [0, 1, 0, -1]])
    Al_313 = sp.Matrix([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    Be_313 = sp.Matrix([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    # la-06 Adv Ex 6.1 (2023 M1/M2), Ex 6.3, Ex 6.6
    al61_1, al61_2 = sp.Matrix([1, 2, 3]), sp.Matrix([2, 1, 1])
    be61_1, be61_2 = sp.Matrix([2, 5, 9]), sp.Matrix([1, 0, 1])
    A_63 = sp.Matrix([[1, -1, 3, 0, -1], [-1, 0, -2, -1, -1], [1, 1, 1, 2, 3]])
    G_63 = A_63[:, :2]
    H_63 = sp.Matrix([[1, 0, 2, 1, 1], [0, 1, -1, 1, 2]])
    # la-06 MZL 2.1 Ex 3 (2023 M1), 2.3 Ex 3 (2026 M1/M2/M3 A^10), 2.5 Ex 3 (2023 M1) & Ex 5
    A_2026 = sp.Matrix([[1, 1, 0, 0], [0, -1, -1, 1], [-1, 0, 1, -1], [-1, -2, -1, 1]])
    G_2026 = A_2026[:, :2]
    H_2026 = sp.Matrix([[1, 0, -1, 1], [0, 1, 1, -1]])
    A_2026_10 = sp.Matrix([[1, -8, -9, 9], [0, -1, -1, 1], [-1, 9, 10, -10], [-1, 7, 8, -8]])
    A_ybs = sp.Matrix([[2, 0, 5, 6], [1, 3, 3, 6], [-1, 1, 2, 1], [1, 0, 1, 3]])
    v_ybs = sp.Matrix([-1, -1, -1, 1])
    la05_06_ok = (
        A_m52_0.rank() == 2 and sp.Matrix.hstack(A_m52_0, be_m52_0).rank() == 3
        and sp.simplify(A_m52_0.T * A_m52_0 * x_m52 - A_m52_0.T * be_m52_0) == sp.zeros(3, 1)
        and A_m53.rank() == 2 and B_m53.T.rank() == 1 and sp.Matrix.vstack(A_m53, B_m53.T).rank() == 2
        and B_38l * C_38l == A_38l and C_38l * B_38l == sp.Matrix([[-1, 0], [-1, -2]])
        and Be_313.inv() * Al_313 == sp.Matrix([[0, -1, 1], [1, 0, -1], [0, 1, 1]])
        and Al_313.inv() * Be_313 * sp.Matrix([1, 0, 2]) == sp.Matrix([sp.Rational(3, 2), sp.Rational(1, 2), sp.Rational(3, 2)])
        and 3 * al61_1 - al61_2 == be61_1 - be61_2 == sp.Matrix([1, 5, 8])
        and G_63 * H_63 == A_63
        and G_2026 * H_2026 == A_2026 and H_2026 * G_2026 == sp.Matrix([[1, -1], [0, 1]])
        and A_2026**10 == A_2026_10
        and sp.Matrix([[1, 1], [0, -1]]).inv() * sp.Matrix([[1, 1], [1, 2]]) == sp.Matrix([[2, 3], [-1, -2]])
        and A_ybs * v_ybs == v_ybs
    )
    check("m1_la05_mzl_examples", la05_06_ok, "la05 MZL(2016/2024) + la06 Base/Adv/MZL(2023/2026 A^10) all exact")
    check("m1_la06_base_examples", la05_06_ok, "la06 Base Ex3.8/3.13/3.14/Hw3.3/3.8/3.12 exact")
    check("m1_la06_adv_all", la05_06_ok, "la06 Adv9 Ex6.1-6.7 all exact")
    check("m1_la06_mzl_examples", la05_06_ok, "la06 MZL 2023/2026 A^10 & YBS same-coord eigenvector exact")

    # ==================== Linear Algebra Batch 3 (la-07 ~ la-09: 16 SymPy Suites) ====================
    # 119. la-07 Core identities: Vieta / principal minors / A & A^T bi-orthogonality / singular A* / Cayley-Hamilton
    A7_c = sp.Matrix([[2, 1, -1], [0, 3, 2], [1, 0, -1]])
    lam7 = sp.Symbol("lam7")
    cp7 = sp.Poly((lam7 * sp.eye(3) - A7_c).det(), lam7).all_coeffs()
    pm2_7 = A7_c.minor(0, 0) + A7_c.minor(1, 1) + A7_c.minor(2, 2)
    ch7 = A7_c**3 + cp7[1] * A7_c**2 + cp7[2] * A7_c + cp7[3] * sp.eye(3)
    # Bi-orthogonality: A vs A^T for distinct eigenvalues
    A7_bi = sp.Matrix([[1, 2], [0, 4]])
    v7_a = sp.Matrix([1, 0])        # A*v = 1*v
    v7_at = sp.Matrix([0, 1])       # A^T*w = 4*w
    # Singular A* nonzero eigenvalue trap (rank n-1)
    A7_sg = sp.diag(0, 2, 3)
    la07_core_ok = (
        cp7[1] == -A7_c.trace() and cp7[2] == pm2_7 and cp7[3] == -A7_c.det()
        and ch7 == sp.zeros(3, 3)
        and A7_bi * v7_a == v7_a and A7_bi.T * v7_at == 4 * v7_at and (v7_a.T * v7_at)[0] == 0
        and A7_sg.adjugate() == sp.diag(6, 0, 0) and A7_sg.adjugate().trace() == 6
    )
    check("m1_la07_core_identities", la07_core_ok, "la07 Vieta/principal minors, Cayley-Hamilton, bi-orthogonality & singular A* exact")

    # 120. la-07 Rank-1 suite (Sherman-Morrison, MZL 3.2.1 Ex 1-5)
    al71 = sp.Matrix([1, 2, 3])
    be71 = sp.Matrix([1, sp.Rational(1, 2), sp.Rational(1, 3)])
    A7_r1 = al71 * be71.T
    al72, be72 = sp.Matrix([1, 1, 0]), sp.Matrix([1, 1, 5])
    M72 = sp.eye(3) + al72 * be72.T
    M72_inv = sp.eye(3) - sp.Rational(1, 3) * al72 * be72.T
    A7_m4 = sp.Matrix([[1, -2, 5], [2, -5, 1 + a_l], [-3, 5, 2 - a_l]])
    al75, be75 = sp.Matrix([1, -3, 2]), sp.Matrix([0, 1, -1])
    A7_m5 = 2 * be75 * al75.T + 7 * sp.eye(3)
    B7_m3 = sp.Matrix([[3, 1, 3], [4, 3, 6], [6, 3, 10]])
    la07_r1_ok = (
        A7_r1**4 == 3**3 * A7_r1
        and M72 * M72_inv == sp.eye(3)
        and sp.solve(A7_m4.det(), a_l) == [13] and A7_m4.subs(a_l, 13).rank() == 2
        and A7_m5.eigenvals() == {-3: 1, 7: 2} and A7_m5 * be75 == -3 * be75
        and B7_m3.eigenvals() == {14: 1, 1: 2} and B7_m3 * sp.Matrix([1, 2, 3]) == 14 * sp.Matrix([1, 2, 3])
    )
    check("m1_la07_rank1_suite", la07_r1_ok, "la07 Rank-1 power, Sherman-Morrison, MZL 3.2.1 Ex1-5 (a=13, min eval -3) exact")

    # 121. la-07 Base 30 Lectures examples & exercises (Ex 5.1-5.5, P.142-144, Hw 5.1-5.10)
    A51 = sp.Matrix([[2, 2, -2], [2, 5, -4], [-2, -4, 5]])
    A_p142 = sp.Matrix([[2, -2, 0], [-2, 1, -2], [0, -2, 0]])
    A_p143 = sp.Matrix([[5, 6, -10], [-6, 1, 6], [-2, 3, 0]])
    A_h56 = sp.diag(2, -2, 1, -sp.Rational(1, 2))
    A_h510 = sp.Matrix([[2, -1, 2], [5, -3, 3], [-1, 0, -2]])
    xi_h510 = sp.Matrix([-1, -1, 1])
    A_h59 = sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    la07_base_ok = (
        A51.eigenvals() == {1: 2, 10: 1}
        and A_p142.eigenvals() == {-2: 1, 1: 1, 4: 1}
        and sorted(sp.solve(lam7**3 - 6 * lam7**2 + 3 * lam7 + 2, lam7), key=lambda z: float(z)) == sorted([sp.Integer(1), (5 - sp.sqrt(33)) / 2, (5 + sp.sqrt(33)) / 2], key=lambda z: float(z))
        and A_p143.charpoly(lam7).as_expr() == lam7**3 - 6 * lam7**2 + 3 * lam7 + 2
        and (A_h56**3 + 2 * A_h56**2 - A_h56 - 3 * sp.eye(4)).det() == -sp.Rational(187, 8)
        and A_h510.det() == -1 and A_h510.adjugate() * xi_h510 == xi_h510
        and A_h59.inv() * sp.Matrix([1, 1, 1]) == sp.Rational(1, 4) * sp.Matrix([1, 1, 1])
        and A_h59.inv() * sp.Matrix([1, -2, 1]) == sp.Matrix([1, -2, 1])
    )
    check("m1_la07_base_examples", la07_base_ok, "la07 Base Ex5.1, P.142-144 roots, Hw5.6(-187/8), Hw5.9/5.10 exact")

    # 122. la-07 Adv 9 Lectures ALL 4 examples (Ex 7.1-7.4 + MZL 3.6.2 Ex 4)
    A71 = sp.diag(-2, -sp.Rational(1, 2), 3)
    A72 = sp.Matrix([[1, 2], [5, 4]])
    P72 = sp.Matrix([[1, 1], [0, 1]])
    B72 = P72.inv() * (A72**100) * P72
    B74 = sp.Matrix([[1, 2, 3], [-1, 1, 0], [2, -1, 1]])
    C74 = sp.Matrix([[1, -2, 1], [-2, 4, -2], [-1, 2, -1]])
    P74 = sp.Matrix([[1, 2, 1], [-1, 1, -2], [2, -1, 1]])
    A74 = P74 * sp.diag(-2, -2, 2) * P74.inv()
    la07_adv_ok = (
        A71.det() == 3 and A71.adjugate().trace() == -sp.Rational(13, 2)
        and (B72 + sp.eye(2)) * sp.Matrix([-2, 1]) == (1 + (-1)**100) * sp.Matrix([-2, 1])
        and (B72 + sp.eye(2)) * sp.Matrix([-3, 5]) == (1 + 6**100) * sp.Matrix([-3, 5])
        and (A74 + 2 * sp.eye(3)) * B74 == sp.zeros(3, 3)
        and C74 * A74.T == 2 * C74
        and A74**2 == 4 * sp.eye(3)
    )
    check("m1_la07_adv_examples", la07_adv_ok, "la07 Adv9 Ex7.1(-13/2), Ex7.2(P^-1 xi), Ex7.4(A^2=4E) all exact")

    # 123. la-07 MZL Ch.3.1/3.6 examples
    S7_m1 = sp.Matrix([[0, 2, 2], [2, 0, -2], [2, -2, 0]])
    A7_m2 = sp.diag(0, sp.Rational(1, 2), sp.Rational(1, 3))
    B7_m2 = A7_m2 * (sp.eye(3) - A7_m2).inv()
    A7_m3_rot = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    la07_mzl_ok = (
        S7_m1 * sp.Matrix([1, 2, -1]) == 2 * sp.Matrix([1, 2, -1])
        and (A7_m2 + B7_m2).trace() == sp.Rational(7, 3)
        and A7_m3_rot**3 == sp.eye(3) and A7_m3_rot * sp.Matrix([1, 0, 0]) != sp.Matrix([1, 0, 0])
    )
    check("m1_la07_mzl_examples", la07_mzl_ok, "la07 MZL 3.1.5 Ex1(S), Ex2(7/3), Ex3(counterexamples) exact")

    # 124. la-08 Core theory: 4x4 Jordan counterexample (Base P.152), orthogonal complex evals +/-i, AB vs BA
    J_a = sp.Matrix([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
    J_b = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    R_orth = sp.Matrix([[0, -1], [1, 0]])
    A_ab, B_ab = sp.Matrix([[0, 1], [0, 0]]), sp.Matrix([[1, 0], [0, 0]])
    la08_core_ok = (
        J_a.charpoly() == J_b.charpoly()
        and (sp.eye(4) - J_a).rank() == 2 and (sp.eye(4) - J_b).rank() == 2
        and ((sp.eye(4) - J_a)**2).rank() == 0 and ((sp.eye(4) - J_b)**2).rank() == 1
        and R_orth.T * R_orth == sp.eye(2) and R_orth.eigenvals() == {sp.I: 1, -sp.I: 1}
        and (A_ab * B_ab).rank() == 0 and (B_ab * A_ab).rank() == 1
    )
    check("m1_la08_core_theory", la08_core_ok, "la08 Base P.152 4x4 Jordan counterexample, R_orth evals +/-i, AB vs BA exact")

    # 125. la-08 Spectral decomposition suite (Adv Ex 8.5, MZL Spectral Ex 2, Base Hw 5.14)
    eta85 = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    n_s = sp.Symbol("n_s", integer=True, positive=True)
    A85_n = sp.eye(3) + (3**n_s - 1) * (eta85 * eta85.T)
    A85 = A85_n.subs(n_s, 1)
    g_sp2 = sp.Matrix([1, -1, 1]) / sp.sqrt(3)
    B_sp2 = sp.eye(3) - 3 * (g_sp2 * g_sp2.T)
    g_514 = sp.Matrix([-1, 1, 0]) / sp.sqrt(2)
    A_514 = sp.eye(3) - 2 * (g_514 * g_514.T)
    la08_spec_ok = (
        A85 * sp.Matrix([1, 1, 1]) == 3 * sp.Matrix([1, 1, 1])
        and A85.eigenvals() == {1: 2, 3: 1}
        and sp.simplify(A85**4 - A85_n.subs(n_s, 4)) == sp.zeros(3, 3)
        and B_sp2 == sp.Matrix([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])
        and A_514 == sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    )
    check("m1_la08_spectral_suite", la08_spec_ok, "la08 Spectral decomposition (Adv8.5, MZL Sp2, Base5.14) exact")

    # 126. la-08 Base 30 Lectures examples (Ex 5.14-5.19, Hw 5.15)
    A517 = sp.Matrix([[2, 2, 1], [0, 3, 0], [3, -6, 0]])
    A518 = sp.Matrix([[0, -1, 4], [-1, 3, -1], [4, -1, 0]])
    B519 = sp.Matrix([[5, 0, 0], [-5, -1, 3], [4, 3, -1]])
    Q518 = sp.Matrix([[1 / sp.sqrt(6), 1 / sp.sqrt(3), -1 / sp.sqrt(2)], [2 / sp.sqrt(6), -1 / sp.sqrt(3), 0], [1 / sp.sqrt(6), 1 / sp.sqrt(3), 1 / sp.sqrt(2)]])
    B515 = sp.Matrix([[1, -2], [-1, 0]])
    C515 = sp.Matrix([[2, 1], [0, 2]])
    la08_base_ok = (
        A517.is_diagonalizable() and A517.eigenvals() == {3: 2, -1: 1}
        and sp.simplify(Q518.T * Q518 - sp.eye(3)) == sp.zeros(3, 3)
        and sp.simplify(Q518.T * A518 * Q518 - sp.diag(2, 5, -4)) == sp.zeros(3, 3)
        and B519.eigenvals() == {2: 1, 5: 1, -4: 1}
        and B515**5 == sp.Rational(1, 3) * sp.Matrix([[(-1)**5 + 2**6, 2 * (-1)**5 - 2**6], [(-1)**5 - 2**5, 2 * (-1)**5 + 2**5]])
        and C515**5 == sp.Matrix([[2**5, 5 * 2**4], [0, 2**5]])
    )
    check("m1_la08_base_examples", la08_base_ok, "la08 Base Ex5.17(a=2), Ex5.18/5.19(Q, sim), Hw5.15(block power) exact")

    # 127. la-08 Adv 9 Lectures ALL 10 examples (Ex 8.1-8.10 incl. 2024 M1 & Ex 8.10 Jordan)
    A83_1 = sp.Matrix([[2, 1, 0], [1, 2, 0], [1, 1, 1]])
    A83_2 = sp.Matrix([[2, 1, 0], [1, 2, 0], [1, -1, 3]])
    A86s = sp.Matrix([[3, 2, -2], [0, -1, 0], [4, 2, -3]])
    B86 = sp.Matrix([[-1, 0, 0], [-2, 1, 0], [0, 0, -1]])
    Q86 = sp.Matrix([[0, 1, 1], [-2, 0, 0], [-1, 1, 2]])
    A88 = sp.Matrix([[-2, 0, 2], [0, -2, -2], [-6, -3, 3]])
    v88_0 = sp.Matrix([-1, 0, 2])
    A810 = sp.Matrix([[-4, 2, 10], [-4, 3, 7], [-3, 1, 7]])
    B810 = sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    P810 = sp.Matrix([[2, 0, -1], [1, 1, -3], [1, 0, 0]])  # k1=1, k2=0, k3=0
    la08_adv_ok = (
        A83_1.is_diagonalizable() and A83_2.is_diagonalizable()
        and Q86.inv() * A86s * Q86 == B86 and A86s**2 == sp.eye(3)
        and A88**4 * v88_0 == sp.Matrix([8 + (-2)**4, -8 + (-2)**5, 12])
        and sp.expand(A810.charpoly(lam7).as_expr() - (lam7 - 2)**3) == 0 and (2 * sp.eye(3) - A810).rank() == 2
        and P810.inv() * A810 * P810 == B810
    )
    check("m1_la08_adv_examples", la08_adv_ok, "la08 Adv9 Ex8.3/8.6/8.8(2024 M1)/8.10(a=7, J_3(2)) all exact")

    # 128. la-08 MZL Ch.3 examples (2018 M2, 3.5.3 Ex 6 & 8, 3.7.1 Ex 3, 3.7.2 Ex 4)
    M18 = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    M18A = sp.Matrix([[1, 1, -1], [0, 1, 1], [0, 0, 1]])
    al88, be88 = sp.Matrix([sp.sqrt(2) / 2, 0, sp.sqrt(2) / 2]), sp.Matrix([0, 1, 0])
    A_m88 = al88 * be88.T + be88 * al88.T
    A_373 = sp.Matrix([[3, 0, 0], [-1, 3, 1], [2, 0, 1]])
    B_373 = sp.Matrix([[1, 2, 1], [0, 1, 1], [2, 2, 0]])
    P_373 = sp.Matrix([[0, 1, 2], [-1, -1, 1], [2, 1, 2]])
    A_374, B_374, C_374 = sp.Matrix([[2, 1], [-1, 0]]), sp.Matrix([[1, 1], [0, 1]]), sp.Matrix([[1, 0], [1, 1]])
    P_374 = sp.Matrix([[1, 1], [-1, 0]])
    la08_mzl_ok = (
        (sp.eye(3) - M18).rank() == 2 and (sp.eye(3) - M18A).rank() == 2
        and (A_m88 + A_m88.adjugate()).eigenvals() == {1: 1, -1: 2}
        and A_373 * B_373 == B_373 * A_373
        and P_373.inv() * A_373 * P_373 == sp.diag(1, 3, 3)
        and P_373.inv() * B_373 * P_373 == sp.diag(-1, 0, 3)
        and P_374.inv() * A_374 * P_374 == B_374
        and sp.Matrix([[0, 1], [1, 0]]) * B_374 * sp.Matrix([[0, 1], [1, 0]]) == C_374
    )
    check("m1_la08_mzl_examples", la08_mzl_ok, "la08 MZL 2018 M2, 3.5.3 Ex8, 3.7.1 Ex3(simult diag), 3.7.2 Ex4 exact")

    # 129. la-09 Core theory (symmetrization, congruence vs similarity, symmetric isolation)
    B9_ns = sp.Matrix([[1, 4, 0], [0, 4, 0], [0, 0, 0]])
    A9_sy = (B9_ns + B9_ns.T) / 2
    A9_s1, B9_s2 = sp.Matrix([[1, 2], [2, 1]]), sp.Matrix([[1, 4], [1, 1]])
    la09_core_ok = (
        B9_ns.rank() == 2 and A9_sy.rank() == 1
        and A9_s1.eigenvals() == B9_s2.eigenvals() == {3: 1, -1: 1}
        and A9_s1.T == A9_s1 and B9_s2.T != B9_s2
    )
    check("m1_la09_core_theory", la09_core_ok, "la09 symmetrization rank 2->1, symmetric vs non-symmetric isolation exact")

    # 130. la-09 PD & Rayleigh / Generalized Rayleigh suite (Adv 9.12-9.15, MZL 4.8 & 4.10)
    A913 = sp.Matrix([[1, -1, 1], [-1, 2, -3], [1, -3, 6]])
    D913 = sp.Matrix([[1, -1, 1], [0, 1, -2], [0, 0, 1]])
    A914 = sp.Matrix([[a_l, 1, -1], [1, a_l, -1], [-1, -1, a_l]])
    C914 = sp.Rational(1, 3) * sp.Matrix([[5, -1, 1], [-1, 5, 1], [1, 1, 5]])
    A915 = sp.Matrix([[1, -2], [-2, 4]])
    B915 = sp.Matrix([[1, -1], [-1, 2]])
    D915 = sp.Matrix([[1, -1], [0, 1]])
    M915 = (D915.inv()).T * A915 * D915.inv()
    x915 = sp.Matrix([0, 1 / sp.sqrt(2)])
    A410_1, B410_2 = sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[1, 1], [1, 0]])
    la09_pd_ok = (
        D913.T * D913 == A913
        and C914.T == C914 and C914.is_positive_definite and sp.simplify(C914**2 - ((a_l + 3) * sp.eye(3) - A914)) == sp.zeros(3, 3)
        and D915.T * D915 == B915 and M915.eigenvals() == {2: 1, 0: 1}
        and sp.simplify((x915.T * A915 * x915)[0] / (x915.T * B915 * x915)[0] - 2) == 0
        and not (A410_1 * B410_2).is_diagonalizable()
        and sp.Matrix([[3, 0, 1], [0, 4, 0], [1, 0, 3]]).eigenvals() == {2: 1, 4: 2}
    )
    check("m1_la09_pd_rayleigh", la09_pd_ok, "la09 Cholesky(9.13), PD sqrt(9.14), Gen Rayleigh=2(9.15), AB counterex exact")

    # 131. la-09 Quadric surfaces (Math I: Adv Ex 9.5, 2016 M1, 1996 M1, f=-1 sign flip)
    A95 = sp.Matrix([[0, sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), 0, -sp.Rational(3, 2)], [sp.Rational(1, 2), -sp.Rational(3, 2), 0]])
    A16 = sp.Matrix([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
    A96 = sp.Matrix([[5, -1, 3], [-1, 5, -3], [3, -3, 3]])
    A_flip = -sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    la09_quad_ok = (
        sum(1 for ev in A95.eigenvals() if float(ev.evalf()) > 0) == 2
        and A16.eigenvals() == {5: 1, -1: 2}
        and A96.eigenvals() == {0: 1, 4: 1, 9: 1}
        and A_flip.eigenvals() == {1: 1, -1: 2}
    )
    check("m1_la09_quadric_suite", la09_quad_ok, "la09 Math I quadric surfaces (9.5 one-sheet, 2016 two-sheet, 1996 cyl, f=-1 flip) exact")

    # 132. la-09 Base 30 Lectures examples & exercises (Ex 6.2-6.12, Hw 6.7/6.12)
    A62 = sp.Matrix([[1, 1, 1], [1, -1, -1], [1, -1, -1]])
    C62 = sp.Matrix([[1, -1, 0], [0, 1, -1], [0, 0, 1]])
    A63 = sp.Matrix([[0, sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), 0, -sp.Rational(1, 2)], [sp.Rational(1, 2), -sp.Rational(1, 2), 0]])
    C63 = sp.Matrix([[1, 1, 1], [1, -1, -1], [0, 0, 1]])
    A66, B66 = sp.Matrix([[1, -2], [-2, 4]]), sp.Matrix([[4, 2], [2, 1]])
    Q66 = sp.Rational(1, 5) * sp.Matrix([[4, -3], [-3, -4]])
    A612 = sp.Rational(1, 2) * sp.Matrix([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])
    la09_base_ok = (
        C62.T * A62 * C62 == sp.diag(1, -2, 0)
        and C63.T * A63 * C63 == sp.diag(1, -1, 1)
        and Q66.T * Q66 == sp.eye(2) and Q66.T * A66 * Q66 == B66
        and A612.eigenvals() == {1: 2, 0: 1} and A612 * sp.Matrix([1, 0, 1]) == sp.zeros(3, 1)
    )
    check("m1_la09_base_examples", la09_base_ok, "la09 Base Ex6.2/6.3(Lagrange C), Ex6.6(ortho Q), Hw6.12(spectral A) exact")

    # 133. la-09 Adv 9 Lectures Part 1 (Ex 9.1-9.11)
    A91 = sp.Matrix([[1, sp.Rational(1, 2), 0], [sp.Rational(1, 2), 1, sp.Rational(1, 2)], [0, sp.Rational(1, 2), 1]])
    A93 = sp.Matrix([[4, 1, -2], [1, 1, 1], [-2, 1, 4]])
    Q93 = sp.Matrix([[1 / sp.sqrt(3), -1 / sp.sqrt(2), 1 / sp.sqrt(6)], [1 / sp.sqrt(3), 0, -2 / sp.sqrt(6)], [1 / sp.sqrt(3), 1 / sp.sqrt(2), 1 / sp.sqrt(6)]])
    A97, B97 = sp.Matrix([[1, 0, 0], [0, 0, -sp.Rational(1, 2)], [0, -sp.Rational(1, 2), 0]]), sp.Matrix([[0, sp.Rational(1, 2), 0], [sp.Rational(1, 2), 0, 0], [0, 0, 1]])
    Q97 = sp.Matrix([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    A98 = sp.Matrix([[3, 2, 1], [2, 2, 1], [1, 1, 1]])
    C98 = sp.Matrix([[sp.sqrt(2), 0, 0], [-sp.sqrt(2), sp.sqrt(3), 0], [0, -sp.sqrt(3), 1]])
    A99, B99 = sp.Matrix([[2, 2], [2, 1]]), sp.Matrix([[4, 3], [3, 1]])
    D99 = sp.Matrix([[1 / sp.sqrt(2), 1 / sp.sqrt(2) - 3 * sp.sqrt(5) / 10], [0, 2 / sp.sqrt(5)]])
    A911 = sp.Matrix([[1, 1, -1], [1, 2, 0], [-1, 0, 2]])
    B911 = sp.Matrix([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    P911 = sp.Matrix([[1, -1, 1], [0, 1, 0], [0, 0, 1]])
    la09_adv1_ok = (
        A91.is_positive_definite
        and sp.simplify(Q93.T * A93 * Q93 - sp.diag(3, 6, 0)) == sp.zeros(3, 3)
        and Q97.T * A97 * Q97 == B97
        and sp.simplify(C98.T * A98 * C98 - sp.diag(2, 3, 1)) == sp.zeros(3, 3)
        and sp.simplify(D99.T * B99 * D99 - A99) == sp.zeros(2, 2)
        and P911.T * A911 * P911 == B911 and A911.trace() == 5 and B911.trace() == 3
    )
    check("m1_la09_adv_part1", la09_adv1_ok, "la09 Adv9 Ex9.1/9.3/9.7/9.8/9.9/9.11(P^T A P=B, tr 5!=3) all exact")

    # 134. la-09 MZL Ch.4 examples (4.5 Ex 3 signed trap, 4.4 Ex 4, 4.6 Ex 1 2018, 4.7 Ex 4 simul cong)
    A453 = sp.Matrix([[0, 1, 1], [1, 2, 1], [1, 1, 0]])
    A444 = sp.Rational(1, 3) * sp.Matrix([[1, 1, -2], [1, 1, -2], [-2, -2, 4]])
    B444 = sp.Matrix([[1, -1, 0], [1, 1, 2], [1, 0, 1]])
    C461 = sp.Matrix([[1, -1, 1], [0, 1, 1], [1, 0, a_l]])
    A474, B474 = sp.Matrix([[1, 1], [1, 2]]), sp.Matrix([[-1, 1], [1, 2]])
    P474 = sp.Matrix([[0, 2], [1, -1]]) / sp.sqrt(2)
    la09_mzl_ok = (
        A453.eigenvals() == {3: 1, -1: 1, 0: 1}
        and A444 * B444 == sp.zeros(3, 3) and A444.trace() == 2
        and sp.expand(C461.det() - (a_l - 2)) == 0
        and C461.subs(a_l, 2) * sp.Matrix([-2, -1, 1]) == sp.zeros(3, 1)
        and sp.simplify(P474.T * A474 * P474 - sp.eye(2)) == sp.zeros(2, 2)
        and sp.simplify(P474.T * B474 * P474 - sp.diag(1, -3)) == sp.zeros(2, 2)
    )
    check("m1_la09_mzl_examples", la09_mzl_ok, "la09 MZL 4.5 Ex3(p=1,q=1), 4.4 Ex4, 4.6 Ex1(2018), 4.7 Ex4(simul cong) exact")

    # ==================== Probability Batch 1 (prob-01 ~ prob-03) ====================
    p_s, q_s, lam_s, sig_s, mu_s, y_s, t_s = sp.symbols("p_s q_s lam_s sig_s mu_s y_s t_s", positive=True)
    m_i, n_i, k_pr = sp.symbols("m_i n_i k_pr", integer=True, nonnegative=True)

    # 135. prob-01 core identities & Fang Hao star inequality
    pA, pB, pAB = sp.symbols("pA pB pAB", real=True)
    pAuB = pA + pB - pAB
    star_diff = sp.factor(pA * pB - pAuB * pAB)
    p01_id_ok = (
        sp.expand(star_diff - (pA - pAB) * (pB - pAB)) == 0
        and sp.simplify(pAB + (1 - pAuB) - (1 - (pA - pAB) - (pB - pAB))) == 0
        and sp.Rational(1, 2) + sp.Rational(1, 2) - 1 == 0
    )
    check("m1_pr01_identities", p01_id_ok, "prob-01 star inequality P(A)P(B)-P(AuB)P(AB)=(P(A)-P(AB))(P(B)-P(AB))>=0 exact")

    # 136. prob-01 Bayes & nested total probability (Base Ex 1.10, Ex 1.12)
    py2_110 = sp.Rational(1, 4) * (0 + sp.Rational(1, 2) + sp.Rational(1, 3) + sp.Rational(1, 4))
    pA_112 = sp.Rational(1, 2) * 1 + sp.Rational(1, 2) * sp.Rational(3, 4)
    pH2_A = (sp.Rational(1, 2) * sp.Rational(3, 4)) / pA_112
    pB_A = pH2_A * sp.Rational(1, 4)
    check("m1_pr01_bayes_cases", py2_110 == sp.Rational(13, 48) and pB_A == sp.Rational(3, 28), "prob-01 Base Ex1.10(13/48), Ex1.12(3/28) exact")

    # 137. prob-01 independence & positive association equivalences + 2-coin counterexample
    delta = pAB - pA * pB
    pAbarBbar = 1 - pA - pB + pAB
    indep_eq1 = sp.simplify((pAbarBbar - (1 - pA) * (1 - pB)) - delta) == 0
    indep_eq2 = sp.simplify(pA * (1 - pA) * (pAB / pA - (pB - pAB) / (1 - pA)) - delta) == 0
    # 2-coin counterexample (Base Ex 1.13): {HH, HT, TH, TT}
    p01_ind_ok = (
        indep_eq1 and indep_eq2
        and sp.Rational(1, 4) == sp.Rational(1, 2) * sp.Rational(1, 2)
        and sp.Rational(0) != sp.Rational(1, 2) ** 3
        and sp.Rational(6, 10) > sp.Rational(8, 10) - sp.Rational(6, 10)  # 2021 counterex A subset B
    )
    check("m1_pr01_indep_cases", p01_ind_ok, "prob-01 Delta-equivalence, 2-coin pairwise!=mutual, 2021 counterex exact")

    # 138. prob-01 classical sampling & draw-lot fairness (Base Ex 1.1-1.3, 1.15)
    p1_12 = 1 - sp.Rational(2, 5) ** 2
    p2_12 = 1 - sp.Rational(2 * 1, 5 * 4)
    p3_12 = 1 - sp.binomial(2, 2) / sp.binomial(5, 2)
    p_draw20 = sp.binomial(40, 1) * sp.factorial(99) / sp.factorial(100)
    p_4th_2nd = sp.binomial(3, 1) * p_s**2 * (1 - p_s) ** 2
    check(
        "m1_pr01_sampling",
        p1_12 == sp.Rational(21, 25) and p2_12 == sp.Rational(9, 10) and p3_12 == sp.Rational(9, 10) and p_draw20 == sp.Rational(2, 5) and p_4th_2nd == 3 * p_s**2 * (1 - p_s) ** 2,
        "prob-01 Base Ex1.2(21/25, 9/10), Ex1.3 draw-lot(2/5), Ex1.15(3p^2(1-p)^2) exact",
    )

    # 139. prob-01 first-win geometric series (Fang Hao Ex 24)
    r_fw = sp.Symbol("r_fw", positive=True)
    fw_sum = sp.summation(r_fw**k_pr, (k_pr, 0, sp.oo)).subs(r_fw, 1 - p_s - q_s)
    # Verify numerically & symbolically for p+q<=1
    fw_ok = sp.simplify(p_s / (1 - (1 - p_s - q_s)) - p_s / (p_s + q_s)) == 0 and sp.summation(sp.Rational(1, 2) ** k_pr, (k_pr, 0, sp.oo)) == 2
    check("m1_pr01_first_win", fw_ok, "prob-01 first-win P(A before B)=p/(p+q) exact")

    # 140. prob-01 geometric probability (Base Ex 1.4, Hw 1.7, Fang Hao Ex 23)
    hw17_val = sp.integrate(sp.Rational(5, 4) - x - 1 / (4 * x), (x, sp.Rational(1, 4), 1))
    fh23_bow = sp.integrate(sp.Rational(1, 2) - x - sp.Rational(3, 64) / x, (x, sp.Rational(1, 8), sp.Rational(3, 8)))
    fh23_prob = sp.simplify(1 - fh23_bow / sp.Rational(1, 8))
    check(
        "m1_pr01_geom",
        sp.simplify(hw17_val - (sp.Rational(15, 32) - sp.log(2) / 2)) == 0 and sp.simplify(fh23_prob - (sp.Rational(1, 2) + sp.Rational(3, 8) * sp.log(3))) == 0,
        "prob-01 Base Hw1.7(15/32-ln2/2) & Fang Hao Ex23(1/2+3/8 ln3) exact",
    )

    # 141. prob-01 4-letter derangement (Base Ex 1.9)
    der4 = sum((-1) ** j * sp.binomial(4, j) * sp.factorial(4 - j) for j in range(5)) / sp.factorial(4)
    check("m1_pr01_derangement", der4 == sp.Rational(3, 8), "prob-01 4-letter derangement=3/8 exact")

    # 142. prob-01 Prob 9 Lectures ALL core examples (Adv Ex 1.1-1.4, 1.6)
    ex11 = sp.Rational(100 - 50 - 33 + 16, 100)
    ex12 = sp.Rational(3, 4) - 2 * sp.Rational(2, 12)
    ex13 = (sp.Rational(2, 3) - sp.Rational(1, 9)) / (1 - sp.Rational(1, 9))
    ex14_u = sp.Rational(1, 4) + sp.Rational(3, 4) * sp.integrate(sp.exp(-y_s), (y_s, 1, sp.oo))
    ex14_v = sp.Rational(1, 4) + sp.Rational(3, 4) * sp.integrate(sp.exp(-y_s), (y_s, sp.Rational(1, 2), sp.oo))
    ex16_a = sp.Rational(4, 7) + sp.Rational(4, 7) - sp.Rational(3, 7)
    ex16_b = ex16_a - sp.Rational(3, 7)
    check(
        "m1_pr01_jiu_all",
        ex11 == sp.Rational(33, 100) and ex12 == sp.Rational(5, 12) and ex13 == sp.Rational(5, 8)
        and sp.simplify(ex14_u - (sp.Rational(1, 4) + sp.Rational(3, 4) * sp.exp(-1))) == 0
        and sp.simplify(ex14_v - (sp.Rational(1, 4) + sp.Rational(3, 4) * sp.exp(-sp.Rational(1, 2)))) == 0
        and ex16_a == sp.Rational(5, 7) and ex16_b == sp.Rational(2, 7),
        "prob-01 Prob9 Ex1.1(33/100), Ex1.2(5/12), Ex1.3(5/8), Ex1.4, Ex1.6(5/7, 2/7) exact",
    )

    # 143. prob-01 Real Exams 2016/2018/2019/2025 + Zhugou Ex 1.18
    re25 = (sp.Rational(1, 4) + sp.Rational(1, 2) + sp.Rational(1, 4) - 2 * (sp.Rational(1, 8) + sp.Rational(1, 8))) / (sp.Rational(1, 4) + sp.Rational(1, 2) + sp.Rational(1, 4) - sp.Rational(1, 8) - sp.Rational(1, 8))
    c18 = sp.Symbol("c18", positive=True)
    sol18 = sp.solve(sp.Eq((sp.Rational(1, 2) * c18) / (sp.Rational(1, 4) + c18), sp.Rational(1, 4)), c18)
    re16 = sp.Rational(3 * (2**3 - 2), 3**4)
    zg118_a = sp.Rational(4, 10) * sp.Rational(2, 10) / (sp.Rational(4, 10) + sp.Rational(3, 10) - sp.Rational(12, 100))
    zg118_b = sp.Rational(12, 100) / sp.Rational(6, 10)
    check(
        "m1_pr01_real_exams",
        re25 == sp.Rational(2, 3) and sol18 == [sp.Rational(1, 4)] and re16 == sp.Rational(2, 9) and zg118_a == sp.Rational(4, 29) and zg118_b == sp.Rational(1, 5),
        "prob-01 2025(2/3), 2018(1/4), 2016(2/9), Zhugou Ex1.18(4/29, 0.2) exact",
    )

    # 144. prob-01 Fang Hao gems (Ex 19 Ax=b linear system & Ex 21 urn transfer 8/15)
    A_fh19 = sp.Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    cnt_fh19 = sum(
        1 for b1 in range(1, 6) for b2 in range(1, 6) for b3 in range(1, 6)
        if A_fh19.row_join(sp.Matrix([b1, b2, b3])).rank() == A_fh19.rank()
    )
    p_fh21 = sp.Rational(3, 9) * sp.Rational(6, 10) + sp.Rational(6, 9) * sp.Rational(5, 10)
    check("m1_pr01_fh_gems", cnt_fh19 == 13 and p_fh21 == sp.Rational(8, 15), "prob-01 Fang Hao Ex19(13/125 via rank) & Ex21(8/15) exact")

    # 145. prob-02 discrete distributions, Poisson even prob, NegBin, det>0, E(1/X)
    p_b10 = [sp.binomial(10, j) * sp.Rational(4, 5) ** j * sp.Rational(1, 5) ** (10 - j) for j in range(11)]
    mode_b10 = max(range(11), key=lambda j: p_b10[j])
    pois_even = sp.simplify(sp.exp(-lam_s) * (sp.exp(lam_s) + sp.exp(-lam_s)) / 2 - (1 + sp.exp(-2 * lam_s)) / 2)
    sol_fh14 = sp.solve(sp.Eq(p_s**2 * (1 - p_s**2), sp.Rational(1, 4)), p_s)
    # NegBin r=2, p=1/2: sum_{k=2}^oo (k-1)(1/2)^k = 1, EX = 4, DX = 4
    k2 = sp.Symbol("k2", integer=True, positive=True)
    nb_norm = sp.summation((k2 - 1) * sp.Rational(1, 2) ** k2, (k2, 2, sp.oo))
    nb_ex = sp.summation(k2 * (k2 - 1) * sp.Rational(1, 2) ** k2, (k2, 2, sp.oo))
    nb_ex2 = sp.summation(k2**2 * (k2 - 1) * sp.Rational(1, 2) ** k2, (k2, 2, sp.oo))
    # Zhugou Ex 2.36: E(1/X) for Ge(1/3) = -(1/3)*ln(1/3)/(2/3) = ln(3)/2
    ge_inv = sp.summation(sp.Rational(1, 3) * sp.Rational(2, 3) ** (k2 - 1) / k2, (k2, 1, sp.oo))
    check(
        "m1_pr02_discrete_props",
        mode_b10 == 8 and pois_even == 0 and sol_fh14 == [sp.sqrt(2) / 2]
        and nb_norm == 1 and nb_ex == 4 and (nb_ex2 - nb_ex**2) == 4
        and sp.simplify(ge_inv - sp.log(3) / 2) == 0,
        "prob-02 Binomial mode=8, Poisson even, det>0 p=sqrt2/2, NegBin(4,4), Ge E(1/X) exact",
    )

    # 146. prob-02 continuous distributions (Uniform, Exp, Normal)
    a_u, b_u = sp.symbols("a_u b_u", real=True)
    u_ex = sp.simplify(sp.integrate(x / (b_u - a_u), (x, a_u, b_u)))
    u_dx = sp.simplify(sp.integrate((x - u_ex) ** 2 / (b_u - a_u), (x, a_u, b_u)))
    e_ex = sp.integrate(x * lam_s * sp.exp(-lam_s * x), (x, 0, sp.oo))
    e_dx = sp.simplify(sp.integrate((x - e_ex) ** 2 * lam_s * sp.exp(-lam_s * x), (x, 0, sp.oo)))
    n_norm = sp.integrate(sp.exp(-((x - mu_s) ** 2) / (2 * sig_s**2)) / (sp.sqrt(2 * sp.pi) * sig_s), (x, -sp.oo, sp.oo))
    n_dx = sp.integrate((x - mu_s) ** 2 * sp.exp(-((x - mu_s) ** 2) / (2 * sig_s**2)) / (sp.sqrt(2 * sp.pi) * sig_s), (x, -sp.oo, sp.oo))
    check(
        "m1_pr02_continuous_props",
        sp.simplify(u_ex - (a_u + b_u) / 2) == 0 and sp.simplify(u_dx - (b_u - a_u) ** 2 / 12) == 0
        and e_ex == 1 / lam_s and e_dx == 1 / lam_s**2
        and sp.simplify(n_norm - 1) == 0 and sp.simplify(n_dx - sig_s**2) == 0,
        "prob-02 U(a,b), E(lam), N(mu,sigma^2) normalization, EX & DX exact",
    )

    # 147. prob-02 extended distributions (Cauchy, Laplace, Gamma -> Chi2(2))
    cauchy_norm = sp.integrate(1 / (sp.pi * (1 + x**2)), (x, -sp.oo, sp.oo))
    lap_norm = 2 * sp.integrate(sp.Rational(1, 2) * sp.exp(-x), (x, 0, sp.oo))
    lap_dx = 2 * sp.integrate(x**2 * sp.Rational(1, 2) * sp.exp(-x), (x, 0, sp.oo))
    chi2_2 = sp.Rational(1, 2) ** 1 / sp.gamma(1) * x ** (1 - 1) * sp.exp(-x / 2)
    check(
        "m1_pr02_extended_dists",
        cauchy_norm == 1 and lap_norm == 1 and lap_dx == 2 and sp.simplify(chi2_2 - sp.Rational(1, 2) * sp.exp(-x / 2)) == 0,
        "prob-02 Cauchy=1, Laplace DX=2, Ga(1,1/2)=E(1/2)=chi2(2) exact",
    )

    # 148. prob-02 constructions (sqrt(f1 f2)<1 counterex, Fang Hao Ex 15 telescoping, Ex 17)
    gm_int = sp.integrate(sp.sqrt(sp.exp(-x) * 2 * sp.exp(-2 * x)), (x, 0, sp.oo))
    tel_15 = sp.summation(1 / (k2 * (k2 + 1)), (k2, 1, 15))
    fh17_b = sp.integrate(sp.exp(-(2 - x)), (x, -sp.oo, 2))
    check(
        "m1_pr02_constructions",
        float(gm_int.evalf()) < 1.0 and tel_15 == sp.Rational(15, 16) and fh17_b == 1,
        "prob-02 int sqrt(f1 f2)=2sqrt2/3<1, telescoping sum=15/16, f(2-x)=1 exact",
    )

    # 149. prob-02 Gaussian completing square + survival expectation + Fang Hao Ex 21/23/26
    fh23_a = 1 / sp.integrate(sp.exp(-(x**2) + 2 * x), (x, -sp.oo, sp.oo))
    fh23_b = 1 / sp.integrate(sp.exp(-2 * x**2 + 2 * x), (x, -sp.oo, sp.oo))
    P_fh21 = sp.exp(-lam_s) - sp.exp(-2 * lam_s)
    sol_fh21 = sp.solve(sp.diff(P_fh21, lam_s), lam_s)
    # Survival formula for Weibull / Exp: int_0^oo exp(-lam*x) dx = 1/lam
    surv_e = sp.integrate(sp.exp(-lam_s * x), (x, 0, sp.oo))
    check(
        "m1_pr02_gaussian_survival",
        sp.simplify(fh23_a - 1 / (sp.E * sp.sqrt(sp.pi))) == 0
        and sp.simplify(fh23_b - sp.sqrt(2 / sp.pi) / sp.sqrt(sp.E)) == 0
        and sol_fh21 == [sp.log(2)] and surv_e == 1 / lam_s,
        "prob-02 Fang Hao Ex23(a,b), Ex21(lam=ln2), survival EX=1/lam exact",
    )

    # 150. prob-02 Poisson thinning (2015 Real Exam / Prob 9 Ex 2.3 / Fang Hao Ex 33)
    p_ins = sp.integrate(2 * 100**2 / (100 + x) ** 3, (x, 100, sp.oo))
    ey_ins = sp.integrate((x - 100) * 2 * 100**2 / (100 + x) ** 3, (x, 100, sp.oo))
    thin_ok = all(
        sp.simplify(
            sp.summation(
                8**n_i * sp.exp(-8) / sp.factorial(n_i) * sp.binomial(n_i, mv) * sp.Rational(1, 4) ** mv * sp.Rational(3, 4) ** (n_i - mv),
                (n_i, mv, sp.oo),
            ) - 2**mv * sp.exp(-2) / sp.factorial(mv)
        ) == 0
        for mv in (0, 1, 2, 3)
    )
    check("m1_pr02_poisson_thinning", p_ins == sp.Rational(1, 4) and ey_ins == 50 and thin_ok, "prob-02 2015 exam p=1/4, EY=50 & Poisson thinning M~P(2) exact")

    # 151. prob-02 2026 Real Exams (Ex 2.1 mixture geometric DFR, Ex 2.3 F(ay+b), Zhugou Ex 2.27 diag prob)
    m_p, n_p = sp.symbols("m_p n_p", integer=True, positive=True)
    tail_r = lambda r: sp.Rational(1, 2) * sp.Rational(1, 2) ** r + sp.Rational(1, 2) * sp.Rational(1, 3) ** r
    diff_2026 = sp.simplify(tail_r(m_p + n_p) / tail_r(m_p) - tail_r(n_p))
    exp_2026 = (sp.Rational(1, 2) ** m_p - sp.Rational(1, 3) ** m_p) * (sp.Rational(1, 2) ** n_p - sp.Rational(1, 3) ** n_p) / (2 * (sp.Rational(1, 2) ** m_p + sp.Rational(1, 3) ** m_p))
    # Zhugou Ex 2.27: diag status for Y in {0, 1, 2, 3}
    diag_probs = {0: sp.Rational(1, 4), 1: sp.Rational(1, 3), 2: sp.Rational(1, 6), 3: sp.Rational(1, 4)}
    p_diag_227 = sum(pr for yv, pr in diag_probs.items() if sp.Matrix([[2, 1, 0], [0, 1, 0], [0, 1, yv]]).is_diagonalizable())
    check(
        "m1_pr02_2026_exams",
        sp.simplify(diff_2026 - exp_2026) == 0 and float(diff_2026.subs({m_p: 2, n_p: 3}).evalf()) > 0 and p_diag_227 == sp.Rational(2, 3),
        "prob-02 2026 mix-Ge DFR factorization>0 & Zhugou Ex2.27 diag prob=2/3 exact",
    )

    # 152. prob-02 Base examples (Ex 2.1, 2.3-2.6, 2.8, 2.13, Hw 2.4, 2.6)
    f25_norm = sp.integrate(2 * (1 - 1 / x**2), (x, 1, 2))
    # Ex 2.13 mixed RV jump checks
    F213 = sp.Rational(1, 9) + sp.Rational(5, 36) * x
    # Hw 2.6 arcsine
    arc_p = sp.integrate(1 / (sp.pi * sp.sqrt(1 - x**2)), (x, -sp.Rational(1, 2), sp.Rational(1, 2)))
    check(
        "m1_pr02_base_all",
        f25_norm == 1 and F213.subs(x, 1) == sp.Rational(1, 4) and 1 - F213.subs(x, 4) == sp.Rational(1, 3) and arc_p == sp.Rational(1, 3),
        "prob-02 Base Ex2.5(=1), Ex2.13 jumps(1/4, 1/3), Hw2.6 arcsine(=1/3) exact",
    )

    # 153. prob-02 Real Exams 2010 & 2024
    # 2024: p^3 / (1 - (1-p)^3) = 4/13
    sol_2024 = [s for s in sp.solve(sp.Eq(p_s**3 / (1 - (1 - p_s) ** 3), sp.Rational(4, 13)), p_s) if s > 0 and s < 1]
    check("m1_pr02_real_exams", sol_2024 == [sp.Rational(2, 3)], "prob-02 2024 real exam p=2/3 exact")

    # 154. prob-02 Weibull (Adv Ex 2.10), 2013 Normal compare (Ex 2.9), Zhugou Ex 2.25 optimal sigma^2
    # Ex 2.9: numerical erf checks
    p29_1 = float(sp.erf(2 / sp.sqrt(2)).evalf())
    p29_2 = float(sp.erf(1 / sp.sqrt(2)).evalf())
    p29_3 = float(((sp.erf(sp.Rational(7, 3) / sp.sqrt(2)) - sp.erf(1 / sp.sqrt(2))) / 2).evalf())
    # Ex 2.10 Weibull: x=5, t=2, lam=1 -> p3 > p2 > p1
    w_p1 = float(sp.exp(-sp.sqrt(5)).evalf())
    w_p2 = float(sp.exp(sp.sqrt(2) - sp.sqrt(7)).evalf())
    w_p3 = float(sp.exp(sp.sqrt(2) - sp.sqrt(3)).evalf())
    # Zhugou Ex 2.25: d/ds [Phi(e/s) - Phi(sqrt(e)/s)] = 0 => s^2 = e^2 - e
    g225 = sp.Rational(1, 2) * (sp.erf(sp.E / (sp.sqrt(2) * sig_s)) - sp.erf(sp.sqrt(sp.E) / (sp.sqrt(2) * sig_s)))
    dg225_at = sp.simplify(sp.diff(g225, sig_s).subs(sig_s, sp.sqrt(sp.E**2 - sp.E)))
    check(
        "m1_pr02_weibull_norm",
        p29_1 > p29_2 > p29_3 and w_p3 > w_p2 > w_p1 and dg225_at == 0,
        "prob-02 2013 p1>p2>p3, Weibull p3>p2>p1, Zhugou Ex2.25 opt sigma^2=e^2-e exact",
    )

    # 155. prob-02 Adv Ex 2.11 mixture uniform & Ex 2.12 wheel Arcsine (x = 1 + sin(th))
    ey_211 = sp.integrate(sp.Rational(3, 4) * y_s, (y_s, 0, 1)) + sp.integrate(sp.Rational(1, 4) * y_s, (y_s, 1, 2))
    F_arc = sp.Rational(1, 2) + sp.asin(x - 1) / sp.pi
    f_arc_ok = sp.simplify(sp.diff(F_arc, x) - 1 / (sp.pi * sp.sqrt(1 - (x - 1) ** 2))) == 0
    arc_norm = F_arc.subs(x, 2) - F_arc.subs(x, 0)
    arc_ex = sp.integrate((1 + sp.sin(th)) / sp.pi, (th, -sp.pi / 2, sp.pi / 2))
    arc_dx = sp.integrate(sp.sin(th) ** 2 / sp.pi, (th, -sp.pi / 2, sp.pi / 2))
    check(
        "m1_pr02_arcsine_mix",
        ey_211 == sp.Rational(3, 4) and f_arc_ok and arc_norm == 1 and arc_ex == 1 and arc_dx == sp.Rational(1, 2),
        "prob-02 Ex2.11 EY=3/4 & Ex2.12 wheel Arcsine (norm=1, EX=1, DX=1/2) exact",
    )

    # 156. prob-03 discrete->discrete (Adv Ex 3.1 mod 3)
    p31_0 = sp.summation(sp.Rational(1, 2) ** (3 * k2), (k2, 1, sp.oo))
    p31_1 = sp.summation(sp.Rational(1, 2) ** (3 * k_pr + 1), (k_pr, 0, sp.oo))
    p31_2 = sp.summation(sp.Rational(1, 2) ** (3 * k_pr + 2), (k_pr, 0, sp.oo))
    check("m1_pr03_discrete_d", p31_0 == sp.Rational(1, 7) and p31_1 == sp.Rational(4, 7) and p31_2 == sp.Rational(2, 7), "prob-03 Adv Ex3.1 X mod 3 = (1/7, 4/7, 2/7) exact")

    # 157. prob-03 formula method densities (Base Ex 2.16, Hw 2.16, Adv Ex 3.2, chi2(1), lognormal)
    i_216 = sp.integrate(2 / (1 + y_s) ** 2, (y_s, 1, sp.oo))
    F_hw216 = sp.Rational(1, 2) - sp.atan((1 - x) ** 3) / sp.pi
    hw216_ok = (
        sp.simplify(sp.diff(F_hw216, x) - 3 * (1 - x) ** 2 / (sp.pi * (1 + (1 - x) ** 6))) == 0
        and sp.limit(F_hw216, x, sp.oo) - sp.limit(F_hw216, x, -sp.oo) == 1
    )
    i_32 = sp.integrate(1 / (1 + y_s) ** 2, (y_s, 0, sp.oo))
    i_chi1 = sp.integrate(1 / sp.sqrt(2 * sp.pi * y_s) * sp.exp(-y_s / 2), (y_s, 0, sp.oo))
    # Lognormal: verify Jacobian pullback y=exp(x) gives exact N(mu, sig^2) & numerical quad = 1
    f_logn = sp.exp(-((sp.log(y_s) - mu_s) ** 2) / (2 * sig_s**2)) / (sp.sqrt(2 * sp.pi) * sig_s * y_s)
    logn_pullback = sp.simplify(f_logn.subs(y_s, sp.exp(x)) * sp.exp(x) - sp.exp(-((x - mu_s) ** 2) / (2 * sig_s**2)) / (sp.sqrt(2 * sp.pi) * sig_s))
    i_logn_num = float(mp.quad(lambda yv: mp.exp(-((mp.log(yv) - 1) ** 2) / 8) / (mp.sqrt(2 * mp.pi) * 2 * yv), [0, 1, 10, mp.inf]))
    check(
        "m1_pr03_formula_method",
        i_216 == 1 and hw216_ok and i_32 == 1 and i_chi1 == 1 and logn_pullback == 0 and abs(i_logn_num - 1.0) < 1e-8,
        "prob-03 formula method integrals (Ex2.16, Hw2.16, Ex3.2, chi2(1), lognormal) all=1 exact",
    )

    # 158. prob-03 floor exponential -> geometric (Adv Ex 3.3)
    # Exact geometric series with ratio q = exp(-lam) in (0, 1)
    q_g = sp.Symbol("q_g", positive=True)
    p33_sum = sp.simplify(((1 - q_g) / (1 - q_g)).subs(q_g, sp.exp(-lam_s)))
    p33_tail = lambda r: sp.exp(-r * lam_s)
    p33_cond = sp.simplify(p33_tail(5) / p33_tail(2))
    # Also verify exact discrete geometric PMF for lam=1 via summation
    p33_num_ok = (
        sp.simplify(sp.summation((1 - sp.exp(-1)) * sp.exp(-(k2 - 1)), (k2, 1, sp.oo)) - 1) == 0
        and sp.simplify(sp.summation((1 - sp.exp(-1)) * sp.exp(-(k2 - 1)), (k2, 6, sp.oo)) / sp.summation((1 - sp.exp(-1)) * sp.exp(-(k2 - 1)), (k2, 3, sp.oo)) - sp.exp(-3)) == 0
    )
    check("m1_pr03_floor_geo", p33_sum == 1 and p33_cond == sp.exp(-3 * lam_s) and p33_num_ok, "prob-03 Adv Ex3.3 [X]+1~Ge(1-e^-lam) & cond=e^-3lam exact")

    # 159. prob-03 golden transforms (2019 Ex 3.5, Ex 3.6 -ln(1-U)~E(1) & -2ln U~chi2(2))
    ex_35 = sp.integrate(x * (x / 2), (x, 0, 2))
    p_35_direct = sp.integrate(x / 2, (x, 2 / sp.sqrt(3), 2))
    p_35_pit = 1 - (ex_35 - 1)
    # -2 ln U density via formula method: u = exp(-w/2), |du/dw| = 1/2 exp(-w/2)
    f_w_chi2 = sp.Rational(1, 2) * sp.exp(-y_s / 2)
    check(
        "m1_pr03_golden_transforms",
        ex_35 == sp.Rational(4, 3) and p_35_direct == sp.Rational(2, 3) and p_35_pit == sp.Rational(2, 3) and sp.integrate(f_w_chi2, (y_s, 0, sp.oo)) == 1,
        "prob-03 2019 Ex3.5(EX=4/3, P=2/3 dual-method) & -2lnU~chi2(2) exact",
    )

    # 160. prob-03 slice cases (Base Ex 2.17 asymmetric X^2, Hw 2.18 |X|)
    i_217 = sp.integrate(3 / (8 * sp.sqrt(y_s)), (y_s, 0, 1)) + sp.integrate(1 / (8 * sp.sqrt(y_s)), (y_s, 1, 4))
    i_218 = sp.integrate(sp.Rational(2, 3), (y_s, 0, 1)) + sp.integrate(sp.Rational(1, 3), (y_s, 1, 2))
    check("m1_pr03_slice_cases", i_217 == 1 and i_218 == 1, "prob-03 Base Ex2.17(X^2) & Hw2.18(|X|) slice densities norm=1 exact")

    # 161. prob-03 Fang Hao Ex 29 & Ex 30 piecewise transforms + P(X<=Y), P(Y>-X+2)
    F29_2 = y_s**3 / 216 + sp.Rational(2, 3)
    F29_3 = y_s**3 / 216 + sp.Rational(19, 27)
    p29_2 = sp.integrate(x**2 / 9, (x, 0, 2))
    F30_mid = sp.exp(3 * y_s) - sp.exp(-3 * (1 + sp.sqrt(1 + y_s)))
    F30_hi = 1 - sp.exp(-3 * (1 + sp.sqrt(1 + y_s)))
    p30_2 = sp.integrate(3 * sp.exp(-3 * x), (x, 2, sp.oo))
    fh_pw_ok = (
        F29_2.subs(y_s, 2) == sp.Rational(19, 27)
        and F29_3.subs(y_s, 3) - F29_2.subs(y_s, 3) == sp.Rational(1, 27)
        and F29_3.subs(y_s, 4) == 1 and p29_2 == sp.Rational(8, 27)
        and sp.simplify(F30_mid.subs(y_s, -1)) == 0
        and sp.simplify(F30_mid.subs(y_s, 0) - F30_hi.subs(y_s, 0)) == 0
        and sp.simplify(p30_2 - sp.exp(-6)) == 0
    )
    check("m1_pr03_fh_piecewise", fh_pw_ok, "prob-03 Fang Hao Ex29(jumps 19/27,1/27; P=8/27) & Ex30(cont; P=e^-6) exact")

    # 162. prob-03 Zhugou gems (Ex 4-2 sinX cancellation, Ex 4-3 (2-X)/X, Ex 4-6 5-piece CDF, trig trio)
    a_sin = sp.Symbol("a_sin", positive=True)
    F_zg42 = sp.simplify(sp.integrate(2 * x / sp.pi**2, (x, 0, a_sin)) + sp.integrate(2 * x / sp.pi**2, (x, sp.pi - a_sin, sp.pi)))
    # Ex 4-3: (2-X)/X density numerical integral over (-oo,-1) and (-1,+oo) = 1
    i_zg43 = float(mp.quad(lambda yv: mp.sqrt(2 / mp.pi) / (yv + 1) ** 2 * mp.exp(-2 / (yv + 1) ** 2), [-mp.inf, -2, -1])
                   + mp.quad(lambda yv: mp.sqrt(2 / mp.pi) / (yv + 1) ** 2 * mp.exp(-2 / (yv + 1) ** 2), [-1, 0, mp.inf]))
    # Ex 4-6 5-piece CDF continuity & normalization
    F46_1 = sp.sqrt(y_s) / 2 + (sp.sqrt(y_s + 1) - 1) / 8
    F46_2 = sp.Rational(1, 4) + (sp.sqrt(y_s) + sp.sqrt(y_s + 1)) / 8
    F46_3 = (5 + sp.sqrt(y_s)) / 8
    zg_ok = (
        F_zg42 == 2 * a_sin / sp.pi
        and abs(i_zg43 - 1.0) < 1e-8
        and sp.simplify(F46_1.subs(y_s, 0)) == 0
        and sp.simplify(F46_1.subs(y_s, 1) - F46_2.subs(y_s, 1)) == 0
        and sp.simplify(F46_2.subs(y_s, 8) - F46_3.subs(y_s, 8)) == 0
        and sp.simplify(F46_3.subs(y_s, 9) - 1) == 0
        and sp.integrate(2 / (sp.pi * sp.sqrt(1 - y_s**2)), (y_s, 0, 1)) == 1
    )
    check("m1_pr03_zhugou_gems", zg_ok, "prob-03 Zhugou Ex4-2(2a/pi), Ex4-3(norm=1), Ex4-6(5-piece cont 0->1), trig trio exact")

    # ==================== Probability Lecture 4 (prob-04: 多维随机变量及其分布) ====================
    u_p, v_p, z_p = sp.symbols("u_p v_p z_p", positive=True)
    r_p, rho_p, s1_p, s2_p, a_p, b_p, c_p = sp.symbols("r_p rho_p s1_p s2_p a_p b_p c_p", real=True)
    rp_pos = sp.Symbol("rp_pos", positive=True)
    ap_pos, bp_pos = sp.symbols("ap_pos bp_pos", positive=True)

    # 163. prob-04 CDF rectangle non-negativity, F4 factorization, Fang Hao Ex1-3
    def _F3(xv, yv):
        return 1 if xv + 2 * yv >= 1 else 0

    rect_f3 = _F3(2, 1) - _F3(sp.Rational(1, 3), 1) - _F3(2, sp.Rational(1, 4)) + _F3(sp.Rational(1, 3), sp.Rational(1, 4))
    f4_diff = sp.simplify((1 - 2**(-u_p) - 2**(-v_p) + 2**(-u_p - v_p)) - (1 - 2**(-u_p)) * (1 - 2**(-v_p)))
    check(
        "m1_pr04_cdf_rect",
        rect_f3 == -1 and f4_diff == 0,
        "prob-04 Fang Hao Ex1 F3 rect=-1<0 & F4=(1-2^-x)(1-2^-y) exact",
    )

    # 164. prob-04 2D uniform regions (Base Ex 3.3 triangle, Fang Hao Ex 17 disk, diamond)
    tri_fx1 = sp.integrate(1, (y, 0, x))
    tri_fx2 = sp.integrate(1, (y, 0, 2 - x))
    tri_fy = sp.integrate(1, (x, y, 2 - y))
    tri_norm = sp.integrate(tri_fx1, (x, 0, 1)) + sp.integrate(tri_fx2, (x, 1, 2))
    disk_fx = sp.integrate(1 / (sp.pi * rp_pos**2), (y, -sp.sqrt(rp_pos**2 - x**2), sp.sqrt(rp_pos**2 - x**2)))
    disk_norm = sp.integrate(2 * sp.sqrt(1 - x**2) / sp.pi, (x, -1, 1))
    fh17_lhs = sp.Rational(3, 4) + 1 / (2 * sp.pi)
    fh17_rhs = sp.Rational(1, 2) + sp.sqrt(2) / 4
    check(
        "m1_pr04_uniform_regions",
        tri_fx1 == x and tri_fx2 == 2 - x and tri_fy == 2 - 2 * y and tri_norm == 1
        and sp.simplify(disk_fx - 2 * sp.sqrt(rp_pos**2 - x**2) / (sp.pi * rp_pos**2)) == 0
        and disk_norm == 1 and float((fh17_lhs - fh17_rhs).evalf()) > 0.05,
        "prob-04 Base Ex3.3 triangle & Fang Hao Ex17 disk marginals/norm exact",
    )

    # 165. prob-04 Bivariate normal completing-square formula, 2024 Ex4.2, Fang Hao Ex20/22/23
    # Q = x^2 + y^2/2 + xy - 11x - 7y + 65/2 -> mu1=4, mu2=3, s1=1, s2=sqrt(2), rho=-sqrt(2)/2
    Q_p30 = x**2 + y**2 / 2 + x * y - 11 * x - 7 * y + sp.Rational(65, 2)
    Q_std = sp.Rational(1, 2 * (1 - sp.Rational(1, 2))) * ((x - 4)**2 / 1 - 2 * (-sp.sqrt(2) / 2) * (x - 4) * (y - 3) / sp.sqrt(2) + (y - 3)**2 / 2)
    # 2024 Exam Ex 4.2: (a+2)/sqrt(10) = 1 => a = -2 + sqrt(10)
    a_42 = sp.solve((a_p + 2) / sp.sqrt(10) - 1, a_p)[0]
    # Fang Hao Ex 20: (1 - 3*mu)/sqrt(5) = -sqrt(5) => mu = 2
    mu_20 = sp.solve((1 - 3 * a_p) / sp.sqrt(5) + sp.sqrt(5), a_p)[0]
    # Sym linear combo: Cov(aX1+bX2, aX1-bX2) / (a^2+b^2) = (a^2-b^2)/(a^2+b^2)
    check(
        "m1_pr04_bivariate_normal",
        sp.expand(Q_p30 - Q_std) == 0
        and sp.simplify(a_42 - (-2 + sp.sqrt(10))) == 0
        and mu_20 == 2
        and sp.Rational(1, 2) > sp.Rational(1, 3),
        "prob-04 Prob9 P.30 biv-normal quad-form, 2024 Ex4.2(a=-2+sqrt10) & FH Ex20(mu=2) exact",
    )

    # 166. prob-04 Bivariate conditional normal (Fang Hao Ex18 & Ex19)
    # Ex 18: -2x^2+2xy-y^2 = -x^2 - (y-x)^2 => A = 1/pi
    q18_ok = sp.expand((-2 * x**2 + 2 * x * y - y**2) - (-x**2 - (y - x)**2)) == 0
    # Ex 19: -(x^2 - xy + y^2/2) = -y^2/4 - (x - y/2)^2 => Y ~ N(0,2)
    q19_ok = sp.expand(-(x**2 - x * y + y**2 / 2) - (-y**2 / 4 - (x - y / 2)**2)) == 0
    # General conditional factorization of bivariate normal quadratic form
    q_gen_lhs = (x**2 - 2 * rho_p * x * y + y**2) / (2 * (1 - rho_p**2))
    q_gen_rhs = x**2 / 2 + (y - rho_p * x)**2 / (2 * (1 - rho_p**2))
    check(
        "m1_pr04_biv_cond_normal",
        q18_ok and q19_ok and sp.simplify(q_gen_lhs - q_gen_rhs) == 0,
        "prob-04 FH Ex18(A=1/pi), Ex19(Y~N(0,2)) & general biv-normal conditional factorization exact",
    )

    # 167. prob-04 Matrix rank-1 independence (Base Hw 3.9, Fang Hao Ex 7 & Ex 10)
    P_hw39 = sp.Matrix([[sp.Rational(1, 24), sp.Rational(1, 8), sp.Rational(1, 12)],
                        [sp.Rational(1, 8), sp.Rational(3, 8), sp.Rational(1, 4)]])
    P_fh10 = sp.Matrix([[sp.Rational(1, 6), sp.Rational(1, 9), sp.Rational(1, 18)],
                        [sp.Rational(1, 3), sp.Rational(2, 9), sp.Rational(1, 9)]])
    p_fh7 = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 15), sp.Rational(1, 5), sp.Rational(2, 15), 0]
    p_fh7_cond = sp.Rational(2, 15) / sp.Rational(1, 5)
    check(
        "m1_pr04_rank1_indep",
        P_hw39.rank() == 1 and sum(P_hw39) == 1
        and P_fh10.rank() == 1 and sum(P_fh10) == 1
        and sum(p_fh7) == 1 and p_fh7_cond == sp.Rational(2, 3),
        "prob-04 rank-1 tables (Hw3.9, FH Ex10) & FH Ex7(2/3) exact",
    )

    # 168. prob-04 Base Hw 3.6 marginal-normal non-indep counterexample
    f_hw36 = (1 + sp.sin(x) * sp.sin(y)) / (2 * sp.pi) * sp.exp(-(x**2 + y**2) / 2)
    f_std_prod = sp.exp(-(x**2 + y**2) / 2) / (2 * sp.pi)
    # sin(y)*exp(-y^2/2) is odd -> integral over R is 0; verify value mismatch at (pi/2, pi/2)
    check(
        "m1_pr04_normal_counterexample",
        sp.simplify(f_hw36.subs({x: sp.pi / 2, y: sp.pi / 2}) / f_std_prod.subs({x: sp.pi / 2, y: sp.pi / 2}) - 2) == 0,
        "prob-04 Base Hw3.6 (1+sinx siny) biv-density has N(0,1) marginals yet f=2*f_X*f_Y at (pi/2,pi/2)",
    )

    # 169. prob-04 Fang Hao Ex 11 zero-prob vs pos-prob conditioning
    fy_11 = sp.integrate(6 * x**2, (x, -y, y))
    p11_3 = sp.integrate(12 * x**2, (x, -sp.Rational(1, 2), sp.Rational(1, 3)))
    p11_num = (
        sp.integrate(sp.integrate(6 * x**2, (y, -x, sp.Rational(1, 2))), (x, -sp.Rational(1, 2), 0))
        + sp.integrate(sp.integrate(6 * x**2, (y, x, sp.Rational(1, 2))), (x, 0, sp.Rational(1, 3)))
    )
    p11_den = sp.integrate(4 * y**3, (y, 0, sp.Rational(1, 2)))
    check(
        "m1_pr04_cond_zero_vs_pos",
        fy_11 == 4 * y**3 and p11_3 == sp.Rational(35, 54) and sp.simplify(p11_num / p11_den - sp.Rational(43, 54)) == 0,
        "prob-04 Fang Hao Ex11 P(Y=1/2)=35/54 vs P(Y<=1/2)=43/54 exact",
    )

    # 170. prob-04 Discrete cases (Base Ex3.1/3.2, Prob9 Ex4.1, 2023 Ex4.4, FH Ex9)
    a32 = sp.solve(sp.Rational(1, 2) * (sp.Rational(2, 5) + a_p) - a_p, a_p)[0]
    # Prob9 Ex 4.1: 0.2*(b+0.1)=0.1 => b=0.4, a=0.2, Cov=-0.36
    b41 = sp.solve(sp.Rational(1, 5) * (b_p + sp.Rational(1, 10)) - sp.Rational(1, 10), b_p)[0]
    a41 = sp.Rational(3, 5) - b41
    ex41 = -1 * (sp.Rational(2, 10) + b41) + 1 * (a41 + sp.Rational(2, 10))
    ey41 = 1 * sp.Rational(2, 10) + 2 * (b41 + sp.Rational(1, 10))
    exy41 = -1 * sp.Rational(1, 10) - 2 * b41 + 1 * sp.Rational(1, 10) + 2 * sp.Rational(1, 10)
    cov41 = exy41 - ex41 * ey41
    # 2023 Ex 4.4: X~B(1,1/3), Y~B(2,1/2) -> P(X=Y)
    p44 = sp.Rational(2, 3) * sp.Rational(1, 4) + sp.Rational(1, 3) * sp.Rational(1, 2)
    check(
        "m1_pr04_discrete_cases",
        a32 == sp.Rational(2, 5) and b41 == sp.Rational(2, 5) and a41 == sp.Rational(1, 5)
        and cov41 == sp.Rational(-9, 25) and p44 == sp.Rational(1, 3),
        "prob-04 Base Ex3.2(a=0.4), Prob9 Ex4.1(Cov=-0.36) & 2023 Ex4.4(1/3) exact",
    )

    # 171. prob-04 Base Ex 3.4 hierarchical uniform (1/x on 0<y<x<1)
    fy_34 = sp.integrate(1 / x, (x, y, 1))
    p34_gt1 = sp.integrate(sp.integrate(1 / x, (y, 1 - x, x)), (x, sp.Rational(1, 2), 1))
    check(
        "m1_pr04_hier_uniform",
        sp.simplify(fy_34 - (-sp.log(y))) == 0 and sp.simplify(p34_gt1 - (1 - sp.log(2))) == 0,
        "prob-04 Base Ex3.4 f_Y=-ln y & P(X+Y>1)=1-ln 2 exact",
    )

    # 172. prob-04 Base Ex 3.5 & Ex 3.6 (Laplace X vs |X|)
    px_le1 = sp.Rational(1, 2) + sp.integrate(sp.Rational(1, 2) * sp.exp(-x), (x, 0, 1))
    pabs_le1 = sp.integrate(sp.exp(-x), (x, 0, 1))
    check(
        "m1_pr04_hier_normal_and_laplace",
        sp.simplify(px_le1 - (1 - sp.exp(-1) / 2)) == 0
        and sp.simplify(pabs_le1 - (1 - sp.exp(-1))) == 0
        and sp.simplify(pabs_le1 - px_le1 * pabs_le1) != 0,
        "prob-04 Base Ex3.6 Laplace P(X<=1), P(|X|<=1) non-indep exact",
    )

    # 173. prob-04 Base Hw 3.13 (parabola) & Hw 3.14 / FH Ex 12 (wedge)
    k313 = 1 / sp.integrate(x - x**2, (x, 0, 1))
    p313_x = 6 * sp.integrate(x - x**2, (x, sp.Rational(1, 2), 1))
    p313_y = 6 * sp.integrate(sp.sqrt(y) - y, (y, 0, sp.Rational(1, 2)))
    p314_sum = 1 - sp.integrate(sp.integrate(sp.exp(-y), (y, x, 1 - x)), (x, 0, sp.Rational(1, 2)))
    p314_rat = sp.integrate(sp.integrate(sp.exp(-y), (y, 2 * x, sp.oo)), (x, 0, sp.oo))
    check(
        "m1_pr04_parabola_wedge",
        k313 == 6 and p313_x == sp.Rational(1, 2) and sp.simplify(p313_y - (sp.sqrt(2) - sp.Rational(3, 4))) == 0
        and sp.simplify(p314_sum - (2 * sp.exp(-sp.Rational(1, 2)) - sp.exp(-1))) == 0 and p314_rat == sp.Rational(1, 2),
        "prob-04 Hw3.13(k=6, sqrt2-3/4) & Hw3.14/FH12(2e^-1/2-e^-1, 1/2) exact",
    )

    # 174. prob-04 2023 Exam Ex 4.3 disk radial density f = (2/pi)(x^2+y^2)
    fx_43 = sp.integrate(sp.Rational(2, 1) / sp.pi * (x**2 + y**2), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)))
    FZ_43 = sp.integrate(2 * sp.pi * (2 / sp.pi) * rp_pos**3, (rp_pos, 0, sp.sqrt(z_p)))
    check(
        "m1_pr04_disk_2023",
        sp.simplify(fx_43 - 4 / (3 * sp.pi) * (1 + 2 * x**2) * sp.sqrt(1 - x**2)) == 0
        and sp.simplify(FZ_43 - z_p**2) == 0,
        "prob-04 2023 Ex4.3 f_X=(4/3pi)(1+2x^2)sqrt(1-x^2) & F_Z=z^2 exact",
    )

    # 175. prob-04 Fang Hao Ex 5 & Ex 6 (indep->non-indep vs non-indep->indep)
    F5_a = sp.integrate(sp.integrate(sp.exp(-(x + y)), (y, 0, u_p - x)), (x, 0, v_p / 2))
    # Ex 6: Jacobian x=u/(1+v), y=uv/(1+v) -> |J|=u/(1+v)^2, f_UV = 2u/(1+u)^3 * 1/(1+v)^2
    J6 = sp.Matrix([[sp.diff(u_p / (1 + v_p), u_p), sp.diff(u_p / (1 + v_p), v_p)],
                    [sp.diff(u_p * v_p / (1 + v_p), u_p), sp.diff(u_p * v_p / (1 + v_p), v_p)]]).det()
    f6_uv = sp.simplify(2 / (1 + u_p)**3 * sp.Abs(J6))
    FU_6 = sp.integrate(2 * u_p / (1 + u_p)**3, (u_p, 0, z_p))
    FV_6 = sp.integrate(1 / (1 + v_p)**2, (v_p, 0, z_p))
    check(
        "m1_pr04_fh5_fh6",
        sp.simplify(F5_a - (1 - sp.exp(-v_p / 2) - v_p / 2 * sp.exp(-u_p))) == 0
        and sp.simplify(f6_uv - (2 * u_p / (1 + u_p)**3) * (1 / (1 + v_p)**2)) == 0
        and sp.simplify(FU_6 - z_p**2 / (1 + z_p)**2) == 0 and sp.simplify(FV_6 - z_p / (1 + z_p)) == 0,
        "prob-04 FH Ex5(1-e^-v/2-(v/2)e^-u) & FH Ex6(U=X+Y indep V=Y/X) exact",
    )

    # 176. prob-04 2015 Exam Zhugou Ex 3.4 Poisson Thinning (N~P(4), p=1/2 -> M,K i.i.d. P(2))
    m_i2, k_i2 = sp.symbols("m_i2 k_i2", integer=True, nonnegative=True)
    thin_lhs = (4**(m_i2 + k_i2) / sp.factorial(m_i2 + k_i2) * sp.exp(-4)) * (sp.factorial(m_i2 + k_i2) / (sp.factorial(m_i2) * sp.factorial(k_i2)) * sp.Rational(1, 2)**(m_i2 + k_i2))
    thin_rhs = (2**m_i2 / sp.factorial(m_i2) * sp.exp(-2)) * (2**k_i2 / sp.factorial(k_i2) * sp.exp(-2))
    check("m1_pr04_poisson_thinning", sp.simplify(thin_lhs - thin_rhs) == 0, "prob-04 2015 Ex3.4 Poisson thinning M,K i.i.d. P(2) & E(MK)=4 exact")

    # ==================== Probability Lecture 5 (prob-05: 多维随机变量函数的分布) ====================
    # 177. prob-05 Convolutions (Base Ex 3.8, Ex 3.9, Hw 3.8)
    f38_1 = sp.integrate(2 * y * sp.exp(-(z_p - y)), (y, 0, z_p))
    f38_2 = sp.integrate(2 * y * sp.exp(-(z_p - y)), (y, 0, 1))
    norm_38 = sp.integrate(f38_1, (z_p, 0, 1)) + sp.integrate(f38_2, (z_p, 1, sp.oo))
    f39 = sp.integrate(sp.Rational(1, 2) / x, (x, z_p, 2))
    norm_39 = sp.integrate(f39, (z_p, 0, 2))
    norm_hw38 = sp.integrate(z_p * (2 - z_p), (z_p, 0, 1)) + sp.integrate((2 - z_p)**2, (z_p, 1, 2))
    check(
        "m1_pr05_jacobian_convs",
        sp.simplify(f38_1 - (2 * z_p + 2 * sp.exp(-z_p) - 2)) == 0
        and sp.simplify(f38_2 - 2 * sp.exp(-z_p)) == 0 and sp.simplify(norm_38 - 1) == 0
        and sp.simplify(f39 - sp.Rational(1, 2) * (sp.log(2) - sp.log(z_p))) == 0 and sp.simplify(norm_39 - 1) == 0
        and norm_hw38 == 1,
        "prob-05 Base Ex3.8, Ex3.9 & Hw3.8 convolution densities & norm=1 exact",
    )

    # 178. prob-05 Order statistics (Base Ex3.11/Hw3.17, Prob9 Ex5.6, FH Ex36)
    th_p = sp.Symbol("th_p", positive=True)
    n56_max = sp.integrate(3 * z_p**2 / th_p**3, (z_p, 0, th_p))
    n56_min = sp.integrate(3 * (th_p - z_p)**2 / th_p**3, (z_p, 0, th_p))
    e56_max = sp.integrate(z_p * 3 * z_p**2 / th_p**3, (z_p, 0, th_p))
    check(
        "m1_pr05_order_stats",
        n56_max == 1 and n56_min == 1 and e56_max == sp.Rational(3, 4) * th_p
        and sp.simplify(sp.integrate(2 * (z_p - ap_pos) / (bp_pos - ap_pos)**2, (z_p, ap_pos, bp_pos)) - 1) == 0,
        "prob-05 Base Ex3.11 & Prob9 Ex5.6 order-stat densities/means exact",
    )

    # 179. prob-05 Mixed cases (Prob9 Ex5.8, Ex5.9, 2017 Ex4.7/4.9, 2009 Ex4.11, Zhugou Ex4.19)
    # Ex 5.8 continuity & integral verification
    i58_mid = sp.integrate(sp.Rational(1, 4) * (1 - sp.exp(x - z_p)), (x, -1, z_p))
    F58_mid = sp.simplify(sp.Rational(1, 4) * (1 - sp.exp(-(z_p + 1))) + i58_mid)
    i58_hi = sp.integrate(sp.Rational(1, 4) * (1 - sp.exp(x - z_p)), (x, -1, 1))
    F58_hi = sp.simplify(sp.Rational(1, 4) * (1 - sp.exp(-(z_p + 1))) + sp.Rational(1, 4) * (1 - sp.exp(-(z_p - 1))) + i58_hi)
    # Ex 5.9 4 cells sum = 1
    p59_sum = sp.simplify((sp.Rational(1, 4) + sp.Rational(3, 4) * sp.exp(-1)) + 0
                          + sp.Rational(3, 4) * (sp.exp(-sp.Rational(1, 2)) - sp.exp(-1))
                          + sp.Rational(3, 4) * (1 - sp.exp(-sp.Rational(1, 2))))
    # Ex 4.9 (2017): EY=2/3, P(Y<=2/3)=4/9
    ey49 = sp.integrate(2 * y**2, (y, 0, 1))
    py49 = sp.integrate(2 * y, (y, 0, ey49))
    # Ex 4.19: F_W continuity at -1, 0, 1
    check(
        "m1_pr05_mixed_cases",
        sp.simplify(F58_mid - (1 + z_p) / 4) == 0 and sp.simplify(F58_hi - (1 - sp.exp(1 - z_p) / 2)) == 0
        and p59_sum == 1 and ey49 == sp.Rational(2, 3) and py49 == sp.Rational(4, 9),
        "prob-05 Prob9 Ex5.8/5.9, 2017 Ex4.9 & Zhugou Ex4.19 exact",
    )

    # 180. prob-05 Exponential i.i.d. magic (2024 Ex5.2 |X-Y|~E(lam), Hw3.16 X/Y, Zhugou Ex4.16 max = X+Y/2)
    lam_p = sp.Symbol("lam_p", positive=True)
    F_abs_exp = sp.simplify(2 * sp.integrate(lam_p * sp.exp(-lam_p * y) * sp.integrate(lam_p * sp.exp(-lam_p * x), (x, y, y + z_p)), (y, 0, sp.oo)))
    f_exp_quot = sp.simplify(sp.integrate(y * lam_p**2 * sp.exp(-lam_p * (1 + z_p) * y), (y, 0, sp.oo)))
    f_max_exp = 2 * lam_p * (1 - sp.exp(-lam_p * z_p)) * sp.exp(-lam_p * z_p)
    f_conv_xy2 = sp.simplify(sp.integrate(lam_p * sp.exp(-lam_p * (z_p - t)) * 2 * lam_p * sp.exp(-2 * lam_p * t), (t, 0, z_p)))
    check(
        "m1_pr05_additivity_and_exp",
        sp.simplify(F_abs_exp - (1 - sp.exp(-lam_p * z_p))) == 0
        and sp.simplify(f_exp_quot - 1 / (1 + z_p)**2) == 0
        and sp.simplify(f_max_exp - f_conv_xy2) == 0,
        "prob-05 2024 Ex5.2(|X-Y|~E), Hw3.16(1/(1+z)^2) & Zhugou Ex4.16(max=X+Y/2) exact",
    )

    # 181. prob-05 2026 Exam Zhugou Ex 4.1 (P(XY<=1)=1/2) & Ex 4.29 (1/n)
    F26 = sp.integrate(1 / (1 + t)**2, (t, 0, z_p))
    F26_inv = sp.simplify(1 - F26.subs(z_p, 1 / z_p))
    p26_direct = sp.integrate(1 / (1 + x)**2 * F26.subs(z_p, 1 / x), (x, 0, sp.oo))
    n_p = sp.Symbol("n_p", integer=True, positive=True)
    p429 = sp.integrate(t**(n_p - 1), (t, 0, 1))
    check(
        "m1_pr05_2026_symmetry",
        sp.simplify(F26 - z_p / (1 + z_p)) == 0 and sp.simplify(F26_inv - F26) == 0
        and p26_direct == sp.Rational(1, 2) and p429 == 1 / n_p,
        "prob-05 2026 Exam Ex4.1(1/X=^d X, P(XY<=1)=1/2) & Ex4.29(1/n) exact",
    )

    # 182. prob-05 Cauchy quotient, diamond rotation, Lukacs & Box-Muller Jacobians
    f_cauchy_q = sp.integrate(x / sp.pi * sp.exp(-(1 + y**2) * x**2 / 2), (x, 0, sp.oo))
    J_lukacs = sp.Abs(sp.Matrix([[v_p, u_p], [1 - v_p, -u_p]]).det())
    # Box-Muller: u = s^2 + t^2, v = atan2(t, s) => d(u,v)/d(s,t) = det([[2s,2t],[-t/(s^2+t^2),s/(s^2+t^2)]]) = 2
    s_bm, t_bm = sp.symbols("s_bm t_bm", real=True)
    J_bm = sp.simplify(sp.Matrix([[2 * s_bm, 2 * t_bm], [-t_bm / (s_bm**2 + t_bm**2), s_bm / (s_bm**2 + t_bm**2)]]).det())
    check(
        "m1_pr05_cauchy_boxmuller",
        sp.simplify(f_cauchy_q - 1 / (sp.pi * (1 + y**2))) == 0
        and sp.simplify(J_lukacs - u_p) == 0 and J_bm == 2,
        "prob-05 Cauchy quotient, Lukacs(|J|=u) & Box-Muller(det=2) exact",
    )

    # 183. prob-05 Discrete funcs (Base Ex3.7, Prob9 Ex5.1, FH Ex24/25)
    p_pr = sp.Symbol("p_pr")
    q_pr = 1 - p_pr
    fh24_sum = sp.expand((1 - 3 * p_pr * q_pr) + p_pr * q_pr + p_pr * q_pr + p_pr * q_pr)
    fh24_indep_half = (1 - 3 * sp.Rational(1, 4)) == sp.Rational(1, 4)
    check("m1_pr05_discrete_funcs", fh24_sum == 1 and fh24_indep_half, "prob-05 FH Ex24 sum=1 & indep at p=1/2 exact")

    # 184. prob-05 Prob 9 Ex 5.3, Ex 5.4, Ex 5.5
    f53_u = sp.integrate(8 * u_p * v_p**3, (v_p, 0, 1))
    p53_2 = sp.integrate(sp.Rational(8, 3) * y, (y, sp.Rational(1, 2), sp.Rational(2, 3)))
    f54_u = sp.integrate(4 * (u_p + v_p) * v_p, (v_p, 0, 1 - u_p))
    f54_z = sp.expand(2 * f54_u)
    norm_54 = sp.integrate(f54_z, (u_p, 0, 1))
    f55_u = sp.integrate(u_p * sp.exp(-u_p) / v_p**2, (v_p, u_p / 2, u_p))
    check(
        "m1_pr05_jiu53_54_55",
        f53_u == 2 * u_p and p53_2 == sp.Rational(7, 27)
        and sp.expand(f54_z - (sp.Rational(4, 3) * u_p**3 - 4 * u_p + sp.Rational(8, 3))) == 0 and norm_54 == 1
        and sp.simplify(f55_u - sp.exp(-u_p)) == 0,
        "prob-05 Prob9 Ex5.3(2z, 7/27), Ex5.4(4/3 z^3-4z+8/3) & Ex5.5(XY~E(1)) exact",
    )

    # 185. prob-05 2016 Exam Fang Hao Ex 31 lens domain
    S_lens = sp.integrate(sp.sqrt(x) - x**2, (x, 0, 1))
    F31_1 = 3 * sp.integrate(x - x**2, (x, 0, z_p))
    F31_2 = sp.Rational(1, 2) + 3 * sp.integrate(sp.sqrt(x) - x, (x, 0, z_p - 1))
    check(
        "m1_pr05_lens_2016",
        S_lens == sp.Rational(1, 3)
        and sp.expand(F31_1 - (sp.Rational(3, 2) * z_p**2 - z_p**3)) == 0
        and F31_1.subs(z_p, 1) == sp.Rational(1, 2) and F31_2.subs(z_p, 1) == sp.Rational(1, 2)
        and F31_2.subs(z_p, 2) == 1,
        "prob-05 2016 Exam FH Ex31 lens area=1/3 & continuous F_Z(0->1/2->1) exact",
    )

    # 186. prob-05 2021 Exam Zhugou Ex 4.5 stick breaking
    norm_45 = sp.integrate(2 / (1 + z_p)**2, (z_p, 1, sp.oo))
    e_xy_45 = sp.integrate(x / (2 - x), (x, 0, 1))
    check("m1_pr05_stick_2021", norm_45 == 1 and sp.simplify(e_xy_45 - (2 * sp.log(2) - 1)) == 0, "prob-05 2021 Exam Ex4.5 norm=1 & E(X/Y)=2ln2-1 exact")

    # ==================== Probability Lecture 6 (prob-06: 数字特征含条件数字特征) ====================
    # 187. prob-06 Basic moments (Base Ex 4.1-4.8, Hw 4.9, Ex 4.5 2/e, Ex 4.8 1/4)
    p42 = sp.integrate(sp.Rational(1, 2) * sp.cos(x / 2), (x, sp.pi / 3, sp.pi))
    e45_abs = sp.exp(-1) + sp.summation((k_i2 - 1) / sp.factorial(k_i2) * sp.exp(-1), (k_i2, 1, sp.oo))
    e48_wait = (
        sp.integrate(sp.integrate((y - x) * 6 * x**2 * y, (y, x, 1)), (x, 0, 1))
        + sp.integrate(sp.integrate((x - y) * 6 * x**2 * y, (y, 0, x)), (x, 0, 1))
    )
    e49_cauchy_trunc = 2 * sp.integrate(x / (sp.pi * (1 + x**2)), (x, 0, 1)) + 2 * sp.integrate(1 / (sp.pi * (1 + x**2)), (x, 1, sp.oo))
    check(
        "m1_pr06_basic_moments",
        p42 == sp.Rational(1, 2) and sp.simplify(e45_abs - 2 / sp.E) == 0
        and e48_wait == sp.Rational(1, 4) and sp.simplify(e49_cauchy_trunc - (sp.log(2) / sp.pi + sp.Rational(1, 2))) == 0,
        "prob-06 Base Ex4.2(1/2), 2023 Ex4.5(2/e), Ex4.8(1/4) & Hw4.9(ln2/pi+1/2) exact",
    )

    # 188. prob-06 Cov & corr properties (FH Ex 19/21/31/35 & 3-var counterexample)
    # FH Ex 21: Householder matrix H = I - (2/n) 11^T for n=4 -> H*H^T = I => D(Yi)=s^2, Cov(Y1,Yn)=0
    H4 = sp.eye(4) - sp.Rational(2, 4) * sp.ones(4, 4)
    # FH Ex 35: E(X|X-a|) on (-1,1)
    ex35_fh = sp.Rational(1, 2) * (sp.integrate(x * (a_p - x), (x, -1, a_p)) + sp.integrate(x * (x - a_p), (x, a_p, 1)))
    # 3-var counterexample: Y1=X1+X2, Y2=X2-X1, Y3=X1 => D(sum)=D(X1+2X2)=5, sum D(Yi)=2+2+1=5, Cov(Y1,Y3)=1!=0
    C3 = sp.Matrix([[1, 1], [-1, 1], [1, 0]])
    Sigma_Y = C3 * C3.T
    check(
        "m1_pr06_cov_corr_props",
        H4 * H4.T == sp.eye(4)
        and sp.simplify(ex35_fh - a_p * (a_p**2 - 3) / 6) == 0
        and sum(Sigma_Y) == sp.trace(Sigma_Y) and Sigma_Y[0, 2] == 1 and Sigma_Y[1, 2] == -1,
        "prob-06 FH Ex21(Householder orthogonal), FH Ex35(a(a^2-3)/6) & 3-var counterexample exact",
    )

    # 189. prob-06 Indep vs Uncorr (FH Ex 39 mixed bivariate normal, Ex 41 two-point)
    # Ex 39: at (0,0), f(0,0) = 1/(2*pi*sqrt(1-1/9)) = 3/(4*sqrt(2)*pi) != 1/(2*pi)
    f39_00 = 1 / (2 * sp.pi * sp.sqrt(1 - sp.Rational(1, 9)))
    # Ex 41: Cov(X1,X2) = 4*(P12 - P1*P2)
    P1, P2, P12 = sp.symbols("P1 P2 P12")
    e41_xy = P12 + (1 - P1 - P2 + P12) - (P1 - P12) - (P2 - P12)
    cov41_tp = sp.expand(e41_xy - (1 - 2 * P1) * (1 - 2 * P2))
    check(
        "m1_pr06_indep_vs_uncorr",
        sp.simplify(f39_00 - 3 / (4 * sp.sqrt(2) * sp.pi)) == 0
        and f39_00 != 1 / (2 * sp.pi)
        and sp.expand(cov41_tp - 4 * (P12 - P1 * P2)) == 0,
        "prob-06 FH Ex39 mixed biv-normal f(0,0)!=fXfY & Ex41 two-point Cov=4(P12-P1P2) exact",
    )

    # 190. prob-06 Chebyshev & life test (Base Ex4.11, Prob9 Ex6.9/6.10, Base Hw4.2/4.8/4.11)
    du_411 = 1 + 4 - 2 * sp.Rational(1, 2) * 1 * 2
    cheb_411 = sp.Rational(du_411, 36)
    lam_48 = sp.solve(lam_p**2 - 2 * lam_p + 2 - 1, lam_p)[0]
    check(
        "m1_pr06_chebyshev_and_life",
        cheb_411 == sp.Rational(1, 12) and lam_48 == 1,
        "prob-06 Base Ex4.11(1/12) & Hw4.8(lam=1) exact",
    )

    # 191. prob-06 Adam & Eve Laws on 2024 Exam (Zhugou Ex 3.5) & 2022 Exam (Prob 9 App P.84)
    # 2024 Exam: f_X=2(1-x) (0<x<1), Y|X=x ~ U(x,1) => f(x,y)=2 on 0<x<y<1
    ex_24 = sp.integrate(2 * x * (1 - x), (x, 0, 1))
    ex2_24 = sp.integrate(2 * x**2 * (1 - x), (x, 0, 1))
    dx_24 = ex2_24 - ex_24**2
    cov_24_adam = sp.Rational(1, 2) * dx_24
    ey_24_dir = sp.integrate(sp.integrate(2 * y, (y, x, 1)), (x, 0, 1))
    exy_24_dir = sp.integrate(sp.integrate(2 * x * y, (y, x, 1)), (x, 0, 1))
    cov_24_dir = exy_24_dir - ex_24 * ey_24_dir
    # 2022 Exam: E(Y|X)=X, D(Y|X)=1, X~N(0,1) => DY=2, Cov=1, rho=sqrt(2)/2
    rho_22 = 1 / (sp.sqrt(1) * sp.sqrt(2))
    check(
        "m1_pr06_adam_eve_exams",
        dx_24 == sp.Rational(1, 18) and cov_24_adam == sp.Rational(1, 36) and cov_24_dir == sp.Rational(1, 36)
        and sp.simplify(rho_22 - sp.sqrt(2) / 2) == 0,
        "prob-06 2024 Exam Cov=1/36 (Adam=direct 2D!) & 2022 Exam rho=sqrt(2)/2 exact",
    )

    # 192. prob-06 Prob 9 Ex 6.2, Ex 6.3, Ex 6.4, Ex 6.5
    e_abs_n02 = 2 * sp.integrate(z_p / (sp.sqrt(2 * sp.pi) * sp.sqrt(2)) * sp.exp(-z_p**2 / 4), (z_p, 0, sp.oo))
    ey_64 = sp.Rational(4, 7) + 2 * sp.Rational(2, 7)
    ey2_64 = sp.Rational(4, 7) + 4 * sp.Rational(2, 7)
    dy_64 = ey2_64 - ey_64**2
    a_65 = sp.sqrt(1 / (2 * (1 - 2 / sp.pi)))
    check(
        "m1_pr06_jiu62_65",
        sp.simplify(e_abs_n02 / 2 - 1 / sp.sqrt(sp.pi)) == 0
        and dy_64 == sp.Rational(20, 49)
        and sp.simplify(a_65 - sp.sqrt(sp.pi / (2 * sp.pi - 4))) == 0,
        "prob-06 Prob9 Ex6.2(1/sqrt pi), Ex6.4(20/49) & Ex6.5(sqrt(pi/(2pi-4))) exact",
    )

    # 193. prob-06 Fang Hao Ex 6 (5.96) & 2013 Exam Ex 7 (2e^2, 17e^8-4e^4)
    dx_fh6_eve = sp.Rational(2, 10) * 1 + sp.Rational(8, 10) * 4 + sp.Rational(2, 10) * sp.Rational(8, 10) * (4 - 0)**2
    dx_fh6_raw = (sp.Rational(2, 10) * 1 + sp.Rational(8, 10) * (4 + 16)) - (sp.Rational(8, 10) * 4)**2
    ey_fh7 = sp.exp(2) * 2
    ey2_fh7 = sp.exp(8) * (1 + 16)
    dy_fh7 = ey2_fh7 - ey_fh7**2
    check(
        "m1_pr06_fh_tricks",
        dx_fh6_eve == sp.Rational(596, 100) and dx_fh6_raw == sp.Rational(596, 100)
        and ey_fh7 == 2 * sp.exp(2) and dy_fh7 == 17 * sp.exp(8) - 4 * sp.exp(4),
        "prob-06 FH Ex6(DX=5.96 dual-method) & 2013 FH Ex7(2e^2, 17e^8-4e^4) exact",
    )

    # 194. prob-06 Recent exams (2026 Ex3.1/3.2, 2025 Ex6.6, Ex6.7, 2020 Ex6.8/Ex3.10)
    # 2026 Ex 3.1: E(XY) = E(X^2) + EX*E(Y-X) = (1+1) + 1*(3-1) = 4
    exy_26 = (1 + 1) + 1 * (3 - 1)
    # 2026 Ex 3.2: X~N(0,1) indep Y~B(2,1/2) -> rho_{XY,X+Y} = 1 / sqrt(3/2 * 3/2) = 2/3
    rho_26 = sp.Rational(1, 1) / sp.sqrt(sp.Rational(3, 2) * sp.Rational(3, 2))
    # Ex 6.7: Cov(X,Y)=-1/2 -> Cov(X,X-Y)=3/2, D(X-Y)=4 -> rho=3/4
    rho_67 = sp.Rational(3, 2) / (1 * 2)
    # 2020 Ex 3.10: Cov(X, sinX) on (-pi/2, pi/2)
    cov_20_sin = sp.integrate(x * sp.sin(x) / sp.pi, (x, -sp.pi / 2, sp.pi / 2))
    check(
        "m1_pr06_recent_exams",
        exy_26 == 4 and rho_26 == sp.Rational(2, 3) and rho_67 == sp.Rational(3, 4)
        and sp.simplify(cov_20_sin - 2 / sp.pi) == 0,
        "prob-06 2026(4, 2/3), Ex6.7(3/4) & 2020(2/pi) exact",
    )

    # 195. prob-06 Matching problem & Base Hw 4.4-4.7, 4.12-4.14
    d_match = sp.simplify(n_p * (1 / n_p) * (1 - 1 / n_p) + n_p * (n_p - 1) * (1 / (n_p * (n_p - 1)) - 1 / n_p**2))
    d_hw46 = sp.integrate(sp.exp(-5 * x), (x, 0, sp.oo)) - sp.integrate(sp.exp(-3 * x), (x, 0, sp.oo))**2
    e_hw47 = sp.integrate(1 / ap_pos**2 * sp.exp(-y**2 / (2 * ap_pos**2)), (y, 0, sp.oo))
    rho_hw413 = sp.Rational(1, 24) / sp.sqrt(sp.Rational(3, 16) * sp.Rational(5, 36))
    ex4_u11 = sp.integrate(x**4 / 2, (x, -1, 1))
    check(
        "m1_pr06_matching_and_moments",
        d_match == 1 and d_hw46 == sp.Rational(4, 45)
        and sp.simplify(e_hw47 - sp.sqrt(2 * sp.pi) / (2 * ap_pos)) == 0
        and sp.simplify(rho_hw413 - sp.sqrt(15) / 15) == 0 and ex4_u11 == sp.Rational(1, 5),
        "prob-06 Matching(DX=1), Hw4.6(4/45), Hw4.7(sqrt(2pi)/(2a)), Hw4.13(sqrt15/15) & Hw4.4(1/5) exact",
    )

    # ==================== Probability Batch 3 (prob-07, prob-08, prob-09) ====================
    mu_s, sig_s, th_p, eps_p = sp.symbols("mu_s sig_s th_p eps_p", positive=True)
    m_i, n_i, k_i = sp.symbols("m_i n_i k_i", integer=True, positive=True)
    c_var, a_var, b_var = sp.symbols("c_var a_var b_var", real=True)

    # 196. prob-07 Chebyshev inequality & 2022 / 2001 exams
    # 2022 Exam: D((1/n) sum X_i^2) = (mu4 - mu2^2)/n -> bound (mu4 - mu2^2)/(n*eps^2)
    mu2_s, mu4_s = sp.symbols("mu2_s mu4_s", positive=True)
    cheb_22 = ((mu4_s - mu2_s**2) / n_i) / eps_p**2
    # 2001 Exam: EX=EY=2, DX=1, DY=4, rho=0.5 -> D(X-Y)=1+4-2*(0.5*1*2)=3 -> P(|X-Y|>=6)<=3/36=1/12
    dz_01 = 1 + 4 - 2 * sp.Rational(1, 2) * 1 * 2
    b_01 = sp.Rational(dz_01, 36)
    # Chi-square Chebyshev: U~chi2(n) -> EU=n, DU=2n -> P(|U-n|<n)>=1-2n/n^2=1-2/n
    b_chi2 = sp.simplify(1 - (2 * n_i) / n_i**2)
    check(
        "m1_pr07_chebyshev_and_exams",
        sp.simplify(cheb_22 - (mu4_s - mu2_s**2) / (n_i * eps_p**2)) == 0
        and b_01 == sp.Rational(1, 12)
        and sp.simplify(b_chi2 - (1 - 2 / n_i)) == 0,
        "prob-07 2022 Chebyshev (mu4-mu2^2)/(n*eps^2), 2001 (1/12) & chi2 (1-2/n) exact",
    )

    # 197. prob-07 Convergence in probability: stretched Cauchy Prob9 Ex 7.1
    p_cauchy = 2 * sp.integrate(n_p / (sp.pi * (1 + (n_p * x)**2)), (x, 0, eps_p))
    lim_cauchy = sp.limit(p_cauchy, n_p, sp.oo)
    check(
        "m1_pr07_prob_conv_cauchy",
        sp.simplify(p_cauchy - 2 / sp.pi * sp.atan(n_p * eps_p)) == 0 and lim_cauchy == 1,
        "prob-07 Prob9 Ex7.1 stretched Cauchy P(|X_n|<eps)=2/pi*atan(n*eps) -> 1 exact",
    )

    # 198. prob-07 LLN series, Markov bridge, 2022 triangle, 2003 Exp(2), Monte Carlo pi
    # Markov ratios: sum_{i=1}^n i / n^2 -> 1/2; sum i^2 / n^2 -> oo
    lim_poiss_markov = sp.limit(sp.summation(k_i, (k_i, 1, n_i)) / n_i**2, n_i, sp.oo)
    lim_exp_markov = sp.limit(sp.summation(k_i**2, (k_i, 1, n_i)) / n_i**2, n_i, sp.oo)
    # 2022 triangle f(x)=1-|x| on (-1,1) -> E(X^2)=1/6
    ex2_tri22 = 2 * sp.integrate(x**2 * (1 - x), (x, 0, 1))
    # 2003 E(2) -> E(X^2) = 1/4 + 1/4 = 1/2
    ex2_e2_03 = sp.integrate(x**2 * 2 * sp.exp(-2 * x), (x, 0, sp.oo))
    # Monte Carlo pi: E[4*sqrt(1-U^2)] = pi, E[16(1-U^2)] = 32/3
    e_mc_pi = 4 * sp.integrate(sp.sqrt(1 - x**2), (x, 0, 1))
    e2_mc_pi = 16 * sp.integrate(1 - x**2, (x, 0, 1))
    check(
        "m1_pr07_lln_series_and_markov",
        lim_poiss_markov == sp.Rational(1, 2)
        and lim_exp_markov == sp.oo
        and ex2_tri22 == sp.Rational(1, 6)
        and ex2_e2_03 == sp.Rational(1, 2)
        and e_mc_pi == sp.pi
        and e2_mc_pi == sp.Rational(32, 3),
        "prob-07 Markov P(i)->1/2, E(1/i)->oo, 2022 tri 1/6, 2003 E(2) 1/2 & MC pi exact",
    )

    # 199. prob-07 CLT core exams (2020, 2001 truck, rounding, 1988 insurance, 5-box, hotel power)
    z_2020 = (55 - 50) / sp.sqrt(100 * sp.Rational(1, 4))
    # 2001 truck: 5*t^2 + t - 500 < 0 -> t_max = (sqrt(10001)-1)/10, n = floor(t^2) = 98
    t_truck = (sp.sqrt(10001) - 1) / 10
    n_truck = int(sp.floor(t_truck**2))
    # Rounding error: U(-0.5, 0.5), var=1/12, 10 / sqrt(n/12) >= 1.645 -> n <= 1200 / 1.645^2 = 443.45
    n_round = int(sp.floor(1200 / sp.Rational(1645, 1000)**2))
    # 1988 insurance: B(100, 0.2), mu=20, sd=4 -> (26-20)/4 = 3/2
    z_ins = sp.Rational(26 - 20, 4)
    # 5-box: B(100, 0.2), (24-20)/4=1, (28-20)/4=2
    z_box_lo, z_box_hi = sp.Rational(24 - 20, 4), sp.Rational(28 - 20, 4)
    # Hotel 500 rooms p=0.8, 2kW: 2*(400 + 2.33*sqrt(80))
    pwr_hotel = float(2 * (400 + sp.Rational(233, 100) * sp.sqrt(80)))
    check(
        "m1_pr07_clt_core_exams",
        z_2020 == 1 and n_truck == 98 and n_round == 443
        and z_ins == sp.Rational(3, 2) and (z_box_lo, z_box_hi) == (1, 2)
        and abs(pwr_hotel - 841.68) < 0.02,
        f"prob-07 CLT 2020(z=1), 2001(n=98), round(443), 1988(1.5), 5box(1..2), hotel({pwr_hotel:.2f}) exact",
    )

    # 200. prob-07 CLT tricks (paired diff, U(0,1) sqrt(3), Exp(1) D(X^2)=20, polling 9604 vs 50000, Poisson 19/3*e^-2)
    dy_pair_b = sp.Rational(1, 4) + sp.Rational(1, 4)  # 1/2 -> a = 1/sqrt(1/2) = sqrt(2)
    z_u01 = sp.Rational(1, 2) / sp.sqrt(sp.Rational(1, 12))
    ex4_e1 = sp.integrate(x**4 * sp.exp(-x), (x, 0, sp.oo))
    ex2_e1 = sp.integrate(x**2 * sp.exp(-x), (x, 0, sp.oo))
    dx2_e1 = ex4_e1 - ex2_e1**2
    n_poll_clt = int((sp.Rational(196, 100) / sp.Rational(2, 100))**2)
    n_poll_cheb = int(sp.Rational(1, 4) / (sp.Rational(1, 100)**2 * sp.Rational(5, 100)))
    p_pois3 = sum(sp.Integer(2)**k / sp.factorial(k) for k in range(4)) * sp.exp(-2)
    check(
        "m1_pr07_clt_tricks",
        sp.simplify(1 / sp.sqrt(dy_pair_b) - sp.sqrt(2)) == 0
        and sp.simplify(z_u01 - sp.sqrt(3)) == 0
        and dx2_e1 == 20
        and n_poll_clt == 9604 and n_poll_cheb == 50000
        and sp.simplify(p_pois3 - sp.Rational(19, 3) * sp.exp(-2)) == 0,
        "prob-07 paired sqrt(2), U(0,1) sqrt(3), E(1) D(X^2)=20, poll 9604 vs 50000 & Poisson 19/3*e^-2 exact",
    )

    # 201. prob-07 Empirical CDF max variance 1/(4n) and covariance identity
    p_f = sp.Symbol("p_f", positive=True)
    max_v_cdf = sp.diff(p_f * (1 - p_f), p_f)
    p_opt_cdf = sp.solve(max_v_cdf, p_f)[0]
    check(
        "m1_pr07_empirical_cdf",
        p_opt_cdf == sp.Rational(1, 2) and (p_opt_cdf * (1 - p_opt_cdf)) == sp.Rational(1, 4),
        "prob-07 Empirical CDF max F(1-F)=1/4 at F=1/2 -> D(F_n)<=1/(4n) exact",
    )

    # 202. prob-08 Sample moments, 2023 a=sqrt(pi)/2, residual Cov, E[(Xbar-S)^2], B(1,p) counterexample
    e_abs_z = 2 * sp.integrate(x / sp.sqrt(2 * sp.pi) * sp.exp(-x**2 / 2), (x, 0, sp.oo))
    a_2023 = sp.simplify(1 / (sp.sqrt(2) * e_abs_z))
    # Residual variance & cov: D(X_k - Xbar) = ((n-1)/n)^2 + (n-1)/n^2 = (n-1)/n
    d_res = sp.simplify(((n_i - 1) / n_i)**2 + (n_i - 1) / n_i**2)
    cov_res = sp.simplify(0 - 1 / n_i - 1 / n_i + 1 / n_i)
    # B(1,p) 2-sample: verify S^2 = 2*Xbar*(1-Xbar) for all 4 outcomes (0,0),(0,1),(1,0),(1,1)
    b1p_ok = all(
        ((x1 - (x1 + x2) / 2)**2 + (x2 - (x1 + x2) / 2)**2) == 2 * ((x1 + x2) / 2) * (1 - (x1 + x2) / 2)
        for x1 in (sp.Integer(0), sp.Integer(1))
        for x2 in (sp.Integer(0), sp.Integer(1))
    )
    check(
        "m1_pr08_sample_moments",
        sp.simplify(e_abs_z - sp.sqrt(2 / sp.pi)) == 0
        and sp.simplify(a_2023 - sp.sqrt(sp.pi) / 2) == 0
        and sp.simplify(d_res - (n_i - 1) / n_i) == 0
        and sp.simplify(cov_res - (-1 / n_i)) == 0
        and b1p_ok,
        "prob-08 E|Z|=sqrt(2/pi), 2023 a=sqrt(pi)/2, residual var/cov & B(1,p) S^2=2Xbar(1-Xbar) exact",
    )

    # 203. prob-08 Three sampling distributions properties (1998, Fang Hao Ex11-14, 2013, Mao k, 2D density)
    # 1998: X_i~N(0,4), D(X1-2X2)=4+16=20 -> a=1/20; D(3X3-4X4)=36+64=100 -> b=1/100
    a_98, b_98 = sp.Rational(1, 1 * 4 + 4 * 4), sp.Rational(1, 9 * 4 + 16 * 4)
    # Fang Hao Ex 11-14: sqrt(3/2)
    c_fh11 = sp.sqrt(sp.Rational(3, 2))
    # Mao Ex 6.22: k/(1-k) = 161.45 -> k = 161.45/162.45
    k_mao = sp.solve(sp.Eq(c_var / (1 - c_var), sp.Rational(16145, 100)), c_var)[0]
    # 2D density normalization check: int int 1/(4pi) exp(-x^2/2 - (y-1)^2/8) = 1
    norm_617 = sp.integrate(sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi), (x, -sp.oo, sp.oo)) * sp.integrate(
        sp.exp(-(y - 1)**2 / 8) / (2 * sp.sqrt(2 * sp.pi)), (y, -sp.oo, sp.oo)
    )
    check(
        "m1_pr08_three_dists_props",
        a_98 == sp.Rational(1, 20) and b_98 == sp.Rational(1, 100)
        and sp.simplify(c_fh11**2 - sp.Rational(3, 2)) == 0
        and k_mao == sp.Rational(16145, 16245)
        and norm_617 == 1,
        "prob-08 1998(1/20,1/100), FH t(3)/t(1) sqrt(3/2), Mao k=16145/16245 & Ex6.17 norm=1 exact",
    )

    # 204. prob-08 Normal sampling theorems (D(S^2), 2023 F, 2004 pooled, prediction c, 6+3 split, Ex6.2-6.3)
    d_s2 = sp.simplify((sig_s**2 / (n_i - 1))**2 * 2 * (n_i - 1))
    # 9-sample 6+3 split: D(Y1-Y2) = (1/6 + 1/3)*sigma^2 = sigma^2/2 -> coeff sqrt(2)
    v_split9 = sp.Rational(1, 6) + sp.Rational(1, 3)
    # Prediction: D(X_{n+1} - Xbar) = (1 + 1/n)*sigma^2 = (n+1)/n * sigma^2 -> c = sqrt(n/(n+1))
    c_pred = sp.sqrt(1 / (1 + 1 / n_i))
    # Ex 6.2-6.3: 4*chi2(9) -> E=36, D=288
    ew_63, dw_63 = 4 * 9, 16 * 18
    check(
        "m1_pr08_normal_sampling_theorems",
        sp.simplify(d_s2 - 2 * sig_s**4 / (n_i - 1)) == 0
        and v_split9 == sp.Rational(1, 2)
        and sp.simplify(c_pred - sp.sqrt(n_i / (n_i + 1))) == 0
        and ew_63 == 36 and dw_63 == 288,
        "prob-08 D(S^2)=2sig^4/(n-1), 6+3 split var=1/2, pred sqrt(n/(n+1)) & 4chi2(9)(36,288) exact",
    )

    # 205. prob-08 Fang Hao "see square find chi2" tricks (D(Xbar^2-S^2/n), D(A2), E(S^4), E(1/S^2))
    d_fh5 = sp.simplify(2 / n_i**2 + 2 * (n_i - 1) / (n_i * (n_i - 1))**2)
    e_s4 = sp.simplify(2 * sig_s**4 / (n_i - 1) + sig_s**4)
    # Inverse chi2(k) expectation for k=5: int_0^oo (1/v) * (1/2)^(5/2)/Gamma(5/2) * v^(5/2-1) e^(-v/2) dv = 1/3 = 1/(5-2)
    e_inv_chi5 = sp.integrate(
        (sp.Rational(1, 2)**sp.Rational(5, 2) / sp.gamma(sp.Rational(5, 2))) * x**(sp.Rational(3, 2) - 1) * sp.exp(-x / 2),
        (x, 0, sp.oo),
    )
    check(
        "m1_pr08_fanghao_chi2_tricks",
        sp.simplify(d_fh5 - 2 / (n_i * (n_i - 1))) == 0
        and sp.simplify(e_s4 - (n_i + 1) / (n_i - 1) * sig_s**4) == 0
        and e_inv_chi5 == sp.Rational(1, 3),
        "prob-08 FH D(Xbar^2-S^2/n)=2/(n(n-1)), E(S^4)=(n+1)/(n-1)sig^4 & E(1/chi2(5))=1/3 exact",
    )

    # 206. prob-08 Non-normal bridges (Laplace quotient F(2,2) density & -2ln U ~ E(1/2))
    # Quotient of two indep E(1) vars: f_Z(z) = int_0^oo v*exp(-z*v)*exp(-v) dv = 1/(1+z)^2
    f_lap_q = sp.integrate(y * sp.exp(-xp * y) * sp.exp(-y), (y, 0, sp.oo))
    # Check F(2,2) pdf formula: Gamma(2)/(Gamma(1)*Gamma(1)) * (2/2)^1 * 1 / (1 + x)^2 = 1/(1+x)^2
    norm_f22 = sp.integrate(f_lap_q, (xp, 0, sp.oo))
    check(
        "m1_pr08_nonnormal_bridges",
        sp.simplify(f_lap_q - 1 / (1 + xp)**2) == 0 and norm_f22 == 1,
        "prob-08 Prob9 Ex8.3 Laplace |X1/X2| quotient pdf = 1/(1+t)^2 = F(2,2) exact",
    )

    # 207. prob-08 Order statistics bridge: U(0, theta) X_(n) moments
    e_xn = sp.integrate(n_p * x**n_p / th_p**n_p, (x, 0, th_p))
    e_xn2 = sp.integrate(n_p * x**(n_p + 1) / th_p**n_p, (x, 0, th_p))
    d_xn = sp.simplify(e_xn2 - e_xn**2)
    check(
        "m1_pr08_order_stats_bridge",
        sp.simplify(e_xn - n_p * th_p / (n_p + 1)) == 0
        and sp.simplify(d_xn - n_p * th_p**2 / ((n_p + 1)**2 * (n_p + 2))) == 0,
        "prob-08 U(0,th) X_(n) E=n*th/(n+1) & D=n*th^2/((n+1)^2*(n+2)) exact",
    )

    # 208. prob-09 MoM & MLE core table (Laplace, shifted exponential)
    ex2_lap = 2 * sp.integrate(x**2 / (2 * th_p) * sp.exp(-x / th_p), (x, 0, sp.oo))
    ex_sh = sp.integrate(x / th_p * sp.exp(-(x - mu_s) / th_p), (x, mu_s, sp.oo))
    dx_sh = sp.simplify(sp.integrate((x - ex_sh)**2 / th_p * sp.exp(-(x - mu_s) / th_p), (x, mu_s, sp.oo)))
    check(
        "m1_pr09_mom_mle_table",
        sp.simplify(ex2_lap - 2 * th_p**2) == 0
        and sp.simplify(ex_sh - (mu_s + th_p)) == 0
        and sp.simplify(dx_sh - th_p**2) == 0,
        "prob-09 Laplace E(X^2)=2*th^2 & shifted-exp (E=mu+th, D=th^2) exact",
    )

    # 209. prob-09 Boundary & special MLE (U[th,3th], capture-recapture N=15000, discrete 5/6)
    # U[th, 3th]: X_(n) = th + 2*th*U_(n), th_hat = X_(n)/3
    e_u3 = sp.simplify((th_p + 2 * th_p * n_p / (n_p + 1)) / 3)
    d_u3 = sp.simplify((2 * th_p / 3)**2 * n_p / ((n_p + 1)**2 * (n_p + 2)))
    # Capture-recapture: (N-1000)(N-150) = N(N-1140) => N = 15000
    n_fish = sp.solve(sp.Eq((x - 1000) * (x - 150), x * (x - 1140)), x)[0]
    # Ex 7.12: L(th) = 2*th^5*(1-th) -> d/dth ln L = 5/th - 1/(1-th) = 0 -> th = 5/6
    th_712 = sp.solve(5 / x - 1 / (1 - x), x)[0]
    check(
        "m1_pr09_boundary_and_special_mle",
        sp.simplify(e_u3 - (3 * n_p + 1) * th_p / (3 * (n_p + 1))) == 0
        and sp.simplify(d_u3 - 4 * n_p * th_p**2 / (9 * (n_p + 1)**2 * (n_p + 2))) == 0
        and n_fish == 15000 and th_712 == sp.Rational(5, 6),
        "prob-09 U[th,3th] E/D, capture-recapture N=15000 & Ex7.12 th=5/6 exact",
    )

    # 210. prob-09 Recent MLE exams (2022 dual-exp, 2021 discrete 1/4, 2020 Weibull, 2017 folded, 2014 Rayleigh, Base Ex6.8)
    # 2022: D(th_hat) = (n*th^2 + 1/4*m*4*th^2)/(n+m)^2 = th^2/(n+m)
    d_2022 = sp.simplify((n_i * th_p**2 + sp.Rational(1, 4) * m_i * 4 * th_p**2) / (n_i + m_i)**2)
    # 2021: 3 ones, 5 twos/threes -> -3/(1-th) + 5/(1+th) = 0 -> th = 1/4
    th_2021 = sp.solve(-3 / (1 - x) + 5 / (1 + x), x)[0]
    # 2014 Rayleigh: F(x)=1-exp(-x^2/th), EX = int_0^oo exp(-x^2/th) dx = sqrt(pi*th)/2, EX^2 = th
    ex_2014 = sp.integrate(sp.exp(-x**2 / th_p), (x, 0, sp.oo))
    ex2_2014 = sp.integrate(x**2 * (2 * x / th_p) * sp.exp(-x**2 / th_p), (x, 0, sp.oo))
    # 2017 folded normal: E|X-mu| = sqrt(2/pi)*sigma
    e_fold17 = sp.integrate(x * sp.sqrt(2 / sp.pi) / sig_s * sp.exp(-x**2 / (2 * sig_s**2)), (x, 0, sp.oo))
    check(
        "m1_pr09_recent_mle_exams",
        sp.simplify(d_2022 - th_p**2 / (n_i + m_i)) == 0
        and th_2021 == sp.Rational(1, 4)
        and sp.simplify(ex_2014 - sp.sqrt(sp.pi * th_p) / 2) == 0
        and sp.simplify(ex2_2014 - th_p) == 0
        and sp.simplify(e_fold17 - sp.sqrt(2 / sp.pi) * sig_s) == 0,
        "prob-09 2022(th^2/(n+m)), 2021(1/4), 2014(sqrt(pi*th)/2, th) & 2017 folded exact",
    )

    # 211. prob-09 Criteria & 2024 / 2026 exams
    # 2024: h(c) = th^2*(n/(n+2)*c^2 - 2n/(n+1)*c + 1) -> c_min = (n+2)/(n+1), MSE gap > 0
    h_24 = th_p**2 * (n_p / (n_p + 2) * c_var**2 - 2 * n_p / (n_p + 1) * c_var + 1)
    c_opt_24 = sp.solve(sp.diff(h_24, c_var), c_var)[0]
    mse_unb_24 = sp.simplify(h_24.subs(c_var, (n_p + 1) / n_p))
    mse_opt_24 = sp.simplify(h_24.subs(c_var, c_opt_24))
    # Variance estimator MSE: T_c = c*sig^2*V (V~chi2(n-1)) -> min at c = 1/(n+1)
    mse_var = sig_s**4 * (2 * (n_p - 1) * c_var**2 + ((n_p - 1) * c_var - 1)**2)
    c_var_opt = sp.solve(sp.diff(mse_var, c_var), c_var)[0]
    # Efficiency U(0,th) n=3: (4/3)^2 * 3/80 = 1/15 vs 16 * 3/80 = 3/5
    d_eff1, d_eff2 = sp.Rational(16, 9) * sp.Rational(3, 80), 16 * sp.Rational(3, 80)
    # Optimal weights a/m = (1-a)/n -> a = m/(m+n)
    a_wt = sp.solve(sp.diff(a_var**2 / m_i + (1 - a_var)**2 / n_i, a_var), a_var)[0]
    check(
        "m1_pr09_criteria_and_2024_2026",
        sp.simplify(c_opt_24 - (n_p + 2) / (n_p + 1)) == 0
        and sp.simplify(mse_unb_24 - mse_opt_24 - th_p**2 / (n_p * (n_p + 1)**2 * (n_p + 2))) == 0
        and sp.simplify(c_var_opt - 1 / (n_p + 1)) == 0
        and d_eff1 == sp.Rational(1, 15) and d_eff2 == sp.Rational(3, 5)
        and sp.simplify(a_wt - m_i / (m_i + n_i)) == 0,
        "prob-09 2024 MSE (n+2)/(n+1), var min-MSE 1/(n+1), eff 1/15<3/5 & opt weight m/(m+n) exact",
    )

    # 212. prob-09 CI & length & Wilson quadratic interval
    lo_2016 = sp.Rational(95, 10) - (sp.Rational(108, 10) - sp.Rational(95, 10))
    n_ci_726 = int(sp.ceiling((2 * sp.Rational(196, 100) / sp.Rational(12, 10))**2))
    n_ci_fh16 = int(sp.ceiling((2 * 2 * 3 / 2)**2))
    # Wilson quadratic roots for xbar=0.1, n=100, z=1.96
    z2n = sp.Rational(196, 100)**2 / 100
    w_roots = sorted(float(r) for r in sp.solve((1 + z2n) * x**2 - (sp.Rational(2, 10) + z2n) * x + sp.Rational(1, 100), x))
    check(
        "m1_pr09_ci_and_length",
        lo_2016 == sp.Rational(82, 10) and n_ci_726 == 11 and n_ci_fh16 == 36
        and abs(w_roots[0] - 0.0552) < 5e-4 and abs(w_roots[1] - 0.1744) < 5e-4,
        f"prob-09 2016 CI [8.2,10.8], min n=11 & 36, Wilson [{w_roots[0]:.4f},{w_roots[1]:.4f}] exact",
    )

    # 213. prob-09 Hypothesis testing & Type I/II errors (2025, 2021, 1998, Base Ex6.16, Mao Ex8.10)
    # 2021: mu=11.5, sigma=2, n=16 -> sd=0.5 -> (11-11.5)/0.5 = -1
    z_2021_beta = (11 - sp.Rational(23, 2)) / (sp.Integer(2) / 4)
    # 1998: |sqrt(36)*(66.5-70)/15| = 1.4
    t_1998 = sp.Abs(6 * (sp.Rational(133, 2) - 70) / 15)
    # Base Ex 6.16: alpha = int_{3/4}^1 1 dx = 1/4, beta = int_0^{3/4} 2x dx = 9/16
    alpha_616 = sp.integrate(1, (x, sp.Rational(3, 4), 1))
    beta_616 = sp.integrate(2 * x, (x, 0, sp.Rational(3, 4)))
    # Mao Ex 8.10: n >= ceil((2*1.645*sqrt(2.5)/2)^2) = ceil(2.5*1.645^2) = 7
    n_810 = int(sp.ceiling(sp.Rational(5, 2) * sp.Rational(1645, 1000)**2))
    check(
        "m1_pr09_hypothesis_and_errors",
        z_2021_beta == -1 and t_1998 == sp.Rational(7, 5)
        and alpha_616 == sp.Rational(1, 4) and beta_616 == sp.Rational(9, 16)
        and alpha_616 + beta_616 == sp.Rational(13, 16) and n_810 == 7,
        "prob-09 2021 beta=1-Phi(1), 1998 |t|=1.4, Ex6.16 (1/4, 9/16) & Ex8.10 n=7 exact",
    )

    # 214. prob-07 Markov boundary proof: DX_i = i^(1/2) obeys Markov LLN whereas DX_i = i does not
    # Integral upper bound for sum_{k=1}^n sqrt(k) <= int_0^{n+1} sqrt(t) dt = 2/3*(n+1)^(3/2) -> /n^2 -> 0
    lim_sqrt_markov = sp.limit(sp.Rational(2, 3) * (n_p + 1)**sp.Rational(3, 2) / n_p**2, n_p, sp.oo)
    lim_log_markov = sp.limit(n_p * sp.log(n_p) / n_p**2, n_p, sp.oo)
    check(
        "m1_pr07_markov_boundary_proof",
        lim_sqrt_markov == 0 and lim_log_markov == 0,
        "prob-07 Unbounded DX_i=sqrt(i) or ln(i) still satisfies Markov (1/n^2)sum DX_i -> 0 exact",
    )

    # 215. prob-08 Explicit 4-variable symbolic Cochran & double-sum algebraic identities
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
    xs4 = [x1, x2, x3, x4]
    xb4 = sum(xs4) / 4
    cochran_diff = sp.expand(sum((xi - mu_s)**2 for xi in xs4) - (sum((xi - xb4)**2 for xi in xs4) + 4 * (xb4 - mu_s)**2))
    dbl_diff = sp.expand(sum((xi - xj)**2 for xi in xs4 for xj in xs4) - 2 * 4 * sum((xi - xb4)**2 for xi in xs4))
    check(
        "m1_pr08_cochran_algebra_n4",
        cochran_diff == 0 and dbl_diff == 0,
        "prob-08 Explicit n=4 symbolic Cochran & double-sum 2n(n-1)S^2 identities exact",
    )

    # 216. prob-08 t(3) exact variance = 3/(3-2) = 3 via direct pdf integration
    # t(3) pdf: Gamma(2)/(sqrt(3*pi)*Gamma(3/2)) * (1 + x^2/3)^(-2) = 2/(pi*sqrt(3)) * (1 + x^2/3)^(-2)
    f_t3 = sp.gamma(2) / (sp.sqrt(3 * sp.pi) * sp.gamma(sp.Rational(3, 2))) * (1 + x**2 / 3)**(-2)
    norm_t3 = sp.integrate(f_t3, (x, -sp.oo, sp.oo))
    var_t3 = sp.integrate(x**2 * f_t3, (x, -sp.oo, sp.oo))
    check(
        "m1_pr08_chi2_f_pdf_check",
        sp.simplify(norm_t3 - 1) == 0 and sp.simplify(var_t3 - 3) == 0,
        "prob-08 t(3) pdf normalization=1 & variance=3/(3-2)=3 exact",
    )

    # 217. prob-09 Inverse Gamma check: X_i ~ E(lam) (n=5) -> S=sum X_i ~ Gamma(5, lam), E(1/Xbar)=5/4*lam
    lam_p = sp.Symbol("lam_p", positive=True)
    f_gam5 = lam_p**5 / sp.factorial(4) * x**4 * sp.exp(-lam_p * x)
    e_inv_xbar5 = sp.integrate((5 / x) * f_gam5, (x, 0, sp.oo))
    check(
        "m1_pr09_exponential_unbiased_rate",
        sp.simplify(e_inv_xbar5 - sp.Rational(5, 4) * lam_p) == 0,
        "prob-09 E(1/Xbar) = n/(n-1)*lam verified explicitly for n=5 exact",
    )

    # 218. prob-09 Dual-normal pooled variance S_w^2 expectation & variance
    df_tot = n_i + m_i - 2
    e_sw2 = sp.simplify(((n_i - 1) * sig_s**2 + (m_i - 1) * sig_s**2) / df_tot)
    d_sw2 = sp.simplify(((n_i - 1)**2 * 2 * sig_s**4 / (n_i - 1) + (m_i - 1)**2 * 2 * sig_s**4 / (m_i - 1)) / df_tot**2)
    check(
        "m1_pr09_dual_normal_pooled_sw",
        sp.simplify(e_sw2 - sig_s**2) == 0 and sp.simplify(d_sw2 - 2 * sig_s**4 / df_tot) == 0,
        "prob-09 Pooled S_w^2 E=sig^2 & D=2*sig^4/(n1+n2-2) exact",
    )

    # 219. prob-09 CI-Test duality & asymmetric chi2 denominator inversion
    # Verify chi2_lo < chi2_hi => (n-1)S^2/chi2_hi < (n-1)S^2/chi2_lo
    q_lo, q_hi = sp.Rational(218, 100), sp.Rational(17535, 1000)  # chi2(8) 0.975 & 0.025
    ci_s2_lo, ci_s2_hi = sp.Rational(26, 100) / q_hi, sp.Rational(26, 100) / q_lo
    check(
        "m1_pr09_ci_test_duality",
        q_lo < q_hi and ci_s2_lo < ci_s2_hi and abs(float(sp.sqrt(ci_s2_lo)) - 0.1218) < 5e-4
        and abs(float(sp.sqrt(ci_s2_hi)) - 0.3454) < 5e-4,
        f"prob-09 Chi2 inversion sigma CI [{float(sp.sqrt(ci_s2_lo)):.4f}, {float(sp.sqrt(ci_s2_hi)):.4f}] exact",
    )

    # 220. prob-09 Errata verification: composite null supremum at boundary & left-tail chi2 direction
    # For H_0: mu <= mu_0 with right-tail W = {Xbar > mu_0 + z*sd}, P_mu(W) = 1 - Phi(z + (mu_0 - mu)/sd)
    # Derivative w.r.t. mu is positive -> supremum over mu <= mu_0 achieved strictly at mu = mu_0!
    z_sym, sd_p = sp.symbols("z_sym sd_p", positive=True)
    # Power function g(mu) = 1 - (1/2)*(1 + erf((z_sym + (mu_s - x)/sd_p)/sqrt(2)))
    g_pow = sp.Rational(1, 2) * sp.erfc((z_sym + (mu_s - x) / sd_p) / sp.sqrt(2))
    dg_dmu = sp.simplify(sp.diff(g_pow, x))
    # For x < mu_s, check dg_dmu > 0 numerically at sample point
    dg_num = float(dg_dmu.subs({z_sym: 1.645, mu_s: 1.0, sd_p: 0.5, x: 0.5}))
    p_interior = float(g_pow.subs({z_sym: 1.645, mu_s: 1.0, sd_p: 0.5, x: 0.5}))
    p_boundary = float(g_pow.subs({z_sym: 1.645, mu_s: 1.0, sd_p: 0.5, x: 1.0}))
    check(
        "m1_pr09_errata_direction_verify",
        dg_num > 0 and p_interior < p_boundary and abs(p_boundary - 0.05) < 5e-4,
        f"prob-09 Fang Hao P.181 rigor: P_int({p_interior:.4f}) < P_bdry({p_boundary:.4f})=alpha exact",
    )

    return results


def main() -> int:
    results = run_all_checks()
    failed = 0
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{tag}] {name:32s} {detail}")
    print(f"\nMath I SymPy Verification Summary: {len(results) - failed}/{len(results)} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
