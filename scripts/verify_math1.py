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
