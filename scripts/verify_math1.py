#!/usr/bin/env python3
r"""
Exact symbolic & numerical verification suite (SymPy) for Math I (27 Zhang Yu 6-book system).
Run with workspace .venv:
    .\.venv\Scripts\python.exe scripts/verify_math1.py
"""

from __future__ import annotations

import math
import sys

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
