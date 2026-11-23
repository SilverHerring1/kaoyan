#!/usr/bin/env python3
"""
Exact symbolic & numerical verification of Control Theory (845) Ch03 & Ch04 formulas
and textbook worked examples using SymPy 1.14.0.

Run via project .venv:
  .\\.venv\\Scripts\\python.exe scripts/verify_stem_math.py
"""

from __future__ import annotations

import math
import sys
import sympy as sp

s, t, tau, T, K, wn, zeta, a, b, eps = sp.symbols(
    "s t tau T K omega_n zeta a b epsilon", positive=True
)
s_c = sp.Symbol("s")
t_r = sp.Symbol("t", real=True, nonnegative=True)


def routh_first_column(poly_coeffs: list[sp.Expr]) -> list[sp.Expr]:
    """Compute standard Routh array first column (for non-degenerate numeric/symbolic polys)."""
    n = len(poly_coeffs) - 1
    row0 = poly_coeffs[0::2]
    row1 = poly_coeffs[1::2]
    cols = max(len(row0), len(row1))
    row0 = row0 + [sp.Integer(0)] * (cols - len(row0))
    row1 = row1 + [sp.Integer(0)] * (cols - len(row1))
    table = [row0, row1]
    for r in range(2, n + 1):
        prev2 = table[r - 2]
        prev1 = table[r - 1]
        pivot = prev1[0]
        if pivot == 0:
            raise ZeroDivisionError(f"Zero pivot at row {r}")
        new_row = []
        for c in range(cols - 1):
            val = sp.simplify((pivot * prev2[c + 1] - prev2[0] * prev1[c + 1]) / pivot)
            new_row.append(val)
        new_row.append(sp.Integer(0))
        table.append(new_row)
    return [row[0] for row in table]


def run_all_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append((name, bool(ok), detail))

    # --- Ch03 Formulas ---
    # 1. First-order step/impulse/ramp Laplace & indicators
    g1_step = sp.inverse_laplace_transform(K / (T * s_c + 1) / s_c, s_c, t)
    check("ch03_f_first_order_step", sp.simplify(g1_step - K * (1 - sp.exp(-t / T))) == 0)
    g1_imp = sp.inverse_laplace_transform(K / (T * s_c + 1), s_c, t)
    check("ch03_f_first_order_impulse", sp.simplify(g1_imp - (K / T) * sp.exp(-t / T)) == 0)
    g1_ramp = sp.inverse_laplace_transform(K / (T * s_c + 1) / s_c**2, s_c, t)
    check("ch03_f_first_order_ramp", sp.simplify(g1_ramp - K * (t - T + T * sp.exp(-t / T))) == 0)
    td_ratio = math.log(2)
    tr_ratio = math.log(9)
    check("ch03_f_first_order_td_tr", abs(td_ratio - 0.6931) < 0.005 and abs(tr_ratio - 2.1972) < 0.005,
          f"Td/T={td_ratio:.4f}, Tr/T={tr_ratio:.4f}")

    # 2. Second-order underdamped overshoot & peak time (exact numerical test at zeta=0.5, wn=4)
    z_val, wn_val = 0.5, 4.0
    wd_val = wn_val * math.sqrt(1 - z_val**2)
    tp_val = math.pi / wd_val
    sigma_val = math.exp(-math.pi * z_val / math.sqrt(1 - z_val**2))
    # Verify derivative of y(t) = 0 at Tp
    z_sym = sp.Rational(1, 2)
    wn_sym = sp.Integer(4)
    wd_sym = wn_sym * sp.sqrt(1 - z_sym**2)
    beta_sym = sp.acos(z_sym)
    y2 = 1 - (1 / sp.sqrt(1 - z_sym**2)) * sp.exp(-z_sym * wn_sym * t) * sp.sin(wd_sym * t + beta_sym)
    dy2_at_tp = sp.simplify(sp.diff(y2, t).subs(t, sp.pi / wd_sym))
    y2_at_tp = sp.simplify(y2.subs(t, sp.pi / wd_sym))
    check("ch03_f_second_order_tp_overshoot",
          dy2_at_tp == 0 and sp.simplify(y2_at_tp - (1 + sp.exp(-sp.pi * z_sym / sp.sqrt(1 - z_sym**2)))) == 0,
          f"sigma(0.5)={sigma_val*100:.2f}%, Tp={tp_val:.4f}")

    # 3. Decay ratio n = B / B' = exp(2*zeta*pi / sqrt(1-zeta^2))
    y2_at_3tp = sp.simplify(y2.subs(t, 3 * sp.pi / wd_sym))
    decay_n = sp.simplify((y2_at_tp - 1) / (y2_at_3tp - 1))
    check("ch03_f_decay_ratio",
          sp.simplify(decay_n - sp.exp(2 * sp.pi * z_sym / sp.sqrt(1 - z_sym**2))) == 0)

    # 4. State transition matrix exp(At) - Main Textbook Ex 3-11 & Ex 3-12 & Ex 3-17
    A311 = sp.Matrix([[2, -1], [-1, 2]])
    eA311 = sp.simplify(sp.exp(A311 * t))
    expected_311 = sp.Rational(1, 2) * sp.Matrix([[sp.exp(t) + sp.exp(3 * t), sp.exp(t) - sp.exp(3 * t)],
                                                  [sp.exp(t) - sp.exp(3 * t), sp.exp(t) + sp.exp(3 * t)]])
    check("ch03_ex_main311_expAt", sp.simplify(eA311 - expected_311) == sp.zeros(2, 2))

    A312 = sp.Matrix([[3, 0, 0, 0], [0, -2, 1, 0], [0, 0, -2, 1], [0, 0, 0, -2]])
    eA312 = sp.simplify(sp.exp(A312 * t))
    expected_312 = sp.Matrix([
        [sp.exp(3 * t), 0, 0, 0],
        [0, sp.exp(-2 * t), t * sp.exp(-2 * t), sp.Rational(1, 2) * t**2 * sp.exp(-2 * t)],
        [0, 0, sp.exp(-2 * t), t * sp.exp(-2 * t)],
        [0, 0, 0, sp.exp(-2 * t)],
    ])
    check("ch03_ex_main312_jordan_expAt", sp.simplify(eA312 - expected_312) == sp.zeros(4, 4))

    A317 = sp.Matrix([[0, 1], [-2, -3]])
    B317 = sp.Matrix([[0], [1]])
    phi317 = sp.simplify((s_c * sp.eye(2) - A317).inv())
    forced_s = sp.simplify(phi317 * B317 * (1 / s_c))
    forced_t = sp.Matrix([sp.inverse_laplace_transform(forced_s[i], s_c, t) for i in range(2)])
    exp_forced = sp.Matrix([
        [sp.Rational(1, 2) - sp.exp(-t) + sp.Rational(1, 2) * sp.exp(-2 * t)],
        [sp.exp(-t) - sp.exp(-2 * t)],
    ])
    check("ch03_ex_main317_nonhomog", sp.simplify(forced_t - exp_forced) == sp.zeros(2, 1))

    # --- Ch03 Guide Examples ---
    # Ex 3-1
    t1_31 = T * sp.log(K / T)
    y1_31 = sp.simplify(K * (1 - sp.exp(-t1_31 / T)))
    y2_31 = sp.simplify(K * (t1_31 - T + T * sp.exp(-t1_31 / T)))
    check("ch03_ex_guide31",
          sp.simplify(y1_31 - (K - T)) == 0 and sp.simplify(y2_31 - (K * T * sp.log(K / T) - K * T + T**2)) == 0)

    # Ex 3-2
    # s^2 + 4.8s + 3.9911 = (s+1.07)(s+3.73)
    p32 = sp.expand((s_c + sp.Rational(107, 100)) * (s_c + sp.Rational(373, 100)))
    tm_32 = 1 / 4.8
    k_32 = 3.9911 / 4.8
    wn_32 = math.sqrt(3.9911)
    zeta_32 = 4.8 / (2 * wn_32)
    ts_32_approx = math.log(0.05 / 1.4) / (-1.07)
    check("ch03_ex_guide32",
          abs(tm_32 - 0.2083) < 0.001 and abs(k_32 - 0.8315) < 0.001
          and abs(zeta_32 - 1.201) < 0.005 and abs(ts_32_approx - 3.114) < 0.005,
          f"Tm={tm_32:.3f}, K={k_32:.3f}, zeta={zeta_32:.3f}, wn={wn_32:.3f}, Ts={ts_32_approx:.3f}")

    # Ex 3-3: Tp=1, sigma=(1.25-0.95)/0.95
    sig_33 = (1.25 - 0.95) / 0.95
    ln_sig = math.log(sig_33)
    z_33 = abs(ln_sig) / math.sqrt(math.pi**2 + ln_sig**2)
    wn_33 = math.pi / math.sqrt(1 - z_33**2)
    check("ch03_ex_guide33",
          abs(z_33 - 0.344) < 0.002 and abs(wn_33 - 3.344) < 0.005,
          f"zeta={z_33:.4f}, wn={wn_33:.4f}, wn^2={wn_33**2:.3f}")

    # Ex 3-4: Phi2 = 2/((s+1)(s+2)), c(t)=1-2e^-t+e^-2t, 5% band -> e^-2Ts - 2e^-Ts + 0.05 = 0
    x_34 = (2 - math.sqrt(4 - 0.2)) / 2
    ts_34 = -math.log(x_34)
    check("ch03_ex_guide34", abs(ts_34 - 3.677) < 0.005, f"x={x_34:.5f}, Ts={ts_34:.3f}")

    # Ex 3-5: s^2 + 25*beta*s + 25 = 0, zeta=0.707 -> wn=5, beta=2*0.707*5/25 = 0.2828, Gc = s^2 + 7.07s
    beta_35 = 2 * 0.707 * 5 / 25
    check("ch03_ex_guide35", abs(beta_35 - 0.2828) < 0.001, f"beta={beta_35:.4f}, 25*beta={25*beta_35:.2f}")

    # Ex 3-6: s^3 + a*s^2 + (2+K)*s + (1+K) = 0 with roots +-2j -> K=2, a=3/4
    sol_36 = sp.solve([sp.re((2 * sp.I)**3 + a * (2 * sp.I)**2 + (2 + K) * (2 * sp.I) + (1 + K)),
                       sp.im((2 * sp.I)**3 + a * (2 * sp.I)**2 + (2 + K) * (2 * sp.I) + (1 + K))], [K, a])
    check("ch03_ex_guide36", sol_36[K] == 2 and sol_36[a] == sp.Rational(3, 4), str(sol_36))

    # Ex 3-8: s^2 + (K*Kh + 1)s + K = 0
    kh_38_1 = (2 * 0.5 * math.sqrt(10) - 1) / 10
    ln002 = math.log(0.02)
    z_38_2 = abs(ln002) / math.sqrt(math.pi**2 + ln002**2)
    wn_38_2 = math.pi / math.sqrt(1 - z_38_2**2)
    k_38_2 = wn_38_2**2
    kh_38_2 = (2 * z_38_2 * wn_38_2 - 1) / k_38_2
    tr_38_2 = (math.pi - math.acos(z_38_2)) / (wn_38_2 * math.sqrt(1 - z_38_2**2))
    check("ch03_ex_guide38",
          abs(kh_38_1 - (math.sqrt(10) - 1) / 10) < 1e-9 and abs(z_38_2 - 0.780) < 0.002
          and abs(wn_38_2 - 5.02) < 0.05 and abs(tr_38_2 - 0.788) < 0.01,
          f"z2={z_38_2:.4f}, wn2={wn_38_2:.3f}, K2={k_38_2:.2f}, Kh2={kh_38_2:.3f}, Tr2={tr_38_2:.3f}")

    # Ex 3-11: Phi = ((2/3)s + 2/9) / (s^2 + (2/3)s + 2/9), z=1/3, wn=sqrt(2)/3, zeta=sqrt(2)/2
    z_311 = 1 / 3
    wn_311 = math.sqrt(2) / 3
    zd_311 = math.sqrt(2) / 2
    r_311 = math.sqrt(z_311**2 - 2 * zd_311 * wn_311 * z_311 + wn_311**2) / (z_311 * math.sqrt(1 - zd_311**2))
    tp_311 = (math.pi / 2) / (wn_311 * math.sqrt(1 - zd_311**2))  # beta_d - psi = pi/4 - (-pi/4) = pi/2
    ov_311 = r_311 * math.sqrt(1 - zd_311**2) * math.exp(-zd_311 * wn_311 * tp_311) * 100
    ts_311 = (4 + math.log(r_311)) / (zd_311 * wn_311)
    check("ch03_ex_guide311",
          abs(r_311 - 1.4142) < 0.005 and abs(tp_311 - 4.712) < 0.02
          and abs(ov_311 - 20.79) < 0.15 and abs(ts_311 - 13.04) < 0.05,
          f"r={r_311:.3f}, Tp={tp_311:.2f}, ov={ov_311:.2f}%, Ts={ts_311:.2f}")

    # Ex 3-12: s(s+4.28)(s+2.22)+10 = s^3 + 6.5s^2 + 9.5016s + 10 ≈ (s+5)(s^2+1.5s+2)
    roots_312 = [complex(r.evalf()) for r in sp.nroots(s_c**3 + sp.Rational(65, 10) * s_c**2 + sp.Rational(95016, 10000) * s_c + 10)]
    real_root = [r.real for r in roots_312 if abs(r.imag) < 1e-3][0]
    check("ch03_ex_guide312", abs(real_root - (-5.0)) < 0.01, f"roots={roots_312}")

    # --- Ch04 Formulas & Examples ---
    # Ex 4-1 (1)-(4): exact RHP root counts via nroots
    def count_rhp(poly):
        return sum(1 for r in sp.nroots(poly) if complex(r.evalf()).real > 1e-6)

    c41_1 = count_rhp(3 * s_c**4 + 10 * s_c**3 + 5 * s_c**2 + s_c + 2)
    c41_2 = count_rhp(s_c**3 + 20 * s_c**2 + 4 * s_c + 50)
    c41_3 = count_rhp(s_c**6 + 4 * s_c**5 - 4 * s_c**4 + 4 * s_c**3 - 7 * s_c**2 - 8 * s_c + 10)
    c41_4 = count_rhp(s_c**4 + 2 * s_c**3 + s_c**2 + 2 * s_c + 1)
    check("ch04_ex_guide41_rhp_counts",
          (c41_1, c41_2, c41_3, c41_4) == (2, 0, 2, 2),
          f"RHP counts={(c41_1, c41_2, c41_3, c41_4)}")

    # Ex 4-2: 2Ts^3 + (2+T)s^2 + (K+1)s + K = 0 -> s^1 entry = (2+T+K(2-T))/(2+T)
    fc42 = routh_first_column([2 * T, 2 + T, K + 1, K])
    check("ch04_ex_guide42", sp.simplify(fc42[2] - (2 + T + K * (2 - T)) / (2 + T)) == 0)

    # Ex 4-3: s^3 + 9s^2 + 18s + 18K = 0, shift s = z - 1 -> z^3 + 6z^2 + 3z + 18K - 10 = 0
    z_sym = sp.Symbol("z")
    p43 = sp.expand((z_sym - 1)**3 + 9 * (z_sym - 1)**2 + 18 * (z_sym - 1) + 18 * K)
    fc43 = routh_first_column([1, 6, 3, 18 * K - 10])
    check("ch04_ex_guide43",
          p43 == z_sym**3 + 6 * z_sym**2 + 3 * z_sym + 18 * K - 10
          and sp.simplify(fc43[2] - (14 - 9 * K) / 3) == 0)

    # Ex 4-4: Ts^2 + s + K = 0, shift s = z - 2 -> Tz^2 + (1-4T)z + (4T+K-2) = 0
    p44 = sp.collect(sp.expand(T * (z_sym - 2)**2 + (z_sym - 2) + K), z_sym)
    check("ch04_ex_guide44", p44 == T * z_sym**2 + (1 - 4 * T) * z_sym + 4 * T + K - 2)

    # Ex 4-6: 0.25s^2 + (K*Kt - 1)s + K = 0, sigma=10%, Ts=1.5s (5% exact envelope: -ln(0.05*sqrt(1-z^2))/(z*wn))
    ln01 = math.log(0.10)
    z_46 = abs(ln01) / math.sqrt(math.pi**2 + ln01**2)
    wn_46_exact = -math.log(0.05 * math.sqrt(1 - z_46**2)) / (z_46 * 1.5)
    k_46 = 0.25 * wn_46_exact**2
    kt_46 = (0.25 * 2 * z_46 * wn_46_exact + 1) / k_46
    check("ch04_ex_guide46_exact_note",
          abs(z_46 - 0.5912) < 0.002 and abs(wn_46_exact - 3.62) < 0.05,
          f"z={z_46:.4f}, exact_env wn={wn_46_exact:.3f} (book uses 3/(z*wn)={3/(z_46*1.5):.3f} -> K={0.25*(3/(z_46*1.5))**2:.2f}, Kt={(0.25*2*z_46*(3/(z_46*1.5))+1)/(0.25*(3/(z_46*1.5))**2):.3f})")

    # Ex 4-7: E(s) = a(Ts + 1 - K*Ki)/(s(Ts^2 + s + K)), lim_{s->0} sE(s) = 0 -> Ki = 1/K
    Ki = sp.Symbol("K_i")
    Phi47 = (Ki * s_c + 1) * (K / (s_c * (T * s_c + 1))) / (1 + K / (s_c * (T * s_c + 1)))
    E47 = sp.simplify((1 - Phi47) * (a / s_c**2))
    ess47 = sp.limit(s_c * E47, s_c, 0)
    check("ch04_ex_guide47", sp.simplify(ess47 - a * (1 - K * Ki) / K) == 0)

    # Ex 4-8: (1) 5s^3+21s^2+4s+10=0 stable, Kv=2.5, ess(t)=0.4; (2) 5s^4+21s^3+4s^2+10s+1=0 unstable (2 RHP)
    check("ch04_ex_guide48",
          count_rhp(5 * s_c**3 + 21 * s_c**2 + 4 * s_c + 10) == 0
          and count_rhp(5 * s_c**4 + 21 * s_c**3 + 4 * s_c**2 + 10 * s_c + 1) == 2)

    # Ex 4-9: s^3+6s^2+5s+K=0 -> 0<K<30, Kv=K/5, ess = 5/K > 5/30 = 1/6
    fc49 = routh_first_column([1, 6, 5, K])
    check("ch04_ex_guide49", sp.simplify(fc49[2] - (30 - K) / 6) == 0)

    # Ex 4-10: G(s)=50(s+1)/(s(s+6)(s+9)), Kv=50/54, r(t)=-2t+20 -> ess = -2/(50/54) = -2.16
    check("ch04_ex_guide410",
          count_rhp(s_c * (s_c + 6) * (s_c + 9) + 50 * (s_c + 1)) == 0
          and abs(-2 / (50 / 54) - (-2.16)) < 1e-12)

    return results


def main() -> int:
    results = run_all_checks()
    failed = 0
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{tag}] {name:34s} {detail}")
    print(f"\nSymPy Verification Summary: {len(results) - failed}/{len(results)} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
