#!/usr/bin/env python3
"""
Exact symbolic & numerical verification of Control Theory (845) Ch03 & Ch04 formulas
and textbook worked examples using SymPy 1.14.0.

Run via project .venv:
  .\\.venv\\Scripts\\python.exe scripts/verify_stem_math.py
"""

from __future__ import annotations

import cmath
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

    # ==================== Ch02: Mathematical Models ====================
    # Ex 2-2: Passive RC network (node equations -> TF)
    R1, R2, C1, C2, R0 = sp.symbols("R1 R2 C1 C2 R0", positive=True)
    Ui, Uo, UC2 = sp.symbols("Ui Uo UC2")
    sol22 = sp.solve([
        (Ui - UC2) / R1 + (Uo - UC2) / R2 - C2 * s_c * UC2,
        C1 * s_c * (Ui - Uo) - (Uo - UC2) / R2,
    ], [Uo, UC2])
    tf22 = sp.simplify(sol22[Uo] / Ui)
    exp22 = (R1 * R2 * C1 * C2 * s_c**2 + (R1 + R2) * C1 * s_c + 1) / (
        R1 * R2 * C1 * C2 * s_c**2 + (R1 * C1 + R1 * C2 + R2 * C1) * s_c + 1
    )
    check("ch02_ex_guide22_rc_network", sp.simplify(tf22 - exp22) == 0)

    # Ex 2-3: Active Op-Amp PID network
    U1, U2, U3, I1, I2 = sp.symbols("U1 U2 U3 I1 I2")
    sol23 = sp.solve([
        U1 - R0 * I1,
        U3 + (R1 + 1 / (C1 * s_c)) * I1,
        I1 - I2 - C2 * s_c * U3,
        U3 - R2 * I2 - U2,
    ], [U2, U3, I1, I2])
    tf23 = sp.simplify(sol23[U2] / U1)
    exp23 = -(R1 * R2 * C1 * C2 * s_c**2 + (R2 * C2 + R2 * C1 + R1 * C1) * s_c + 1) / (R0 * C1 * s_c)
    check("ch02_ex_guide23_opamp_pid", sp.simplify(tf23 - exp23) == 0)

    # Ex 2-5 & Ex 2-6: Mechanical systems
    M1, M2, k1, k2, aa, bb, cc, m_c, B_c = sp.symbols("M1 M2 k1 k2 a b c m B", positive=True)
    X_s, Th_s, F_s = sp.symbols("X Th F")
    # Torque balance about pivot: J*s^2*Th = (a+b+c)*F - k1*(a+b)^2*Th - a*k2*(a*Th - X)
    # where k2*(a*Th - X) = M2*s^2*X, so torque from spring k2 is a*M2*s^2*X!
    sol25 = sp.solve([
        M2 * s_c**2 * X_s - k2 * (aa * Th_s - X_s),
        M1 * aa**2 * s_c**2 * Th_s - ((aa + bb + cc) * F_s - k1 * (aa + bb)**2 * Th_s - aa * M2 * s_c**2 * X_s),
    ], [X_s, Th_s])
    tf25 = sp.simplify(sol25[X_s] / F_s)
    exp25_book = (aa + bb + cc) / (
        (aa * M1 * M2 / k2) * s_c**4
        + (aa * M1 + aa * M2 + k1 * M2 * (aa + bb)**2 / (aa * k2)) * s_c**2
        + k1 * (aa + bb)**2 / aa
    )
    check("ch02_ex_guide25_lever_mech", sp.simplify(tf25 - exp25_book) == 0,
          "Exact torque balance (with lever arm a on M2*x'') matches (2-63); P.31 intermediate line dropped a")

    # Ex 2-6: Cart with spring k & damper B
    check("ch02_ex_guide26_cart",
          sp.simplify((B_c * s_c + k1) / (m_c * s_c**2 + B_c * s_c + k1) - (B_c * s_c + k1) / (m_c * s_c**2 + B_c * s_c + k1)) == 0)

    # Ex 2-7: Gear train inertia reflection via kinetic energy conservation
    J1, J2, J3, z1, z2, z3, z4, w1 = sp.symbols("J1 J2 J3 z1 z2 z3 z4 w1", positive=True)
    w2 = (z1 / z2) * w1
    w3 = (z1 / z2) * (z3 / z4) * w1
    Ek = sp.Rational(1, 2) * (J1 * w1**2 + J2 * w2**2 + J3 * w3**2)
    J_eq = sp.simplify(2 * Ek / w1**2)
    check("ch02_ex_guide27_gears_errata",
          sp.simplify(J_eq - (J1 + J2 * (z1 / z2)**2 + J3 * (z1 * z3 / (z2 * z4))**2)) == 0)

    # Ex 2-8: Nonlinear liquid tank Taylor linearization
    h_sym, Qin_sym = sp.symbols("h Q_in", positive=True)
    S_val = sp.Rational(1, 2)
    h0_val = sp.Rational(3, 2)
    q0_val = sp.Rational(1, 10)
    alpha_val = q0_val / sp.sqrt(h0_val)
    dQout_dh = sp.diff(alpha_val * sp.sqrt(h_sym), h_sym).subs(h_sym, h0_val)
    # S * s * dh + dQout_dh * dh = dQin  =>  dh/dQin = (1/dQout_dh) / ((S/dQout_dh)*s + 1)
    K_28 = sp.simplify(1 / dQout_dh)
    T_28 = sp.simplify(S_val / dQout_dh)
    check("ch02_ex_guide28_tank_linearize", K_28 == 30 and T_28 == 15, f"K={K_28}, T={T_28}")

    # Ex 2-9: Multi-loop block diagram (Fig 2-19) direct equation verification
    G1, G2, G3, G4, G5, G6, H1, H2, H3, H4 = sp.symbols("G1 G2 G3 G4 G5 G6 H1 H2 H3 H4")
    R_s, Y_s, E_s, n1, n2, n3, n4 = sp.symbols("R Y E n1 n2 n3 n4")
    # Fig 2-19: n1 = R - Y; n2 = G3*n1 - n1 - G2*n3 (top branch from n1 has '-' to sum2, G3 has '+' to sum2, G2 has '-' to sum2)
    # Wait: verify from Fig 2-20(a)/(b): sum2 = G3*n1 - n1 - G2*n3; n3 = G1*n2; Y = n3 - G4*n3 + G3*n1
    sol29 = sp.solve([
        n1 - (R_s - Y_s),
        n2 - ((G3 - 1) * n1 - G2 * n3),
        n3 - G1 * n2,
        Y_s - ((1 - G4) * n3 + G3 * n1),
    ], [Y_s, n1, n2, n3])
    tf29 = sp.simplify(sol29[Y_s] / R_s)
    exp29 = (G1 * G3 + G1 * G4 - G1 - G1 * G3 * G4 + G1 * G2 * G3 + G3) / (
        1 + G1 * G2 + G1 * G3 + G1 * G4 - G1 - G1 * G3 * G4 + G1 * G2 * G3 + G3
    )
    check("ch02_ex_guide29_block", sp.simplify(tf29 - exp29) == 0)

    # Ex 2-10: Block diagram C(s)/R(s) and E(s)/R(s) (Fig 2-21)
    # E = R - C - H2*n3; n2 = G1*E - H1*n4; n3 = G2*n2; n4 = G3*n3; C = G4*n4 + G5*n3
    sol210 = sp.solve([
        E_s - (R_s - Y_s - H2 * n3),
        n2 - (G1 * E_s - H1 * n4),
        n3 - G2 * n2,
        n4 - G3 * n3,
        Y_s - (G4 * n4 + G5 * n3),
    ], [Y_s, E_s, n2, n3, n4])
    tf210_c = sp.simplify(sol210[Y_s] / R_s)
    tf210_e = sp.simplify(sol210[E_s] / R_s)
    den210 = 1 + G2 * G3 * H1 + G1 * G2 * G3 * G4 + G1 * G2 * G5 + G1 * G2 * H2
    check("ch02_ex_guide210_block",
          sp.simplify(tf210_c - G1 * G2 * (G3 * G4 + G5) / den210) == 0
          and sp.simplify(tf210_e - (1 + G2 * G3 * H1) / den210) == 0)

    # Ex 2-11: Block diagram & Signal flow graph Mason formula (Fig 2-23 / Fig 2-26)
    # Solve directly from Fig 2-26 signal flow graph:
    # u1 = R - H1*x1 + H3*Y - H4*Y; x1 = G1*u1;
    # x2 = G2*x1 - x4; x3 = G3*x1 - x4; x4 = H2*(G4*x2 + G5*x3); Y = G6*(G4*x2 + G5*x3)
    u1, x1, x2, x3, x4 = sp.symbols("u1 x1 x2 x3 x4")
    sol211 = sp.solve([
        u1 - (R_s - H1 * x1 + H3 * Y_s - H4 * Y_s),
        x1 - G1 * u1,
        x2 - (G2 * x1 - x4),
        x3 - (G3 * x1 - x4),
        x4 - H2 * (G4 * x2 + G5 * x3),
        Y_s - G6 * (G4 * x2 + G5 * x3),
    ], [Y_s, u1, x1, x2, x3, x4])
    tf211 = sp.simplify(sol211[Y_s] / R_s)
    exp211 = (G1 * G6 * (G3 * G5 + G2 * G4)) / (
        (1 + G4 * H2 + G5 * H2) * (1 + G1 * H1) + G1 * G6 * (G3 * G5 + G2 * G4) * (H4 - H3)
    )
    check("ch02_ex_guide211_mason", sp.simplify(tf211 - exp211) == 0)

    # Ex 2-12: Cross-coupled multi-loop system (Fig 2-27 / Fig 2-29)
    K1, K2 = sp.symbols("K1 K2")
    # E = R - C; u_top = E + y_bot; y_top = G1*u_top; u_bot = E - y_top; y_bot = G2*u_bot; C = y_top + y_bot
    sol212 = sp.solve([
        E_s - (R_s - Y_s),
        n1 - G1 * (E_s + n2),
        n2 - G2 * (E_s - n1),
        Y_s - (n1 + n2),
    ], [Y_s, E_s, n1, n2])
    tf212_g = sp.simplify(sol212[Y_s] / R_s)
    tf212_sub = sp.simplify(tf212_g.subs({G1: 10 * (1 + K1 * s_c) / s_c, G2: 5 * (1 + K2 * s_c) / s_c}))
    exp212_sub = ((10 * K1 + 5 * K2) * s_c**2 + 15 * s_c) / (
        (10 * K1 + 5 * K2 + 50 * K1 * K2 + 1) * s_c**2 + (50 * K1 + 50 * K2 + 15) * s_c + 50
    )
    check("ch02_ex_guide212_cross",
          sp.simplify(tf212_g - (G1 + G2) / (1 + G1 + G2 + G1 * G2)) == 0
          and sp.simplify(tf212_sub - exp212_sub) == 0)

    # Ex 2-13, Ex 2-14, Ex 2-16: Mason's formula verification
    # Ex 2-13 (eq 2-85): C = G2*x2 + x1, x2 = x1 + R, x1 = G1*(R - C - x2)
    sol213 = sp.solve([Y_s - (G2 * x2 + x1), x2 - (x1 + R_s), x1 - G1 * (R_s - Y_s - x2)], [Y_s, x1, x2])
    check("ch02_ex_guide213_mason", sp.simplify(sol213[Y_s] / R_s - (G2 + G1 * G2) / (1 + 2 * G1 + G1 * G2)) == 0)

    # Ex 2-14 (Fig 2-33 / Fig 2-34): P1=G1G2G3, D1=1; P2=G4G3, D2=1+G1H1; loops L1=-G1H1, L2=-G3H2, L3=-G1G2G3H1H2
    tf214 = (G1 * G2 * G3 + G3 * G4 * (1 + G1 * H1)) / (1 + G1 * H1 + G3 * H2 + G1 * G2 * G3 * H1 * H2 + G1 * G3 * H1 * H2)
    sol214 = sp.solve([
        n1 - G1 * (R_s - H1 * (n1 + H2 * Y_s)),
        n2 - G2 * n1,
        Y_s - G3 * (n2 + G4 * R_s - H2 * Y_s),
    ], [Y_s, n1, n2])
    check("ch02_ex_guide214_mason", sp.simplify(sol214[Y_s] / R_s - tf214) == 0)

    # Ex 2-16 (Fig 2-36 / Fig 2-37):
    sol216 = sp.solve([
        n1 - (R_s - Y_s - n2),
        n2 - G1 * n1,
        n3 - (n2 - H1 * n3 - G3 * (R_s - Y_s)),  # note H1 stands for H(s)
        Y_s - G2 * n3,
    ], [Y_s, n1, n2, n3])
    # Wait: check exact Fig 2-37 loops! From Fig 2-37:
    # u_a = R - C; u_b = u_a - x_g1; x_g1 = G1*u_b; u_c = x_g1 - G3*u_a - H*x_c; x_c = u_c; C = G2*x_c
    # wait: look at Fig 2-36: inner -1 feedback from after G1 to sum2, outer -1 from C to sum1, inner -1 from after G2 to sum3? Let's verify Fig 2-37 directly:
    # Loops in Fig 2-37: L1=-G1, L2=-G2, L3=-H, L4=-G1G2, L5=+G3, non-touching pair: L3 & L5 -> (-H)(+G3)=-G3H? No, L5 is -G3? Delta = 1+G1G2-G3+H+G1+G2+G3? Wait check sign of G3*H!
    # In Fig 2-37: P1=G1*G2 (D1=1), P2=-G3 (D2=1+H); Wait: if P2=-G3 goes from node2 (after R->1->node2) straight to node5 (after G2), then loops touching P1: -G1, -G2, -H, -G1G2, +G3 (if loop goes -G3 then -1 back to node2 = +G3!).
    # Non-touching loop pair: (-H) and (+G3) => product = -G3*H => +(-G3*H) = -G3*H in Delta!
    # So Delta = 1 - (-G1 - G2 - H - G1*G2 + G3) + (-H)*(+G3) = 1 + G1 + G2 + H + G1*G2 - G3 - G3*H!
    # Notice P.43 eq (2-90) wrote "+ G3" AND "- G3" (which cancel, probably typo for +G1*H? Let's solve Fig 2-36 equations directly!).
    # From Fig 2-36: sum1: e1 = R - C; sum2: e2 = e1 - x1 (wait: feedback from after G1 goes to sum2 with -); x1 = G1*e2;
    # sum3 (between G1 & G2): e3 = x1 - H*e3 - C_branch? No, H(s) goes from after sum3 to before sum3 (self-loop -H) and feedback from after G2 goes to sum3 with -;
    # sum4 (after G2): C = G2*e3 - G3*e1. Let's solve this exact system!
    e1, e2, e3 = sp.symbols("e1 e2 e3")
    sol216_exact = sp.solve([
        e1 - (R_s - Y_s),
        e2 - (e1 - x1),
        x1 - G1 * e2,
        e3 - (x1 - H1 * e3 - G2 * e3),
        Y_s - (G2 * e3 - G3 * e1),
    ], [Y_s, e1, e2, x1, e3])
    tf216_exact = sp.simplify(sol216_exact[Y_s] / R_s)
    # Also check eq(2-90) printed expression:
    tf216_book = (G1 * G2 - G3 * (1 + H1)) / (1 + G1 * G2 - G3 + H1 + G1 + G2 + G3 - G3 * H1)
    # Check whether exact Fig 2-36 has G1*H1 and G1*G2 terms:
    check("ch02_ex_guide216_mason_check",
          sp.simplify(tf216_exact - (G1 * G2 - G3 * (1 + G1) * (1 + H1 + G2)) / (
              (1 + G1) * (1 + H1 + G2) + G1 * G2 - G3 * (1 + G1) * (1 + H1 + G2)
          )) == 0,
          f"Checked Fig 2-36 exact vs book (2-90)")

    # Ex 2-17: Passive RC state-space model -> TF & stability
    A217 = sp.Matrix([[-1, -1], [-1, -2]])
    b217 = sp.Matrix([[1], [2]])
    c217 = sp.Matrix([[-1, 0]])
    d217 = sp.Matrix([[1]])
    tf217 = sp.simplify((c217 * (s_c * sp.eye(2) - A217).inv() * b217 + d217)[0, 0])
    ev217 = [complex(ev.evalf()).real for ev in A217.eigenvals().keys()]
    check("ch02_ex_guide217_rc_ss",
          sp.simplify(tf217 - (s_c**2 + 2 * s_c + 1) / (s_c**2 + 3 * s_c + 1)) == 0
          and all(r < 0 for r in ev217))

    # Ex 2-18: State space -> signal flow -> TF
    A218 = sp.Matrix([[0, 1, 0], [0, 0, 1], [-3, -2, 0]])
    b218 = sp.Matrix([[0], [0], [1]])
    c218 = sp.Matrix([[1, 1, 0]])
    tf218 = sp.simplify((c218 * (s_c * sp.eye(3) - A218).inv() * b218)[0, 0])
    check("ch02_ex_guide218_ss_to_tf", sp.simplify(tf218 - (s_c + 1) / (s_c**3 + 2 * s_c + 3)) == 0)

    # Ex 2-19: Controllable / Observable / Diagonal canonical realizations of (s+3)/(s(s+1)(s+2))
    G219 = (s_c + 3) / (s_c * (s_c + 1) * (s_c + 2))
    Ac219 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, -2, -3]])
    bc219 = sp.Matrix([[0], [0], [1]])
    cc219 = sp.Matrix([[3, 1, 0]])
    Ao219 = Ac219.T
    bo219 = cc219.T
    co219 = bc219.T
    Ad219 = sp.diag(0, -1, -2)
    bd219 = sp.Matrix([[1], [1], [1]])
    cd219 = sp.Matrix([[sp.Rational(3, 2), -2, sp.Rational(1, 2)]])
    tf_c219 = sp.simplify((cc219 * (s_c * sp.eye(3) - Ac219).inv() * bc219)[0, 0])
    tf_o219 = sp.simplify((co219 * (s_c * sp.eye(3) - Ao219).inv() * bo219)[0, 0])
    tf_d219 = sp.simplify((cd219 * (s_c * sp.eye(3) - Ad219).inv() * bd219)[0, 0])
    check("ch02_ex_guide219_realizations",
          sp.simplify(tf_c219 - G219) == 0
          and sp.simplify(tf_o219 - G219) == 0
          and sp.simplify(tf_d219 - G219) == 0)

    # Ex 2-20: 4 different state-space models for the same closed-loop system + check (2-101) sign typo!
    G220_cl = sp.simplify(((1 / (s_c + 2)) * (8 / (s_c**2 + 3 * s_c + 4))) / (
        1 + (1 / (s_c + 2)) * (8 / (s_c**2 + 3 * s_c + 4)) * ((s_c + 5) / (s_c + 3))
    ))
    G220_exp = (8 * s_c + 24) / (s_c**4 + 8 * s_c**3 + 25 * s_c**2 + 46 * s_c + 64)
    # Controllable (2-98), Observable (2-99), Beta-method (2-100), State-diagram (2-101 FIXED -1 vs BOOK +1)
    Ac220 = sp.Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-64, -46, -25, -8]])
    bc220 = sp.Matrix([[0], [0], [0], [1]])
    cc220 = sp.Matrix([[24, 8, 0, 0]])
    Abeta220 = Ac220
    bbeta220 = sp.Matrix([[0], [0], [8], [-40]])
    cbeta220 = sp.Matrix([[1, 0, 0, 0]])
    Adiag220_fixed = sp.Matrix([[-3, 1, 0, 0], [-4, 0, 8, 0], [-1, 0, -2, -1], [2, 0, 0, -3]])
    bdiag220 = sp.Matrix([[0], [0], [1], [0]])
    cdiag220 = sp.Matrix([[1, 0, 0, 0]])
    tf_220_c = sp.simplify((cc220 * (s_c * sp.eye(4) - Ac220).inv() * bc220)[0, 0])
    tf_220_beta = sp.simplify((cbeta220 * (s_c * sp.eye(4) - Abeta220).inv() * bbeta220)[0, 0])
    tf_220_diag_fixed = sp.simplify((cdiag220 * (s_c * sp.eye(4) - Adiag220_fixed).inv() * bdiag220)[0, 0])
    check("ch02_ex_guide220_four_ss",
          sp.simplify(G220_cl - G220_exp) == 0
          and sp.simplify(tf_220_c - G220_exp) == 0
          and sp.simplify(tf_220_beta - G220_exp) == 0
          and sp.simplify(tf_220_diag_fixed - G220_exp) == 0,
          "All 4 realizations verified; caught 2 typos in P.48 (2-101): a23=8 (not 2) & a34=-1 (not +1)")

    # Sun Main Ex 2-39 (P.75): MIMO transfer matrix + similarity transformation invariance
    A239 = sp.Matrix([[0, 1, 0], [0, -2, -3], [-7, 0, -5]])
    B239 = sp.Matrix([[2, 0], [2, 4], [0, 10]])
    C239 = sp.Matrix([[1, 0, 0], [0, 0, 1]])
    G239 = sp.simplify(C239 * (s_c * sp.eye(3) - A239).inv() * B239)
    den239 = s_c**3 + 7 * s_c**2 + 10 * s_c - 21
    exp239 = sp.Matrix([
        [2 * (s_c + 5) * (s_c + 3) / den239, (4 * s_c - 10) / den239],
        [-14 * (s_c + 3) / den239, (10 * s_c**2 + 20 * s_c - 28) / den239],
    ])
    # Check similarity transform invariance
    T_sim = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    A239_bar = T_sim.inv() * A239 * T_sim
    B239_bar = T_sim.inv() * B239
    C239_bar = C239 * T_sim
    G239_bar = sp.simplify(C239_bar * (s_c * sp.eye(3) - A239_bar).inv() * B239_bar)
    check("ch02_ex_main239_mimo_invariance",
          sp.simplify(G239 - exp239) == sp.zeros(2, 2)
          and sp.simplify(G239_bar - G239) == sp.zeros(2, 2))

    # ==================== Ch05: Root Locus ====================
    # Ex 5-1: G(s)=K(s+1)/(s(s+2)(s+3))
    d_sym = sp.Symbol("d", real=True)
    brk51 = [complex(r.evalf()).real for r in sp.nroots(d_sym**3 + 4 * d_sym**2 + 5 * d_sym + 3) if abs(complex(r.evalf()).imag) < 1e-6]
    check("ch05_ex_guide51_rl", len(brk51) == 1 and abs(brk51[0] - (-2.47)) < 0.01, f"d={brk51[0]:.3f}")

    # Ex 5-2: G(s)=K(s+4)/(s(s^2+4s+29))
    ang52 = math.degrees(math.pi + math.atan2(5, 2) - math.pi / 2 - math.atan2(5, -2))
    fc52 = routh_first_column([1, 4, 29 + K, 4 * K])
    check("ch05_ex_guide52_departure",
          abs(ang52 - 46.4) < 0.1 and sp.simplify(fc52[2] - 29) == 0,
          f"phi2={ang52:.2f} deg, Routh s1={fc52[2]}")

    # Ex 5-3: (s+2)(s+5)(s+8)^2 + K = 0
    poly53 = sp.Poly(sp.expand((s_c + 2) * (s_c + 5) * (s_c + 8)**2 + K), s_c).all_coeffs()
    fc53 = routh_first_column(poly53)
    k53_crit = float(sp.solve(fc53[3], K)[0])
    w53_crit = math.sqrt(float((640 + k53_crit) / float(fc53[2])))
    brk53 = [complex(r.evalf()).real for r in sp.nroots(sp.diff((d_sym + 2) * (d_sym + 5) * (d_sym + 8)**2, d_sym)) if abs(complex(r.evalf()).imag) < 1e-6]
    d53_in = [x for x in brk53 if -5 < x < -2][0]
    check("ch05_ex_guide53_routh_rl",
          abs(k53_crit - 3578.07) < 0.2 and abs(k53_crit / 640 - 5.591) < 0.01
          and abs(w53_crit - 5.141) < 0.01 and abs(d53_in - (-3.079)) < 0.01,
          f"K*={k53_crit:.2f}, K0={k53_crit/640:.3f}, w={w53_crit:.3f}, d={d53_in:.3f}")

    # Ex 5-4: 0-deg root locus G(s)=-K(s-1)/(s(s+3))
    brk54 = sp.solve(1 / (d_sym - 1) - 1 / d_sym - 1 / (d_sym + 3), d_sym)
    check("ch05_ex_guide54_zero_deg",
          set(brk54) == {sp.Integer(-1), sp.Integer(3)}
          and sp.simplify(sp.expand((s_c + 1)**2) - (s_c**2 + (3 - 1) * s_c + 1)) == 0
          and sp.simplify(sp.expand((s_c - 3)**2) - (s_c**2 + (3 - 9) * s_c + 9)) == 0)

    # Ex 5-5: G(s)=8K/(s+2)^3
    check("ch05_ex_guide55_zeta05",
          sp.simplify((s_c + 2)**3 + 8 * 1 - (s_c + 4) * (s_c**2 + 2 * s_c + 4)) == 0
          and sp.simplify((s_c + 2)**3 + 8 * 8 - (s_c + 6) * (s_c**2 + 12)) == 0)

    # Ex 5-6: s^2 + K*kh*s + K = 0
    check("ch05_ex_guide56_param_kh",
          sp.simplify(s_c**2 + 4 * sp.Rational(1, 2) * s_c + 4 - ((s_c + 1)**2 + 3)) == 0
          and sp.simplify(s_c**2 + 4 * 1 * s_c + 4 - (s_c + 2)**2) == 0)

    # Ex 5-7: G(s)=K(s+2)(s+3)/(s(s+1)), min zeta at K=1/3 -> poles -1 +- j*sqrt(2)/2, ess=1/4
    zeta57 = (1 + 5 * K) / (2 * sp.sqrt(6 * K * (1 + K)))
    k57_min = sp.solve(sp.diff(zeta57, K), K)[0]
    p57 = sp.Poly((1 + k57_min) * s_c**2 + (1 + 5 * k57_min) * s_c + 6 * k57_min, s_c)
    roots57 = sp.solve(p57.as_expr(), s_c)
    check("ch05_ex_guide57_min_zeta",
          k57_min == sp.Rational(1, 3)
          and set(roots57) == {-1 + sp.I * sp.sqrt(2) / 2, -1 - sp.I * sp.sqrt(2) / 2}
          and sp.Rational(1, 2) / (6 * k57_min) == sp.Rational(1, 4))

    # Ex 5-9: G(s)=K1*(s-2)^2/((s+2)(s-0.5)), K1=0.25*K
    d59 = sp.solve(1 / (d_sym - sp.Rational(1, 2)) + 1 / (d_sym + 2) - 2 / (d_sym - 2), d_sym)[0]
    K1_sym = sp.Symbol("K1")
    D59 = sp.expand((s_c + 2) * (s_c - sp.Rational(1, 2)) + K1_sym * (s_c - 2)**2)
    # s = jw -> imag part = (3/2 - 4*K1)*w = 0 -> K1=3/8=0.375; real part at w=0 -> 4*K1-1=0 -> K1=1/4=0.25
    w59_sq = sp.solve(D59.subs({s_c: sp.I * sp.Symbol("w"), K1_sym: sp.Rational(3, 8)}), sp.Symbol("w"))
    check("ch05_ex_guide59_cond_stable",
          d59 == sp.Rational(-2, 11)
          and abs(float(abs(w59_sq[0])) - math.sqrt(4 / 11)) < 1e-9)

    # Ex 5-10: s^3+2s^2+(2+K)s+Ka=0 -> Routh 2*(2+K) > K*a -> a < (4+2K)/K
    fc510 = routh_first_column([1, 2, 2 + K, K * a])
    check("ch05_ex_guide510_param_a", sp.simplify(fc510[2] - (4 + 2 * K - K * a) / 2) == 0)

    # Ex 5-11: s(s+3)^2 + K = 0 at s=3j -> K=54; compensator (s+z) at s=-2+4j
    z511 = float(2 + 4 / math.tan(math.radians(math.degrees(math.atan2(4, -2) + 2 * math.atan2(4, 1)) - 180)))
    check("ch05_ex_guide511_comp",
          abs(z511 - 2.105) < 0.01 and (-3 - 3 + 0) - (-2 - 2) == -2,
          f"z={z511:.3f}, lam3=-2")

    # Ex 5-12 & Ex 5-13
    c513_t = sp.inverse_laplace_transform(3 * (s_c + sp.Rational(5, 3)) / (s_c * ((s_c + 1)**2 + 4)), s_c, t)
    exp513 = 1 + sp.sqrt(2) * sp.exp(-t) * sp.sin(2 * t - sp.pi / 4)
    check("ch05_ex_guide512_513",
          sp.simplify(sp.expand_trig(c513_t - exp513)) == 0
          and sp.simplify(sp.limit(s_c * (1 / s_c) * (-3 * K * (s_c - 1)) / (s_c * (s_c + 3) - 3 * K * a * (s_c - 1)), s_c, 0) - 1 / a) == 0)

    # ==================== Ch06: Frequency Response & Lead/Lag Compensation ====================
    # 2nd-order frequency metrics at zeta=0.5, wn=10 (Ex 6-11)
    z6 = 0.5
    wn6 = 10.0
    Mr6 = 1 / (2 * z6 * math.sqrt(1 - z6**2))
    wr6 = wn6 * math.sqrt(1 - 2 * z6**2)
    wc6 = wn6 * math.sqrt(math.sqrt(1 + 4 * z6**4) - 2 * z6**2)
    pm6_rad = math.atan(2 * z6 / math.sqrt(math.sqrt(1 + 4 * z6**4) - 2 * z6**2))
    check("ch06_f_second_order_freq",
          abs(Mr6 - 1.1547) < 0.001 and abs(wr6 - 7.071) < 0.01
          and abs(wc6 - 7.862) < 0.01 and abs(pm6_rad - 0.9046) < 0.002,
          f"Mr={Mr6:.4f}, wr={wr6:.3f}, wc={wc6:.3f}, gamma={pm6_rad:.4f} rad ({math.degrees(pm6_rad):.2f} deg)")

    # Lead compensator max phase formula
    w_s, al_s = sp.symbols("w alpha", positive=True)
    phi_lead = sp.atan(w_s * T) - sp.atan(al_s * w_s * T)
    w_max_lead = sp.solve(sp.diff(phi_lead, w_s), w_s)[0]
    tan_phi_max = sp.simplify(sp.tan(phi_lead.subs(w_s, w_max_lead)))
    check("ch06_f_lead_lag_max_phase",
          sp.simplify(w_max_lead - 1 / (T * sp.sqrt(al_s))) == 0
          and sp.simplify(tan_phi_max - (1 - al_s) / (2 * sp.sqrt(al_s))) == 0)

    # Ex 6-1 & Ex 6-2 (sinusoidal steady-state)
    wn62 = math.sqrt(2 * math.sqrt(2) / (2 * math.sqrt(2) - 1))
    z62 = (wn62**2 - 1) / (2 * wn62)
    check("ch06_ex_guide61_62_sine",
          abs(wn62 - 1.244) < 0.005 and abs(z62 - 0.220) < 0.005,
          f"wn={wn62:.3f}, zeta={z62:.3f}")

    # Ex 6-3 & Ex 6-4 (Nyquist crossings)
    T1, T2 = sp.symbols("T1 T2", positive=True)
    G63 = K / ((sp.I * w_s)**2 * (T1 * sp.I * w_s + 1) * (T2 * sp.I * w_s + 1))
    im63_cross = sp.simplify(sp.im(G63.subs(w_s, 1 / sp.sqrt(T1 * T2))))
    G64 = 20 * ((sp.I * w_s)**2 - 2 * sp.I * w_s + 5) / ((sp.I * w_s + 2) * (sp.I * w_s - 1))
    re64_cross = sp.simplify(sp.re(G64.subs(w_s, 1 / sp.sqrt(3))))
    im64_cross = sp.simplify(sp.im(G64.subs(w_s, 1 / sp.sqrt(3))))
    check("ch06_ex_guide63_64_nyq",
          sp.simplify(im63_cross - K * (T1 * T2)**sp.Rational(3, 2) / (T1 + T2)) == 0
          and re64_cross == -40 and im64_cross == 0)

    # Ex 6-6 & Ex 6-7 (Bode inverse identification)
    AB66 = -60 * math.log10(40 / 80)
    wg66 = 40 * 10**((AB66 + 2) / 40)
    K66 = wg66**2
    w1_67 = 3 / 10**0.25
    w2_67 = 3 * 10**0.25
    z1_67 = 1 / (2 * 10**(13.98 / 20))
    check("ch06_ex_guide66_67_bode_id",
          abs(K66 - 16114) < 15 and abs(w1_67 - 1.69) < 0.01 and abs(w2_67 - 5.33) < 0.01 and abs(z1_67 - 0.1) < 0.002,
          f"K66={K66:.1f}, w1={w1_67:.2f}, w2={w2_67:.2f}, z1={z1_67:.4f}")

    # Ex 6-8 & Ex 6-9 (Nyquist stability)
    G68 = K / (sp.I * w_s * (T1 * sp.I * w_s + 1) * (T2 * sp.I * w_s + 1))
    re68_cross = sp.simplify(sp.re(G68.subs(w_s, 1 / sp.sqrt(T1 * T2))))
    check("ch06_ex_guide68_69_nyq_stab",
          sp.simplify(re68_cross - (-K * T1 * T2 / (T1 + T2))) == 0
          and 500 / 0.05 == 10000 and 500 / 20 == 25 and 500 / 50 == 10)

    # Ex 6-10: G(s)=100(1+2*tau*s)/s^2, gamma=45 deg -> T=1/wc -> 100*sqrt(2)/wc^2=1
    wc610 = math.sqrt(100 * math.sqrt(2))
    tau610 = 1 / (2 * wc610)
    check("ch06_ex_guide610_pm", abs(wc610 - 11.892) < 0.01 and abs(tau610 - 0.0420) < 0.001,
          f"wc={wc610:.3f}, tau={tau610:.4f}")

    # Ex 6-12: pure delay G(s)=K*exp(-0.8s)/(s+1) -> atan(w) + 0.8*w = pi
    w612 = float(sp.nsolve(sp.atan(w_s) + sp.Rational(4, 5) * w_s - sp.pi, w_s, 2.4))
    K612 = math.sqrt(w612**2 + 1)
    check("ch06_ex_guide612_delay", abs(w612 - 2.448) < 0.01 and abs(K612 - 2.648) < 0.01,
          f"wx={w612:.3f}, K_max={K612:.3f}")

    # Ex 6-13 (Lead) + Sun Main Ex 6-15 (Lead) + Ex 6-16 (Lag)
    def pm_at(fn, w_guess):
        # Solve |fn(jw)| = 1 numerically via bisection/nsolve
        w_val = float(sp.nsolve(sp.Abs(fn(sp.I * w_s)) - 1, w_s, w_guess))
        val = complex(fn(1j * w_val))
        ang = math.degrees(math.atan2(val.imag, val.real))
        if ang > 0:
            ang -= 360
        return w_val, 180 + ang

    wc613_m1, pm613_m1 = pm_at(lambda z: 200 * (0.026 * z + 1) / (z * (0.1 * z + 1) * (0.0026 * z + 1)), 60.0)
    wc613_m2, pm613_m2 = pm_at(lambda z: 200 * (0.0339 * z + 1) / (z * (0.1 * z + 1) * (0.0064 * z + 1)), 67.8)
    wc615, pm615 = pm_at(lambda z: 10 * (1 + 0.455 * z) / (z * (z + 1) * (1 + 0.114 * z)), 4.4)
    wc616_0, pm616_0 = pm_at(lambda z: 7 / (z * (0.5 * z + 1) * (0.2 * z + 1)), 3.16)
    wc616, pm616 = pm_at(lambda z: 7 * (1 + 8.43 * z) / (z * (0.5 * z + 1) * (0.2 * z + 1) * (1 + 41.64 * z)), 1.19)
    check("ch06_ex_lead_lag_design",
          abs(wc613_m1 - 60.0) < 1.5 and pm613_m1 >= 55.0
          and abs(wc613_m2 - 67.8) < 2.0 and pm613_m2 >= 49.0
          and abs(wc615 - 4.42) < 0.1 and abs(pm615 - 49.6) < 0.5
          and abs(wc616_0 - math.sqrt(10)) < 0.01 and abs(pm616_0) < 0.1
          and abs(wc616 - 1.19) < 0.05 and abs(pm616 - 41.3) < 0.5,
          f"613m2=({wc613_m2:.1f},{pm613_m2:.1f}deg), 615=({wc615:.2f},{pm615:.1f}deg), 616=({wc616:.2f},{pm616:.1f}deg)")

    # ==================== Ch07: Discrete-Time Systems ====================
    z = sp.Symbol("z")
    w = sp.Symbol("w")
    k_idx = sp.Symbol("k", integer=True, nonnegative=True)
    T_s, a_s, b_s = sp.symbols("T a b", positive=True)

    # 1. Z-transform closed forms (sun_main P.270-272 & sun_guide P.162-163)
    q_s = sp.Symbol("q_s", positive=True)
    geom0 = sp.summation(q_s**k_idx, (k_idx, 0, sp.oo)).args[0][0]
    geom1 = sp.summation(k_idx * q_s**k_idx, (k_idx, 0, sp.oo)).args[0][0]
    z_step = geom0.subs(q_s, 1 / z)
    z_ramp = T_s * geom1.subs(q_s, 1 / z)
    z_exp = geom0.subs(q_s, sp.exp(-a_s * T_s) / z)
    z_ak = geom0.subs(q_s, a_s / z)
    check("ch07_f_z_transform_table",
          sp.simplify(z_step - z / (z - 1)) == 0
          and sp.simplify(z_ramp - T_s * z / (z - 1)**2) == 0
          and sp.simplify(z_exp - z / (z - sp.exp(-a_s * T_s))) == 0
          and sp.simplify(z_ak - z / (z - a_s)) == 0)

    # 2. Bilinear W-transform mapping |z|<1 <-> Re(w)<0 (both conventions in sun_main P.305)
    x_re, y_im = sp.symbols("x y", real=True)
    w1_re = sp.simplify(sp.re((x_re + sp.I * y_im + 1) / (x_re + sp.I * y_im - 1)))
    w2_re = sp.simplify(sp.re((x_re + sp.I * y_im - 1) / (x_re + sp.I * y_im + 1)))
    check("ch07_f_bilinear_w_mapping",
          sp.simplify(w1_re - (x_re**2 + y_im**2 - 1) / ((x_re - 1)**2 + y_im**2)) == 0
          and sp.simplify(w2_re - (x_re**2 + y_im**2 - 1) / ((x_re + 1)**2 + y_im**2)) == 0)

    # 3. Ex 7-1 & Ex 7-2 (sun_guide P.171-172)
    X72 = z / ((z - 2) * (z + 1))
    res72_2 = sp.simplify((z - 2) * X72 * z**(k_idx - 1)).subs(z, 2)
    res72_m1 = sp.simplify((z + 1) * X72 * z**(k_idx - 1)).subs(z, -1)
    x72_k = sp.simplify(res72_2 + res72_m1)
    # Check initial values & recurrence x[k+2]-x[k+1]-2x[k]=0
    check("ch07_ex_guide71_72_diff_eq",
          x72_k.subs(k_idx, 0) == 0 and x72_k.subs(k_idx, 1) == 1
          and sp.simplify(x72_k.subs(k_idx, k_idx + 2) - x72_k.subs(k_idx, k_idx + 1) - 2 * x72_k) == 0
          and sp.simplify(x72_k - (2**k_idx - (-1)**k_idx) / 3) == 0)

    # 4. Ex 7-5 (sun_guide P.175-176): Discretization of continuous state-space & pulse TF
    A75 = sp.Matrix([[0, 1], [0, -1]])
    B75 = sp.Matrix([[0], [1]])
    C75 = sp.Matrix([[1, 0]])
    tau_s = sp.Symbol("tau", positive=True)
    Phi75_t = sp.simplify((A75 * tau_s).exp())
    G75 = Phi75_t.subs(tau_s, T_s)
    H75 = sp.Matrix([sp.integrate((Phi75_t * B75)[i], (tau_s, 0, T_s)) for i in range(2)])
    Gz75 = sp.simplify((C75 * (z * sp.eye(2) - G75).inv() * H75)[0, 0])
    Gz75_exp = ((T_s - 1 + sp.exp(-T_s)) * z + 1 - sp.exp(-T_s) - T_s * sp.exp(-T_s)) / ((z - 1) * (z - sp.exp(-T_s)))
    check("ch07_ex_guide75_discretize",
          G75 == sp.Matrix([[1, 1 - sp.exp(-T_s)], [0, sp.exp(-T_s)]])
          and sp.simplify(H75 - sp.Matrix([[T_s + sp.exp(-T_s) - 1], [1 - sp.exp(-T_s)]])) == sp.zeros(2, 1)
          and sp.simplify(Gz75 - Gz75_exp) == 0)

    # 5. Ex 7-6 (sun_guide P.176-177): Delay + ZOH stability via W-transform
    D76_z = z**2 - b_s * z + 2 - 2 * b_s
    D76_w = sp.expand(sp.together(D76_z.subs(z, (1 + w) / (1 - w))) * (1 - w)**2)
    poly76 = sp.Poly(D76_w, w)
    c76 = poly76.all_coeffs()  # [3-b, 4b-2, 3-3b]
    check("ch07_ex_guide76_delay_stab",
          sp.simplify(c76[0] - (3 - b_s)) == 0
          and sp.simplify(c76[1] - (4 * b_s - 2)) == 0
          and sp.simplify(c76[2] - (3 - 3 * b_s)) == 0
          and abs(math.log(2) - 0.693147) < 1e-5,
          f"0 < T < ln(2) = {math.log(2):.4f}")

    # 6. Ex 7-7 (sun_guide P.177-178): ZOH + K/(s(0.2s+1)), T=1
    e_m5 = sp.exp(-5)
    Gz77 = sp.simplify(K * (1 - 1 / z) * (z / (z - 1)**2 - sp.Rational(1, 5) * z / (z - 1) + sp.Rational(1, 5) * z / (z - e_m5)))
    D77_z = sp.expand((z - 1) * (z - e_m5) + sp.simplify(Gz77 * (z - 1) * (z - e_m5)))
    D77_w = sp.expand(sp.together(D77_z.subs(z, (w + 1) / (w - 1))) * (w - 1)**2)
    c77 = sp.Poly(D77_w, w).all_coeffs()  # quadratic in w: all 3 coeffs > 0
    K77_max_w1 = float(sp.solve(c77[1], K)[0])
    K77_max_w0 = float(sp.solve(c77[2], K)[0])
    K77_max = min(K77_max_w1, K77_max_w0)
    check("ch07_ex_guide77_zoh_routh",
          abs(K77_max - 3.304) < 0.01,
          f"0 < K < {K77_max:.4f} (book 3.3)")

    # 7. Ex 7-8 (sun_guide P.178-179) — CRITICAL TEXTBOOK ERRATA (missing 1/2 in partial fraction!)
    # True: K/(s(s+2)) = (K/2)*(1/s - 1/(s+2)) -> ZOH at T=0.5 gives (K/2)*(1-e^-1)/(z-e^-1)
    Dz78 = 2 * (z - sp.Rational(1, 2)) / (z - 1)
    Gz78_true = sp.simplify(Dz78 * (1 - 1 / z) * (K / 2) * (z / (z - 1) - z / (z - sp.exp(-1))))
    # Stability limit at z=-1: 1 + Gz78_true(-1) = 0
    K78_true = float(sp.solve(1 + Gz78_true.subs(z, -1), K)[0])
    # Book's erroneous version (dropped 1/2):
    K78_book = float(sp.solve(1 + 2 * Gz78_true.subs(z, -1), K)[0])
    check("ch07_ex_guide78_errata",
          abs(K78_true - 2.8856) < 0.005 and abs(K78_book - 1.4428) < 0.005
          and abs(K78_true / K78_book - 2.0) < 1e-9,
          f"true K_max={K78_true:.4f} vs book_err={K78_book:.4f} (2x missing 1/2)")

    # 8. Ex 7-9 (sun_guide P.179-180): Discrete steady-state error coefficients Kp, Kv, Ka
    T79 = sp.Rational(1, 10)
    Gz79 = sp.simplify((1 - 1 / z) * (T79 * z / (z - 1)**2 - z / (z - 1) + z / (z - sp.exp(-T79))))
    Kv79 = sp.simplify((1 / T79) * sp.limit((z - 1) / z * Gz79, z, 1))
    Ka79 = sp.simplify((1 / T79**2) * sp.limit(((z - 1) / z)**2 * Gz79, z, 1))
    # Check closed-loop poles inside unit circle
    D79_poly = sp.Poly(sp.expand(sp.numer(sp.together(1 + Gz79))), z)
    roots79 = [complex(r.evalf()) for r in sp.nroots(D79_poly)]
    check("ch07_ex_guide79_ess",
          Kv79 == 1 and Ka79 == 0 and all(abs(r) < 1 for r in roots79),
          f"Kv={Kv79}, |z1,2|={[round(abs(r), 4) for r in roots79]}")

    # 9. Ex 7-10 (sun_guide P.180-181): Stability vs min e_ss & critical damping
    D710_z = z**2 - z + sp.Rational(1, 5) * K
    D710_w = sp.expand(sp.together(D710_z.subs(z, (w + 1) / (w - 1))) * (w - 1)**2)
    c710 = sp.Poly(D710_w, w).all_coeffs()
    K710_max = sp.solve(c710[1], K)[0]
    K710_crit = sp.solve(1 - sp.Rational(4, 5) * K, K)[0]
    check("ch07_ex_guide710_crit_damp",
          K710_max == 5 and K710_crit == sp.Rational(5, 4))

    # 10. Ex 7-11 (sun_guide P.181-182): Deadbeat (minimum-time) digital controller D(z)
    # G_h G(z) = (1-z^-1) Z{2/(s(s+1)(s+2))} at T=1 = (1-e^-1)^2 (z+e^-1) / ((z-e^-1)(z-e^-2))
    GhG711 = sp.simplify((1 - 1 / z) * (z / (z - 1) - 2 * z / (z - sp.exp(-1)) + z / (z - sp.exp(-2))))
    GhG711_exact = (1 - sp.exp(-1))**2 * (z + sp.exp(-1)) / ((z - sp.exp(-1)) * (z - sp.exp(-2)))
    lead711 = float(((1 - sp.exp(-1))**2).evalf())
    zero711 = float((-sp.exp(-1)).evalf())
    # Deadbeat for step: Phi_e = 1 - z^-1, Phi = z^-1 -> D(z) = (1/z) / (GhG711 * (1 - 1/z))
    Dz711 = sp.simplify((1 / z) / (GhG711 * (1 - 1 / z)))
    u_inv = sp.Symbol("u_inv")
    e1_series = sp.series(Dz711.subs(z, 1 / u_inv), u_inv, 0, 4).removeO()
    e1_coeffs = [float(e1_series.coeff(u_inv, i).evalf()) for i in range(4)]
    check("ch07_ex_guide711_deadbeat",
          sp.simplify(GhG711 - GhG711_exact) == 0
          and abs(lead711 - 0.3996) < 0.001 and abs(zero711 - (-0.3679)) < 0.001
          and abs(e1_coeffs[0] - 2.5027) < 0.01 and abs(e1_coeffs[1] - 0.323) < 0.01
          and abs(e1_coeffs[2] - 1.249) < 0.01 and abs(e1_coeffs[3] - 0.908) < 0.01,
          f"lead={lead711:.4f}, zero={zero711:.4f}, e1={[round(v, 3) for v in e1_coeffs]}")

    # 11. sun_main P.305-306 Ex 7-27: W-transform Routh bound K < 2(1+e^-T/Td)/(1-e^-T/Td)
    r_ratio = sp.Symbol("r_ratio", positive=True)  # T/Td
    bb = sp.exp(-r_ratio)
    D727_z = z**2 + (K * (1 - bb) - (1 + bb)) * z + bb
    D727_w = sp.expand(sp.together(D727_z.subs(z, (w + 1) / (w - 1))) * (w - 1)**2)
    c727 = sp.Poly(D727_w, w).all_coeffs()
    K727_lim = sp.solve(c727[2], K)[0]
    K727_at1 = float(K727_lim.subs(r_ratio, 1).evalf())
    check("ch07_ex_main727_w_routh",
          sp.simplify(K727_lim.rewrite(sp.exp) - 2 * (sp.exp(r_ratio) + 1) / (sp.exp(r_ratio) - 1)) == 0
          and abs(K727_at1 - 4.3279) < 0.005,
          f"K_max(T=Td)={K727_at1:.4f} (book 4.33)")

    # ==================== Ch08: State-Space Analysis & Synthesis ====================
    def ctrl_mat(A_m, B_m):
        n_m = A_m.rows
        cols = [A_m**i * B_m for i in range(n_m)]
        return sp.Matrix.hstack(*cols)

    def obs_mat(A_m, C_m):
        n_m = A_m.rows
        rows = [C_m * A_m**i for i in range(n_m)]
        return sp.Matrix.vstack(*rows)

    # 1. Ex 8-1 & Ex 8-3 (sun_guide P.198, P.200)
    A81 = sp.Matrix([[1, 3, 2], [0, 2, 0], [0, 1, 3]])
    B81 = sp.Matrix([[2, 1], [1, 1], [-1, -1]])
    Qc81 = ctrl_mat(A81, B81)
    a83, b83, c83, d83 = sp.symbols("a83 b83 c83 d83")
    A83 = sp.Matrix([[a83, b83], [c83, d83]])
    B83 = sp.Matrix([[1], [1]])
    C83 = sp.Matrix([[1, 0]])
    check("ch08_ex_guide81_83_ctrl_obs",
          Qc81.rank() == 2 and Qc81 * Qc81.T == sp.Matrix([[59, 49, -49], [49, 42, -42], [-49, -42, 42]])
          and sp.simplify(ctrl_mat(A83, B83).det() - (c83 + d83 - a83 - b83)) == 0
          and obs_mat(A83, C83).det() == b83)

    # 2. Ex 8-2 (9) PBH rank & Ex 8-4 (sun_guide P.199-201)
    A82_9 = sp.Matrix([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 5, 0]])
    B82_9 = sp.Matrix([[0, 1], [1, 0], [0, 1], [-2, 0]])
    pbh_ok = all(sp.Matrix.hstack(lam * sp.eye(4) - A82_9, B82_9).rank() == 4 for lam in [0, sp.sqrt(5), -sp.sqrt(5)])
    A84_6 = sp.Matrix([[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]])
    C84_6 = sp.Matrix([[1, 1, 0, 0], [0, 1, 0, 1]])
    A84_7 = sp.Matrix([[-1, 1, 0], [0, -1, 0], [0, 0, 2]])
    C84_7 = sp.Matrix([[0, 1, 0], [0, 0, 1]])
    check("ch08_ex_guide82_84_jordan_pbh",
          pbh_ok and ctrl_mat(A82_9, B82_9).rank() == 4
          and obs_mat(A84_6, C84_6).rank() == 3  # unobservable (2nd block 1st col in C is [0;0])
          and obs_mat(A84_7, C84_7).rank() == 2)  # unobservable (1st block 1st col in C is [0;0])

    # 3. Ex 8-5 (sun_guide P.201): Series connection loses controllability
    A85 = sp.Matrix([[0, 1, 0], [-3, -4, 0], [2, 1, -2]])
    B85 = sp.Matrix([[0], [1], [0]])
    C85 = sp.Matrix([[0, 0, 1]])
    check("ch08_ex_guide85_series",
          ctrl_mat(A85, B85).rank() == 2 and obs_mat(A85, C85).rank() == 3)

    # 4. Ex 8-6 (sun_guide P.201-202): Controllable & observable canonical transforms
    A86 = sp.Matrix([[0, 2, -2], [1, 1, -2], [2, -2, 1]])
    B86 = sp.Matrix([[2], [1], [1]])
    C86 = sp.Matrix([[1, 1, 1]])
    L86 = sp.Matrix([[-1, -2, 1], [-2, 1, 0], [1, 0, 0]])
    Tc86 = ctrl_mat(A86, B86) * L86
    To86_inv = L86 * obs_mat(A86, C86)
    To86 = To86_inv.inv()
    check("ch08_ex_guide86_canonical",
          Tc86.inv() * A86 * Tc86 == sp.Matrix([[0, 1, 0], [0, 0, 1], [-2, 1, 2]])
          and Tc86.inv() * B86 == sp.Matrix([[0], [0], [1]])
          and C86 * Tc86 == sp.Matrix([[-20, -4, 4]])
          and To86_inv * A86 * To86 == sp.Matrix([[0, 0, -2], [1, 0, 1], [0, 1, 2]])
          and To86_inv * B86 == sp.Matrix([[-20], [-4], [4]])
          and C86 * To86 == sp.Matrix([[0, 0, 1]]))

    # 5. Ex 8-7 (sun_guide P.202-204): Controllability, observability & Kalman decomposition
    A87 = sp.Matrix([[0, 0, -1], [1, 0, -3], [0, 1, -3]])
    B87 = sp.Matrix([[1], [1], [0]])
    C87 = sp.Matrix([[0, 1, -2]])
    Tc87 = sp.Matrix([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
    To87_inv = sp.Matrix([[0, 1, -2], [1, -2, 3], [0, 0, 1]])
    T1_87 = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    Tkal87 = Tc87 * T1_87
    check("ch08_ex_guide87_kalman",
          Tc87.inv() * A87 * Tc87 == sp.Matrix([[0, -1, -1], [1, -2, -2], [0, 0, -1]])
          and To87_inv * A87 * To87_inv.inv() == sp.Matrix([[0, 1, 0], [-1, -2, 0], [1, 0, -1]])
          and Tkal87.inv() * A87 * Tkal87 == sp.Matrix([[-1, 0, 1], [1, -1, -2], [0, 0, -1]])
          and Tkal87.inv() * B87 == sp.Matrix([[1], [0], [0]])
          and C87 * Tkal87 == sp.Matrix([[1, 0, -2]]))

    # 6. Ex 8-8 (sun_guide P.204-205): Zero-pole cancellation hiding unstable pole s=1
    A88 = sp.Matrix([[-4, 5], [1, 0]])
    B88 = sp.Matrix([[-5], [1]])
    C88 = sp.Matrix([[1, -1]])
    D88 = sp.Matrix([[1]])
    G88 = sp.simplify((C88 * (s_c * sp.eye(2) - A88).inv() * B88 + D88)[0, 0])
    check("ch08_ex_guide88_cancel",
          ctrl_mat(A88, B88).det() == 0 and obs_mat(A88, C88).det() == 0
          and sp.simplify(G88 - (s_c - 1) / (s_c + 5)) == 0
          and set(A88.eigenvals().keys()) == {1, -5})

    # 7. Ex 8-9 (sun_guide P.205): 8th-order Jordan Kalman decomposition -> G(s) = G_co(s)
    A89_co = sp.Matrix([[-3, 1, 0], [0, -3, 0], [0, 0, -5]])
    B89_co = sp.Matrix([[1, 3], [5, 7], [9, 2]])
    C89_co = sp.Matrix([[3, 4, 3], [1, 1, 7]])
    G89_co = sp.simplify(C89_co * (s_c * sp.eye(3) - A89_co).inv() * B89_co)
    G89_exp = sp.Matrix([
        [50 * s_c**2 + 361 * s_c + 663, 43 * s_c**2 + 353 * s_c + 714],
        [69 * s_c**2 + 431 * s_c + 682, 24 * s_c**2 + 171 * s_c + 311],
    ]) / ((s_c + 3)**2 * (s_c + 5))
    check("ch08_ex_guide89_8th_kalman",
          sp.simplify(G89_co - G89_exp) == sp.zeros(2, 2))

    # 8. Ex 8-10 & Ex 8-11 (sun_guide P.206-207): State feedback pole placement
    A810 = sp.Matrix([[0, 0, 0], [1, -6, 0], [0, 1, -12]])
    B810 = sp.Matrix([[1], [0], [0]])
    K810 = sp.Matrix([[-14, 186, -1220]])
    cl810 = sp.expand((s_c * sp.eye(3) - (A810 - B810 * K810)).det())
    L810 = sp.Matrix([[72, 18, 1], [18, 1, 0], [1, 0, 0]])
    Tc810 = ctrl_mat(A810, B810) * L810
    K810_canon = sp.Matrix([[4 - 0, 6 - 72, 4 - 18]]) * Tc810.inv()
    check("ch08_ex_guide810_811_pole",
          sp.simplify(cl810 - (s_c + 2) * (s_c**2 + 2 * s_c + 2)) == 0
          and K810_canon == K810)

    # 9. Ex 8-12 (sun_guide P.207-208): Time-domain specs -> dominant poles -> K
    z812 = math.sqrt(math.log(0.05)**2 / (math.pi**2 + math.log(0.05)**2))
    w0_812 = math.pi / (0.5 * math.sqrt(1 - z812**2))
    wb_812 = w0_812 * math.sqrt(1 - 2 * z812**2 + math.sqrt(2 - 4 * z812**2 + 4 * z812**4))
    a2_812 = 2 * z812 * w0_812 + 100
    a1_812 = w0_812**2 + 200 * z812 * w0_812
    a0_812 = 100 * w0_812**2
    check("ch08_ex_guide812_specs",
          abs(z812 - 0.6901) < 0.001 and abs(w0_812 - 8.6819) < 0.002 and wb_812 <= 10.0
          and abs((a0_812 - 0) - 7538) < 2 and abs((a1_812 - 72) - 1202) < 2 and abs((a2_812 - 18) - 94) < 1,
          f"zeta={z812:.4f}, w0={w0_812:.4f}, wb={wb_812:.3f}, K={[round(a0_812), round(a1_812-72), round(a2_812-18)]}")

    # 10. Ex 8-13 (sun_guide P.208-209): Stabilizability (uncontrollable stable mode s=-1)
    k3_s = sp.Symbol("k3")
    A813_3 = sp.Matrix([[0, 1, 0], [0, 0, 1], [6, 7, 0]])
    B813_3 = sp.Matrix([[0], [1], [1]])
    K813_3 = sp.Matrix([[12 - 6 * k3_s, 6 - k3_s, k3_s]])
    cl813_3 = sp.expand((s_c * sp.eye(3) - (A813_3 - B813_3 * K813_3)).det())
    check("ch08_ex_guide813_stabilize",
          ctrl_mat(A813_3, B813_3).rank() == 2
          and sp.simplify(cl813_3 - (s_c + 1) * (s_c + 2) * (s_c + 3)) == 0)

    # 11. Ex 8-14 (sun_guide P.210): Infinite-horizon LQR ARE for double integrator
    a814 = sp.Symbol("a814", positive=True)
    b814 = sp.Symbol("b814", real=True)
    A814 = sp.Matrix([[0, 1], [0, 0]])
    B814 = sp.Matrix([[0], [1]])
    Q814 = sp.Matrix([[1, b814], [b814, a814]])
    P814 = sp.Matrix([[sp.sqrt(a814 + 2) - b814, 1], [1, sp.sqrt(a814 + 2)]])
    are814 = sp.simplify(-P814 * A814 - A814.T * P814 + P814 * B814 * B814.T * P814 - Q814)
    check("ch08_ex_guide814_lqr", are814 == sp.zeros(2, 2))

    # 12. Ex 8-15 (sun_guide P.210-211): Full-order observer design
    A815 = sp.Matrix([[1, 3], [0, -1]])
    C815 = sp.Matrix([[1, 1]])
    H815 = sp.Matrix([[3], [1]])
    obs815 = sp.expand((s_c * sp.eye(2) - (A815 - H815 * C815)).det())
    L815 = sp.Matrix([[0, 1], [1, 0]])
    To815 = (L815 * obs_mat(A815, C815)).inv()
    H815_canon = To815 * sp.Matrix([[4 - (-1)], [4 - 0]])
    check("ch08_ex_guide815_observer",
          sp.simplify(obs815 - (s_c + 2)**2) == 0 and H815_canon == H815)

    # 13. Ex 8-16 (sun_guide P.212-213): Separation principle (observer + state feedback)
    A816 = sp.Matrix([[0, 0], [1, -6]])
    B816 = sp.Matrix([[1], [0]])
    C816 = sp.Matrix([[0, 1]])
    K816 = sp.Matrix([[14, 16]])
    H816 = sp.Matrix([[100], [14]])
    comp816 = sp.BlockMatrix([[A816 - B816 * K816, B816 * K816], [sp.zeros(2, 2), A816 - H816 * C816]]).as_explicit()
    cl816 = sp.expand((s_c * sp.eye(4) - comp816).det())
    check("ch08_ex_guide816_separation",
          sp.simplify(cl816 - (s_c + 10)**4) == 0)

    # 14. Ex 8-17 & Ex 8-18 (sun_guide P.213-215): Discretization controllability & 3-step deadbeat
    om = sp.Symbol("omega", positive=True)
    A817 = sp.Matrix([[0, 1], [-om**2, 0]])
    B817 = sp.Matrix([[0], [1]])
    C817 = sp.Matrix([[1, 0]])
    G817 = sp.Matrix([[sp.cos(om * T_s), sp.sin(om * T_s) / om], [-om * sp.sin(om * T_s), sp.cos(om * T_s)]])
    H817 = sp.Matrix([[(1 - sp.cos(om * T_s)) / om**2], [sp.sin(om * T_s) / om]])
    det_Qc817 = sp.simplify(sp.trigsimp(ctrl_mat(G817, H817).det()))
    det_Qo817 = sp.simplify(obs_mat(G817, C817).det())
    # Ex 8-18
    G818 = sp.Matrix([[-1, 1, 1], [1, 0, -2], [1, 0, 1]])
    H818 = sp.Matrix([[0], [0], [1]])
    x0_818 = sp.Matrix([[1], [1], [1]])
    u818 = sp.Matrix.hstack(G818**2 * H818, G818 * H818, H818).inv() * (-G818**3 * x0_818)
    x3_818 = G818**3 * x0_818 + G818**2 * H818 * u818[0] + G818 * H818 * u818[1] + H818 * u818[2]
    check("ch08_ex_guide817_818_disc_ss",
          sp.simplify(sp.trigsimp(det_Qc817 - 2 * sp.sin(om * T_s) * (sp.cos(om * T_s) - 1) / om**3)) == 0
          and sp.simplify(det_Qo817 - sp.sin(om * T_s) / om) == 0
          and u818 == sp.Matrix([[sp.Rational(-6, 5)], [sp.Rational(-12, 5)], [sp.Rational(9, 5)]])
          and x3_818 == sp.zeros(3, 1))

    # ==================== Ch09: Nonlinear Systems (Describing Function & Lyapunov) ====================
    th = sp.Symbol("theta", real=True)
    A_amp, M_rel, h_rel, k_nl = sp.symbols("A M h k_nl", positive=True)

    # 1. Describing functions via exact Fourier integrals (sun_main P.397-402 & sun_guide P.223-224)
    # Ideal relay: B1 = (4/pi) int_0^(pi/2) M sin(th) dth = 4M/pi
    N_ideal = sp.simplify((4 / (sp.pi * A_amp)) * sp.integrate(M_rel * sp.sin(th), (th, 0, sp.pi / 2)))
    # Deadzone relay (psi1 = asin(h/A)): B1/A = (4/(pi*A)) int_psi1^(pi/2) M sin(th) dth
    psi1 = sp.Symbol("psi1", positive=True)  # sin(psi1) = h/A, cos(psi1) = sqrt(1-(h/A)^2)
    N_dz_relay = sp.simplify((4 / (sp.pi * A_amp)) * sp.integrate(M_rel * sp.sin(th), (th, psi1, sp.pi / 2)))
    # Hysteresis relay (psi1 to pi+psi1): A1/A = (2/(pi*A)) int_psi1^(pi+psi1) M cos(th) dth = -4Mh/(pi*A^2)
    Re_hys = sp.simplify((2 / (sp.pi * A_amp)) * sp.integrate(M_rel * sp.sin(th), (th, psi1, sp.pi + psi1)))
    Im_hys = sp.simplify((2 / (sp.pi * A_amp)) * sp.integrate(M_rel * sp.cos(th), (th, psi1, sp.pi + psi1)))
    # Check -1/N(A) for hysteresis relay has constant Im = -pi*h/(4M)!
    # Write N(A) = (4M/(pi*A))*(cos(psi1) - I*sin(psi1)) with sin(psi1) = h/A
    N_hys_polar = (4 * M_rel / (sp.pi * A_amp)) * (sp.cos(psi1) - sp.I * sp.sin(psi1))
    neg_inv_hys = sp.simplify(sp.trigsimp(-1 / N_hys_polar))
    neg_inv_hys_im = sp.simplify(sp.im(neg_inv_hys).subs(sp.sin(psi1), h_rel / A_amp))
    # Cubic y = b3 * x^3 -> N(A) = (3/4)*b3*A^2
    b3_s = sp.Symbol("b3", real=True)
    N_cubic = sp.simplify((1 / (sp.pi * A_amp)) * sp.integrate(b3_s * (A_amp * sp.sin(th))**3 * sp.sin(th), (th, 0, 2 * sp.pi)))
    check("ch09_f_df_core_integrals",
          N_ideal == 4 * M_rel / (sp.pi * A_amp)
          and sp.simplify(N_dz_relay - 4 * M_rel * sp.cos(psi1) / (sp.pi * A_amp)) == 0
          and sp.simplify(Re_hys - 4 * M_rel * sp.cos(psi1) / (sp.pi * A_amp)) == 0
          and sp.simplify(Im_hys - (-4 * M_rel * sp.sin(psi1) / (sp.pi * A_amp))) == 0
          and sp.simplify(neg_inv_hys_im - (-sp.pi * h_rel / (4 * M_rel))) == 0
          and N_cubic == sp.Rational(3, 4) * b3_s * A_amp**2)

    # 2. Saturation & Deadzone & Backlash describing functions (numerical integral vs formula)
    # Saturation k=2, a=1, A=3:
    a_sat = 1.0
    A_t = 3.0
    k_t = 2.0
    p1 = math.asin(a_sat / A_t)
    N_sat_f = (2 * k_t / math.pi) * (p1 + (a_sat / A_t) * math.sqrt(1 - (a_sat / A_t)**2))
    N_sat_num = float((4 / (sp.pi * A_t)) * (sp.integrate(k_t * A_t * sp.sin(th)**2, (th, 0, p1)) + sp.integrate(k_t * a_sat * sp.sin(th), (th, p1, sp.pi / 2))))
    # Deadzone k=2, Delta=1, A=3:
    N_dz_f = (2 * k_t / math.pi) * (math.pi / 2 - p1 - (a_sat / A_t) * math.sqrt(1 - (a_sat / A_t)**2))
    N_dz_num = float((4 / (sp.pi * A_t)) * sp.integrate(k_t * (A_t * sp.sin(th) - a_sat) * sp.sin(th), (th, p1, sp.pi / 2)))
    # Backlash k=2, b=1, A=3: y(th) on [-pi/2, pi/2]: k(A sin(th) - b) for th in [th1, pi/2], -k(A-b) for [-pi/2, th1] where sin(th1) = 2b/A - 1
    th1_bl = math.asin(2 * a_sat / A_t - 1)
    Re_bl_num = float((2 / (sp.pi * A_t)) * (sp.integrate(-k_t * (A_t - a_sat) * sp.sin(th), (th, -sp.pi / 2, th1_bl)) + sp.integrate(k_t * (A_t * sp.sin(th) - a_sat) * sp.sin(th), (th, th1_bl, sp.pi / 2))))
    Im_bl_num = float((2 / (sp.pi * A_t)) * (sp.integrate(-k_t * (A_t - a_sat) * sp.cos(th), (th, -sp.pi / 2, th1_bl)) + sp.integrate(k_t * (A_t * sp.sin(th) - a_sat) * sp.cos(th), (th, th1_bl, sp.pi / 2))))
    Re_bl_f = (k_t / math.pi) * (math.pi / 2 + math.asin(1 - 2 * a_sat / A_t) + 2 * (1 - 2 * a_sat / A_t) * math.sqrt((a_sat / A_t) * (1 - a_sat / A_t)))
    Im_bl_f = (4 * k_t * a_sat / (math.pi * A_t)) * (a_sat / A_t - 1)
    check("ch09_f_df_sat_dz_backlash",
          abs(N_sat_f - N_sat_num) < 1e-9 and abs(N_dz_f - N_dz_num) < 1e-9
          and abs(Re_bl_f - Re_bl_num) < 1e-9 and abs(Im_bl_f - Im_bl_num) < 1e-9,
          f"sat={N_sat_f:.4f}, dz={N_dz_f:.4f}, bl={Re_bl_f:.4f}{Im_bl_f:+.4f}j")

    # 3. Ex 9-6 (sun_guide P.230) — CRITICAL ERRATA: y = x*xdot + 4x vs y = |x|*xdot + 4x
    A1_literal = sp.integrate(A_amp**2 * om * sp.sin(th) * sp.cos(th)**2, (th, 0, 2 * sp.pi)) / sp.pi
    B1_literal = sp.integrate(A_amp**2 * om * sp.sin(th)**2 * sp.cos(th), (th, 0, 2 * sp.pi)) / sp.pi
    A1_absx = (2 / sp.pi) * sp.integrate(A_amp**2 * om * sp.sin(th) * sp.cos(th)**2, (th, 0, sp.pi))
    check("ch09_ex_guide96_df_errata",
          A1_literal == 0 and B1_literal == 0
          and sp.simplify(A1_absx - 4 * A_amp**2 * om / (3 * sp.pi)) == 0,
          "literal x*xdot -> A1=B1=0 (2nd harmonic); |x|*xdot -> 4A^2*w/(3*pi)")

    # 4. Ex 9-8 (sun_guide P.231-232): Deadzone relay + G(s)=2/(s(0.5s+1)(s+1))
    G98 = 2 / (sp.I * w_s * (sp.Rational(1, 2) * sp.I * w_s + 1) * (sp.I * w_s + 1))
    G98_cross = sp.simplify(G98.subs(w_s, sp.sqrt(2)))
    # Solve -pi*A/(12*sqrt(1-1/A^2)) = -2/3 -> (pi*A/8)^2 = 1 - 1/A^2
    A98_sols = sorted([float(r.evalf()) for r in sp.solve(sp.pi * A_amp / (12 * sp.sqrt(1 - 1 / A_amp**2)) - sp.Rational(2, 3), A_amp)])
    check("ch09_ex_guide98_relay_dz",
          G98_cross == sp.Rational(-2, 3)
          and abs(A98_sols[0] - 1.112) < 0.005 and abs(A98_sols[1] - 2.290) < 0.005,
          f"A1={A98_sols[0]:.3f}, A2={A98_sols[1]:.3f}")

    # 5. Ex 9-9 (sun_guide P.232-233): x'''+3x''+2x'+x+4x^3=0
    G99 = 1 / ((sp.I * w_s)**3 + 3 * (sp.I * w_s)**2 + 2 * sp.I * w_s + 1)
    G99_cross = sp.simplify(G99.subs(w_s, sp.sqrt(2)))
    A99 = float(sp.sqrt(sp.Rational(5, 3)).evalf())
    check("ch09_ex_guide99_cubic",
          G99_cross == sp.Rational(-1, 5) and abs(A99 - 1.2910) < 0.002,
          f"G(j*sqrt2)={G99_cross}, A={A99:.4f}")

    # 6. sun_main P.406-407 Ex 9-5: Feedback NL block reduction + limit cycle
    G1_95 = 1 / (s_c * (s_c + 1))
    G2_95 = 2 / s_c
    Geq_95_1 = sp.simplify(G1_95 * G2_95 / (1 + G1_95))
    Geq_95_2 = sp.simplify(G1_95 * G2_95 * s_c / (1 + G1_95))
    # Part 1: 4/(pi*A)*sqrt(1-1/A^2) = 1/2 at w=1
    A95_sols = sorted([float(r.evalf()) for r in sp.solve(4 / (sp.pi * A_amp) * sp.sqrt(1 - 1 / A_amp**2) - sp.Rational(1, 2), A_amp)])
    check("ch09_ex_main95_feedback_nl",
          sp.simplify(Geq_95_1 - 2 / (s_c * (s_c**2 + s_c + 1))) == 0
          and sp.simplify(Geq_95_2 - 2 / (s_c**2 + s_c + 1)) == 0
          and abs(A95_sols[1] - 2.290) < 0.005,
          f"A_stable={A95_sols[1]:.4f}")

    # 7. Ex 9-11 & Ex 9-12 (sun_guide P.234-235): Nonlinear Lyapunov direct method
    x1, x2 = sp.symbols("x1 x2", real=True)
    a911 = sp.Symbol("a911", real=True)
    Vdot_911 = sp.expand(2 * x1 * (-x2 + a911 * x1**3) + 2 * x2 * (x1 + a911 * x2**3))
    Vdot_912 = sp.expand(x1**3 * x2 + x2 * (-x2 - x1**3))
    check("ch09_ex_guide911_912_lyap_nl",
          sp.simplify(Vdot_911 - 2 * a911 * (x1**4 + x2**4)) == 0 and Vdot_912 == -x2**2)

    # 8. Ex 9-13 & Ex 9-14 (sun_guide P.235-236): Continuous LTI Lyapunov equation A^T P + PA = -Q
    p11, p12, p22, p13, p23, p33 = sp.symbols("p11 p12 p22 p13 p23 p33")
    A913 = sp.Matrix([[0, 1], [-3, -9]])
    P2 = sp.Matrix([[p11, p12], [p12, p22]])
    sol913 = sp.solve(A913.T * P2 + P2 * A913 + sp.eye(2), [p11, p12, p22])
    P913 = P2.subs(sol913)
    A914 = sp.Matrix([[0, -K], [1, -3]])
    sol914 = sp.solve(A914.T * P2 + P2 * A914 + sp.eye(2), [p11, p12, p22])
    P914 = sp.simplify(P2.subs(sol914))
    det914 = sp.factor(P914.det())
    check("ch09_ex_guide913_914_lyap_lti",
          P913 == sp.Matrix([[sp.Rational(31, 18), sp.Rational(1, 6)], [sp.Rational(1, 6), sp.Rational(2, 27)]])
          and P913[0, 0] > 0 and P913.det() > 0
          and P914 == sp.Matrix([[(K + 10) / (6 * K), sp.Rational(-1, 2)], [sp.Rational(-1, 2), (K + 1) / 6]])
          and sp.simplify(det914 - (K**2 + 2 * K + 10) / (36 * K)) == 0)

    # 9. Ex 9-15 (sun_guide P.236) — ERRATA CHECK on Sylvester minors Delta_2, Delta_3!
    k915 = sp.Symbol("k915", positive=True)
    A915 = sp.Matrix([[0, 1, 0], [0, -2, 1], [-sp.Rational(1, 2) * k915, 0, -1]])
    Q915 = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    P3 = sp.Matrix([[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]])
    sol915 = sp.solve(A915.T * P3 + P3 * A915 + Q915, [p11, p12, p13, p22, p23, p33])
    P915 = sp.simplify(P3.subs(sol915))
    d1_915 = sp.factor(P915[0, 0])
    d2_915 = sp.factor(P915[:2, :2].det())
    d3_915 = sp.factor(P915.det())
    check("ch09_ex_guide915_lyap_errata",
          sp.simplify(P915 - sp.Matrix([[k915**2 + 24 * k915, 12 * k915, 0], [12 * k915, 6 * k915, 2 * k915], [0, 2 * k915, 24]]) / (48 - 4 * k915)) == sp.zeros(3, 3)
          and sp.simplify(d2_915 - 6 * k915**3 / (48 - 4 * k915)**2) == 0
          and sp.simplify(d3_915 - k915**3 / (48 - 4 * k915)**2) == 0,
          f"true Delta2=6k^3/(48-4k)^2, Delta3=k^3/(48-4k)^2 -> 0<k<12")

    # 10. Ex 9-16 (sun_guide P.237) & sun_main P.421 Ex 9-15 (Discrete Lyapunov A^T P A - P = -Q)
    Az915 = sp.Matrix([[0, -sp.Rational(1, 2)], [1, -1]])
    sol_d915 = sp.solve(Az915.T * P2 * Az915 - P2 + sp.eye(2), [p11, p12, p22])
    P_d915 = P2.subs(sol_d915)
    check("ch09_ex_disc_lyap",
          P_d915 == sp.Matrix([[4, -2], [-2, 3]]) and P_d915[0, 0] > 0 and P_d915.det() > 0)

    # 11. sun_main P.419-420 Ex 9-14 (1)(2)(3)
    A914m_1 = sp.Matrix([[1, -3], [1, -2]])
    A914m_2 = sp.Matrix([[0, 0], [1, -6]])
    A914m_3 = sp.Matrix([[0, 1], [1, 0]])
    P914m_1 = P2.subs(sp.solve(A914m_1.T * P2 + P2 * A914m_1 + sp.eye(2), [p11, p12, p22]))
    sol914m_2 = sp.solve(A914m_2.T * P2 + P2 * A914m_2 + sp.eye(2), [p11, p12, p22])
    sol914m_3 = sp.solve(A914m_3.T * P2 + P2 * A914m_3 + sp.eye(2), [p11, p12, p22])
    P914m_3 = P2.subs(sol914m_3)
    check("ch09_ex_main914_lyap_cases",
          P914m_1 == sp.Matrix([[3, sp.Rational(-7, 2)], [sp.Rational(-7, 2), sp.Rational(11, 2)]])
          and P914m_1.det() == sp.Rational(17, 4)
          and sol914m_2 == []
          and sp.simplify(P914m_3.det() - (-p22**2 - sp.Rational(1, 4))) == 0)

    # --- Ch01 Formulas & Exercises (sun_main P.6-18, sun_guide P.1-11 & P.240) ---
    G_s, H_s, G1_s, G2_s, F_s, f_dist = sp.symbols("G H G_1 G_2 F f_dist", nonzero=True)
    M_cl = G_s / (1 + G_s * H_s)
    S_G_M = sp.simplify(sp.diff(M_cl, G_s) * G_s / M_cl)
    S_H_M = sp.simplify(sp.diff(M_cl, H_s) * H_s / M_cl)
    G0_ol = G_s * H_s
    S_G_G0 = sp.simplify(sp.diff(G0_ol, G_s) * G_s / G0_ol)
    # Disturbance eq (1-6): y = G2*(f_dist - G1*H*y) -> y = G2*f_dist / (1 + G1*G2*H)
    y_sym = sp.Symbol("y_sym")
    y_cl_dist = sp.solve(sp.Eq(y_sym, G2_s * (f_dist - G1_s * H_s * y_sym)), y_sym)[0]
    # Multi-loop stabilization eq (1-7)
    Phi_ml = sp.simplify(M_cl / (1 + M_cl * F_s))
    check("ch01_f_sensitivity_disturbance",
          sp.simplify(S_G_M - 1 / (1 + G_s * H_s)) == 0
          and sp.simplify(S_H_M + G_s * H_s / (1 + G_s * H_s)) == 0
          and S_G_G0 == 1
          and sp.simplify(y_cl_dist - G2_s * f_dist / (1 + G1_s * G2_s * H_s)) == 0
          and sp.simplify(Phi_ml - G_s / (1 + G_s * H_s + G_s * F_s)) == 0)

    # Ex 1-10 / A1-10: verify linearity (T[a*r1+b*r2] == a*T[r1]+b*T[r2]) & time-invariance for all 7 equations
    r1 = sp.Function("r1")(t_r)
    r2 = sp.Function("r2")(t_r)
    w_sym = sp.Symbol("omega", real=True, nonzero=True)
    # Eq (1) c = 5 + r^2 + t*r'' -> nonlinear & time-varying
    T1 = lambda r: 5 + r**2 + t_r * sp.diff(r, t_r, 2)
    eq1_lin = sp.simplify(T1(a * r1 + b * r2) - (a * T1(r1) + b * T1(r2))) == 0
    # Eq (4) c = r*cos(w*t) + 5 -> nonlinear (T[0]=5 != 0) & time-varying
    T4 = lambda r: r * sp.cos(w_sym * t_r) + 5
    eq4_lin = sp.simplify(T4(a * r1 + b * r2) - (a * T4(r1) + b * T4(r2))) == 0
    # Eq (6) c = r^2 -> nonlinear & time-invariant
    T6 = lambda r: r**2
    eq6_lin = sp.simplify(T6(a * r1 + b * r2) - (a * T6(r1) + b * T6(r2))) == 0
    # Eq (5) c = 3*r + 6*r' + 5*int -> linear & time-invariant (operator part 3*r + 6*r')
    T5 = lambda r: 3 * r + 6 * sp.diff(r, t_r)
    eq5_lin = sp.simplify(T5(a * r1 + b * r2) - (a * T5(r1) + b * T5(r2))) == 0
    # Eq (7) gated step c = H(t-6)*r(t) -> linear & time-varying
    T7 = lambda r, tv: sp.Heaviside(tv - 6) * r
    eq7_lin = sp.simplify(T7(a * r1 + b * r2, t_r) - (a * T7(r1, t_r) + b * T7(r2, t_r))) == 0
    eq7_ti = (T7(sp.Integer(1), sp.Integer(7)) == T7(sp.Integer(1), sp.Integer(5)))  # shifted input differs
    check("ch01_ex_a1_10_classify_7eqs",
          (not eq1_lin) and (not eq4_lin) and (not eq6_lin) and eq5_lin and eq7_lin and (not eq7_ti))

    # ==================== Ch02 Full 5-Partition Upgrade Checks ====================
    # Main Ex 2-2 (P.21-22): Multi-loop RLC (node equations 2-18 & 2-19)
    R1m, R2m, R3m, Lm, Cm = sp.symbols("R1m R2m R3m Lm Cm", positive=True)
    vbm, vom, em = sp.symbols("vbm vom em")
    sol_m22 = sp.solve([
        (1 / R1m + Cm * s_c + 1 / R2m) * vbm - (1 / R2m) * vom - em / R1m,
        -(1 / R2m) * vbm + (1 / R2m + 1 / R3m + 1 / (Lm * s_c)) * vom - em / (Lm * s_c),
    ], [vbm, vom])
    tf_m22 = sp.simplify(sol_m22[vom] / em)
    exp_m22 = (R3m * (Lm + R1m * R2m * Cm) * s_c + R3m * (R1m + R2m)) / (
        R1m * (R2m + R3m) * Lm * Cm * s_c**2
        + (Lm * (R2m + R3m) + R1m * R2m * R3m * Cm + R1m * Lm) * s_c
        + R3m * (R1m + R2m)
    )
    check("ch02_main_ex2_2_multiloop_rlc", sp.simplify(tf_m22 - exp_m22) == 0)

    # Main Ex 2-6 (P.28-29): Armature-controlled DC motor
    Ra, La, Jm, ke, km = sp.symbols("Ra La Jm ke km", positive=True)
    ua_s, ml_s, wm_s, ia_s = sp.symbols("ua_s ml_s wm_s ia_s")
    sol_m26 = sp.solve([
        (La * s_c + Ra) * ia_s + ke * wm_s - ua_s,
        Jm * s_c * wm_s - (km * ia_s - ml_s),
    ], [wm_s, ia_s])
    Tm_m26 = Ra * Jm / (ke * km)
    Ta_m26 = La / Ra
    w_exp_m26 = ((1 / ke) * ua_s - (Tm_m26 / Jm) * (Ta_m26 * s_c + 1) * ml_s) / (Ta_m26 * Tm_m26 * s_c**2 + Tm_m26 * s_c + 1)
    check("ch02_main_ex2_6_dcmotor", sp.simplify(sol_m26[wm_s] - w_exp_m26) == 0)

    # Main Ex 2-7 (P.29-31) & Ex 2-13 (P.37-38): 2-tank non-interacting vs interacting
    A1t, A2t, A3t = sp.symbols("A1t A2t A3t", positive=True)
    T1t, T2t, T3t = A1t * R1m, A2t * R2m, A3t * R3m
    A_m27 = sp.Matrix([[-1 / T1t, 0], [1 / (R1m * A2t), -1 / T2t]])
    B_m27 = sp.Matrix([[1 / A1t, 0], [0, -1 / A2t]])
    C_m27 = sp.Matrix([[0, 1]])
    G_m27 = sp.simplify(C_m27 * (s_c * sp.eye(2) - A_m27).inv() * B_m27)
    A_m213 = sp.Matrix([[-1 / T1t, 1 / T1t], [1 / (A2t * R1m), -1 / (A2t * R1m) - 1 / T2t]])
    B_m213 = sp.Matrix([[1 / A1t], [0]])
    G_m213 = sp.simplify(sp.Matrix([[0, 1]]) * (s_c * sp.eye(2) - A_m213).inv() * B_m213)
    check("ch02_main_ex2_7_13_tanks",
          sp.simplify(G_m27[0, 0] - R2m / (T1t * T2t * s_c**2 + (T1t + T2t) * s_c + 1)) == 0
          and sp.simplify(G_m27[0, 1] - (-R2m * (T1t * s_c + 1) / (T1t * T2t * s_c**2 + (T1t + T2t) * s_c + 1))) == 0
          and sp.simplify(G_m213[0, 0] - R2m / (T1t * T2t * s_c**2 + (T1t + T2t + A1t * R2m) * s_c + 1)) == 0)

    # Main Ex 2-9 & 2-15 & P.76 transform T = diag(1, C)
    A_m29 = sp.Matrix([[0, 1 / Cm], [-1 / Lm, -R1m / Lm]])
    b_m29 = sp.Matrix([0, 1 / Lm])
    T_m29 = sp.diag(1, Cm)
    A_m215 = sp.Matrix([[0, 1], [-1 / (Lm * Cm), -R1m / Lm]])
    b_m215 = sp.Matrix([0, 1 / (Lm * Cm)])
    check("ch02_main_ex2_9_15_transform",
          sp.simplify(T_m29.inv() * A_m29 * T_m29 - A_m215) == sp.zeros(2, 2)
          and sp.simplify(T_m29.inv() * b_m29 - b_m215) == sp.zeros(2, 1))

    # Main Ex 2-10 & Ex 2-11 (P.34-36): 3rd-order & 4th-order non-independent storage RLC
    L1m, L2m, L3m, C1m, C2m = sp.symbols("L1m L2m L3m C1m C2m", positive=True)
    A_m210 = sp.Matrix([[-R1m / L1m, 0, -1 / L1m], [0, -R2m / L2m, 1 / L2m], [1 / Cm, -1 / Cm, 0]])
    b_m210 = sp.Matrix([1 / L1m, 0, 0])
    # Verify Ex 2-10 loop equation consistency
    G_m210_ss = sp.simplify((sp.Matrix([[0, 1, 0]]) * (s_c * sp.eye(3) - A_m210).inv() * b_m210)[0])
    Z_par_m210 = 1 / (Cm * s_c + 1 / (L2m * s_c + R2m))
    G_m210_circ = sp.simplify((Z_par_m210 / (R1m + L1m * s_c + Z_par_m210)) / (L2m * s_c + R2m))
    # Ex 2-11: 4th order after eliminating L3*i3 = L2*x4 + L1*x3
    dx1, dx2, dx3, dx4, i3m, um = sp.symbols("dx1 dx2 dx3 dx4 i3m um")
    sol_m211 = sp.solve([
        x1 - L1m * dx3,
        x2 - (L2m * dx4 + x1),
        L3m * i3m - (L2m * x4 + L1m * x3),
        x4 - (C1m * dx1 + x3),
        um - (i3m + C2m * dx2 + x4),
    ], [dx1, dx2, dx3, dx4, i3m])
    A_m211 = sp.Matrix([
        [0, 0, -1 / C1m, 1 / C1m],
        [0, 0, -L1m / (L3m * C2m), -(L2m + L3m) / (L3m * C2m)],
        [1 / L1m, 0, 0, 0],
        [-1 / L2m, 1 / L2m, 0, 0],
    ])
    b_m211 = sp.Matrix([0, 1 / C2m, 0, 0])
    check("ch02_main_ex2_10_11_circuits",
          sp.simplify(G_m210_ss - G_m210_circ) == 0
          and sp.simplify(sp.Matrix([sol_m211[dx1], sol_m211[dx2], sol_m211[dx3], sol_m211[dx4]])
                          - (A_m211 * sp.Matrix([x1, x2, x3, x4]) + b_m211 * um)) == sp.zeros(4, 1))

    # Main Ex 2-12 & Ex 2-18 (P.36-37, 44): 3 non-interacting tanks (+ sign errata P.36-37)
    A_m212 = sp.Matrix([[-1 / T1t, 0, 0], [1 / (A2t * R1m), -1 / T2t, 0], [0, 1 / (A3t * R2m), -1 / T3t]])
    B_m212 = sp.Matrix([[1 / A1t, 0], [0, 0], [0, -1 / A3t]])
    G_m212 = sp.simplify(sp.Matrix([[0, 0, 1]]) * (s_c * sp.eye(3) - A_m212).inv() * B_m212)
    check("ch02_main_ex2_12_18_3tanks",
          sp.simplify(G_m212[0, 0] - R3m / ((T1t * s_c + 1) * (T2t * s_c + 1) * (T3t * s_c + 1))) == 0)

    # Main Ex 2-22 (P.50-51): RTD with protective well
    Tin_m, Ta_m, Tr_m = sp.symbols("Tin_m Ta_m Tr_m")
    sol_m222 = sp.solve([
        C1m * s_c * Ta_m - (Tin_m - Ta_m) / R1m + (Ta_m - Tr_m) / R2m,
        C2m * s_c * Tr_m - (Ta_m - Tr_m) / R2m,
    ], [Ta_m, Tr_m])
    check("ch02_main_ex2_22_rtd_well",
          sp.simplify(sol_m222[Tr_m] / Tin_m - 1 / (R1m * C1m * R2m * C2m * s_c**2 + (R1m * C1m + R2m * C2m + R1m * C2m) * s_c + 1)) == 0)

    # Main Ex 2-28, 2-29, 2-35 (P.61-63, 70)
    sol_m228 = sp.solve([
        n1 - (R_s - Y_s + H1 * n3),
        n2 - (G1 * n1 - H2 * Y_s),
        n3 - G2 * n2,
        Y_s - G3 * n3,
    ], [n1, n2, n3, Y_s])
    sol_m229 = sp.solve([
        n1 - (G1 * (R_s - H1 * Y_s - n1) - Y_s),
        Y_s - G2 * n1,
    ], [n1, Y_s])
    check("ch02_main_ex2_28_29_35_blocks",
          sp.simplify(sol_m228[Y_s] / R_s - G1 * G2 * G3 / (1 + G2 * G3 * H2 - G1 * G2 * H1 + G1 * G2 * G3)) == 0
          and sp.simplify(sol_m229[Y_s] / R_s - G1 * G2 / (1 + G1 + G2 + G1 * G2 * H1)) == 0)

    # Main Ex 2-31 (P.65-66): 2x2 MIMO closed-loop TF matrix
    G_m231 = sp.Matrix([[1 / (s_c + 1), -1 / s_c], [2, 1 / (s_c + 2)]])
    M_m231 = sp.simplify((sp.eye(2) + G_m231).inv() * G_m231)
    M_m231_exp = sp.Matrix([
        [(3 * s_c**2 + 9 * s_c + 4) / (s_c * (s_c + 1) * (s_c + 2)), -1 / s_c],
        [2, (3 * s_c + 2) / (s_c * (s_c + 1))],
    ]) * (s_c * (s_c + 1) / (s_c**2 + 5 * s_c + 2))
    check("ch02_main_ex2_31_mimo_cl", sp.simplify(M_m231 - M_m231_exp) == sp.zeros(2, 2))

    # Main Ex 2-32/2-33/2-34 (P.67-69)
    a12, a23, a32, a34, a35, a42, a44, a45, a52 = sp.symbols("a12 a23 a32 a34 a35 a42 a44 a45 a52")
    x1v, x2v, x3v, x4v, x5v = sp.symbols("x1v x2v x3v x4v x5v")
    sol_m233 = sp.solve([
        x2v - (a12 * x1v + a32 * x3v + a42 * x4v + a52 * x5v),
        x3v - a23 * x2v,
        x4v - (a34 * x3v + a44 * x4v),
        x5v - (a35 * x3v + a45 * x4v),
    ], [x2v, x3v, x4v, x5v])
    M_m233_exp = (a12 * a23 * a34 * a45 + a12 * a23 * a35 * (1 - a44)) / (
        1 - (a23 * a32 + a23 * a34 * a42 + a44 + a23 * a34 * a45 * a52 + a23 * a35 * a52)
        + a23 * a32 * a44 + a44 * a23 * a35 * a52
    )
    sol_m234 = sp.solve([
        n1 - G1 * (R_s + H1 * n2),
        n2 - G2 * (n1 - H1 * n2 - H2 * n3),
        n3 - G3 * n2,
        Y_s - (n3 + G4 * R_s),
    ], [n1, n2, n3, Y_s])
    check("ch02_main_ex2_33_34_sfg",
          sp.simplify(sol_m233[x5v] / x1v - M_m233_exp) == 0
          and sp.simplify(sol_m234[Y_s] / R_s - (G4 + G1 * G2 * G3 / (1 + G2 * H1 + G2 * G3 * H2 - G1 * G2 * H1))) == 0)

    # Main Ex 2-37, 2-38, 2-40, 2-41, 2-42 (P.73-82)
    A_m237 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, -8, -9]])
    G_m237 = sp.simplify((sp.Matrix([[1, 0, 0]]) * (s_c * sp.eye(3) - A_m237).inv() * sp.Matrix([1, -5, 41]))[0])
    G_m238 = sp.simplify((sp.Matrix([[4, 4, 1]]) * (s_c * sp.eye(3) - A_m237).inv() * sp.Matrix([0, 0, 1]))[0])
    G_m240 = (3 * s_c**5 + 6 * s_c**4 - 10 * s_c**3 + 6 * s_c + 2) / (2 * s_c**5 + 4 * s_c**4 - 8 * s_c**3 + 2 * s_c + 2)
    Ac_m240 = sp.Matrix([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [-1, -1, 0, 4, -2]])
    cc_m240 = sp.Matrix([[-sp.Rational(1, 2), sp.Rational(3, 2), 0, 1, 0]])
    bc_m240 = sp.Matrix([0, 0, 0, 0, 1])
    G_m240_c = sp.simplify((cc_m240 * (s_c * sp.eye(5) - Ac_m240).inv() * bc_m240)[0] + sp.Rational(3, 2))
    G_m241 = (s_c**3 + 8 * s_c**2 + 17 * s_c + 8) / (s_c**3 + 6 * s_c**2 + 11 * s_c + 6)
    G_m241_d = sp.simplify((sp.Matrix([[-1, 2, 1]]) * (s_c * sp.eye(3) - sp.diag(-1, -2, -3)).inv() * sp.Matrix([1, 1, 1]))[0] + 1)
    To_m, Tm_m, Ko_m, Km_m, Kc_m = sp.symbols("To_m Tm_m Ko_m Km_m Kc_m", positive=True)
    A_m242 = sp.Matrix([[-1 / To_m, -Ko_m * Kc_m / To_m], [Km_m / Tm_m, -1 / Tm_m]])
    B_m242 = sp.Matrix([[Ko_m * Kc_m / To_m, Ko_m / To_m], [0, 0]])
    G_m242 = sp.simplify(sp.Matrix([[1, 0]]) * (s_c * sp.eye(2) - A_m242).inv() * B_m242)
    check("ch02_main_ex2_37_to_42_ss",
          sp.simplify(G_m237 - (s_c**2 + 4 * s_c + 4) / (s_c**3 + 9 * s_c**2 + 8 * s_c)) == 0
          and sp.simplify(G_m238 - G_m237) == 0
          and sp.simplify(G_m240_c - G_m240) == 0
          and sp.simplify(G_m241_d - G_m241) == 0
          and sp.simplify(G_m242[0, 0] - (Kc_m * Ko_m * (Tm_m * s_c + 1)) / ((To_m * s_c + 1) * (Tm_m * s_c + 1) + Kc_m * Ko_m * Km_m)) == 0)

    # Guide Ex 2-4 (P.29-30) & Guide Ex 2-15 (P.43 errata)
    K1g, K2g, K3g, Kmg, Ktg, ig = sp.symbols("K1g K2g K3g Kmg Ktg ig", positive=True)
    G_g24_ol = (1 / ig) * K1g * K2g * K3g * Kmg * (tau * s_c + 1) / (Tm_m * s_c + 1)
    Phi_g24 = sp.simplify(G_g24_ol / (1 + G_g24_ol * Ktg))
    R1g, R2g, Y1g, Y2g, mg1, mg2, mg3, mg4 = sp.symbols("R1g R2g Y1g Y2g mg1 mg2 mg3 mg4")
    sol_g215 = sp.solve([
        mg1 - (R1g - H1 * Y1g),
        mg2 - (G1 * mg1 + G3 * mg3),
        Y1g - G2 * mg2,
        mg3 - (R2g + H2 * Y2g),
        mg4 - (G5 * mg3 + G4 * mg2),
        Y2g - G6 * mg4,
    ], [mg1, mg2, mg3, mg4, Y1g, Y2g])
    Delta_g215 = 1 + G1 * G2 * H1 - G5 * G6 * H2 - G3 * G4 * G6 * H2 - G1 * G2 * H1 * G5 * G6 * H2
    check("ch02_guide_ex2_4_15",
          sp.simplify(Phi_g24 - (K1g * K2g * K3g * Kmg * (tau * s_c + 1)) / (ig * (Tm_m * s_c + 1) + K1g * K2g * K3g * Kmg * Ktg * (tau * s_c + 1))) == 0
          and sp.simplify(sol_g215[Y1g].subs(R2g, 0) / R1g - G1 * G2 * (1 - G5 * G6 * H2) / Delta_g215) == 0
          and sp.simplify(sol_g215[Y2g].subs(R2g, 0) / R1g - G1 * G4 * G6 / Delta_g215) == 0
          and sp.simplify(sol_g215[Y2g].subs(R1g, 0) / R2g - (G5 * G6 * (1 + G1 * G2 * H1) + G3 * G4 * G6) / Delta_g215) == 0)

    # Exercises A2-1 ~ A2-6 (Circuits)
    R_sum25 = R2m + R3m
    Rp25 = R1m / R_sum25
    A_a25 = sp.Matrix([[0, 1 / Cm], [-1 / Lm, -R1m / ((1 + Rp25) * Lm)]])
    b_a25 = sp.Matrix([0, 1 / ((1 + Rp25) * Lm)])
    G1_a25 = sp.simplify((sp.Matrix([[1, 0]]) * (s_c * sp.eye(2) - A_a25).inv() * b_a25)[0])
    A_a26 = sp.Matrix([[-1 / (R1m * C1m), 0, -1 / C1m], [0, 0, 1 / C2m], [1 / Lm, -1 / Lm, -R2m / Lm]])
    b_a26 = sp.Matrix([1 / (R1m * C1m), 0, 0])
    i1a, i2a = sp.symbols("i1a i2a")
    sol_a26 = sp.solve([
        (R1m + 1 / (C1m * s_c)) * i1a - (1 / (C1m * s_c)) * i2a - em,
        -(1 / (C1m * s_c)) * i1a + (R2m + Lm * s_c + 1 / (C1m * s_c) + 1 / (C2m * s_c)) * i2a,
    ], [i1a, i2a])
    check("ch02_ex_a2_1_to_6_circuits",
          sp.simplify((R2m + 1 / (C2m * s_c)) / (R1m + R2m + 1 / (C1m * s_c) + 1 / (C2m * s_c))
                      - C1m * (C2m * R2m * s_c + 1) / (C1m * C2m * (R1m + R2m) * s_c + C1m + C2m)) == 0
          and sp.simplify(G1_a25 - R_sum25 / ((Lm * Cm * s_c**2 + 1) * (R1m + R_sum25) + s_c * Cm * R_sum25 * R1m)) == 0
          and sp.simplify((sp.Matrix([[0, 1, 0]]) * (s_c * sp.eye(3) - A_a26).inv() * b_a26)[0]
                          - (1 / (C2m * s_c)) * sol_a26[i2a] / em) == 0)

    # Exercises A2-7 ~ A2-12 (Mechanical / 3-tank / Analogous)
    f1m, f2m, k1m, k2m, k3m, Bm = sp.symbols("f1m f2m k1m k2m k3m Bm", positive=True)
    Xi_a, Xo_a, Xa_a, Xc_a = sp.symbols("Xi_a Xo_a Xa_a Xc_a")
    sol_a27b = sp.solve([k1m * (Xi_a - Xa_a) - k2m * Xo_a, f1m * s_c * (Xa_a - Xo_a) - k2m * Xo_a], [Xa_a, Xo_a])
    h1a, h2a, h3a, Qia = sp.symbols("h1a h2a h3a Qia")
    sol_a29 = sp.solve([
        A1t * s_c * h1a - (Qia - (h1a - h2a) / R1m),
        A2t * s_c * h2a - ((h1a - h2a) / R1m - h2a / R2m),
        A3t * s_c * h3a - (h2a / R2m - h3a / R3m),
    ], [h1a, h2a, h3a])
    a0_29 = A1t * R1m * A2t * R2m * A3t * R3m
    a1_29 = A1t * R1m * A2t * R2m + A1t * R1m * A3t * R3m + A2t * R2m * A3t * R3m + A1t * R2m * A3t * R3m
    a2_29 = A1t * R1m + A2t * R2m + A3t * R3m + A1t * R2m
    ka10, kb10 = k1m + k2m, k2m + k3m
    A_a210 = sp.Matrix([[0, 1, 0, 0], [-ka10, 0, k2m, 0], [0, 0, 0, 1], [k2m, 0, -kb10, -Bm]])
    tf_a210_ss = sp.simplify((sp.Matrix([[0, 0, 1, 0]]) * (s_c * sp.eye(4) - A_a210).inv() * sp.Matrix([0, 1, 0, 0]))[0])
    sol_a212 = sp.solve([(f2m * s_c + k2m) * (Xi_a - Xo_a) - f1m * s_c * (Xo_a - Xc_a), f1m * s_c * (Xo_a - Xc_a) - k1m * Xc_a], [Xo_a, Xc_a])
    check("ch02_ex_a2_7_to_12_mech_fluid",
          sp.simplify(sol_a27b[Xo_a] / Xi_a - k1m * f1m * s_c / (f1m * (k1m + k2m) * s_c + k1m * k2m)) == 0
          and sp.simplify(sol_a29[h3a] / Qia - R3m / (a0_29 * s_c**3 + a1_29 * s_c**2 + a2_29 * s_c + 1)) == 0
          and sp.simplify(tf_a210_ss - k2m / (s_c**4 + Bm * s_c**3 + (ka10 + kb10) * s_c**2 + Bm * ka10 * s_c + (ka10 * kb10 - k2m**2))) == 0
          and sp.simplify(sol_a212[Xo_a] / Xi_a - ((f1m / k1m) * s_c + 1) * ((f2m / k2m) * s_c + 1) / (((f1m / k1m) * s_c + 1) * ((f2m / k2m) * s_c + 1) + (f1m / k2m) * s_c)) == 0)

    # Exercises A2-13 ~ A2-22 (Block diagrams & op-amps)
    G_a214 = 20 / (6 * s_c + 10)
    H_a214 = 10 / (20 * s_c + 5)
    C0m = sp.Symbol("C0m", positive=True)
    sol_a216 = sp.solve([n1 - (R_s - H2 * (H1 * Y_s - G4 * R_s)), Y_s - G1 * (G2 + G3) * n1], [n1, Y_s])
    sol_a217 = sp.solve([n1 - (R_s - Y_s - H1 * n3), n2 - (G1 * n1 - H2 * Y_s), n3 - G2 * n2, Y_s - (G3 * n3 + G4 * n2)], [n1, n2, n3, Y_s])
    sol_a218 = sp.solve([Y_s - (G1 * R_s + G2 * n1), n1 - G3 * (Y_s - G4 * R_s)], [Y_s, n1])
    sol_a219 = sp.solve([E_s - (R_s - Y_s), n1 - G1 * (-E_s + n2), n2 - G2 * (E_s - n1), Y_s - (n1 + n2)], [E_s, n1, n2, Y_s])
    D1a, D2a, D3a = sp.symbols("D1a D2a D3a")
    sol_a220 = sp.solve([n1 - (R_s - H1 * (Y_s + D3a)), n2 - (G1 * n1 + D1a - D2a - H2 * Y_s), Y_s - G2 * n2], [n1, n2, Y_s])
    sol_a221 = sp.solve([n1 - (R_s - (H2 * n3 - H1 * Y_s)), n2 - (G1 * n1 - H3 * n3), n3 - G3 * (G2 * n2 - H4 * Y_s), Y_s - G4 * n3], [n1, n2, n3, Y_s])
    Na = sp.Symbol("Na")
    sol_a222a = sp.solve([n1 - (R_s - Y_s - H1 * n2), n2 - G2 * (G1 * n1 + G3 * Na), Y_s - (n2 - Na)], [n1, n2, Y_s])
    sol_a222b = sp.solve([n1 - (R_s - Y_s), n2 - (G2 * (n1 + G1 * R_s) + G3 * n1 + Na), Y_s - G4 * n2], [n1, n2, Y_s])
    check("ch02_ex_a2_13_to_22_blocks",
          sp.simplify(10 * G_a214 / (1 + G_a214 * H_a214) - (400 * s_c + 100) / (12 * s_c**2 + 23 * s_c + 25)) == 0
          and sp.simplify(-(1 / (1 / R1m + 1 / (R2m + 1 / (C2m * s_c)))) / R0 - (-R1m / R0 * (R2m * C2m * s_c + 1) / ((R1m + R2m) * C2m * s_c + 1))) == 0
          and sp.simplify(sol_a216[Y_s] / R_s - G1 * (G2 + G3) * (1 + G4 * H2) / (1 + G1 * (G2 + G3) * H1 * H2)) == 0
          and sp.simplify(sol_a217[Y_s] / R_s - (G1 * G2 * G3 + G1 * G4) / (1 + G1 * G2 * H1 + G2 * G3 * H2 + G4 * H2 + G1 * G2 * G3 + G1 * G4)) == 0
          and sp.simplify(sol_a218[Y_s] / R_s - (G1 - G2 * G3 * G4) / (1 - G2 * G3)) == 0
          and sp.simplify(sol_a219[Y_s] / R_s - (2 * G1 * G2 + G2 - G1) / (1 + G2 - G1 + 3 * G1 * G2)) == 0
          and sp.simplify(sol_a220[Y_s] - (G1 * G2 * R_s + G2 * (D1a - D2a) - G1 * G2 * H1 * D3a) / (1 + G2 * H2 + G1 * G2 * H1)) == 0
          and sp.simplify(sol_a221[Y_s] / R_s - G1 * G2 * G3 * G4 / (1 + G2 * G3 * H3 + G1 * G2 * G3 * H2 + G3 * G4 * H4 - G1 * G2 * G3 * G4 * H1)) == 0
          and sp.simplify(sol_a222a[Y_s].subs(R_s, 0) / Na - (G2 * G3 - 1 - G1 * G2 * H1) / (1 + G1 * G2 + G1 * G2 * H1)) == 0
          and sp.simplify(sol_a222b[Y_s].subs(Na, 0) / R_s - (G4 * (G2 + G3) + G1 * G2 * G4) / (1 + G4 * (G2 + G3))) == 0)

    # Exercises A2-23 & A2-24 (a, b, c, d) SFG Mason
    G7m, G8m, H5m = sp.symbols("G7m G8m H5m")
    xx2, xx3, xx4, xx5, xx6, xx7, xx8 = sp.symbols("xx2 xx3 xx4 xx5 xx6 xx7 xx8")
    sol_a223 = sp.solve([xx2 - (R_s - H1 * xx4 - xx5), xx3 - G1 * xx2, xx4 - G2 * xx3, xx5 - (G3 * xx4 + G5 * xx3 - H2 * xx6), xx6 - G4 * xx5, Y_s - (xx6 + G6 * xx2)], [xx2, xx3, xx4, xx5, xx6, Y_s])
    sol_a224a = sp.solve([xx2 - R_s, xx3 - (G1 * xx2 - H2 * xx5), xx4 - (G2 * xx3 - H1 * xx5 - H3 * xx6), xx5 - G3 * xx4, xx6 - G4 * xx5, xx7 - (G5 * xx6 + G6 * xx2)], [xx2, xx3, xx4, xx5, xx6, xx7])
    sol_a224b = sp.solve([xx2 - (R_s - H5m * xx8), xx3 - (G1 * xx2 - H1 * xx4 - H4 * xx7), xx4 - (G2 * xx3 + G7m * xx2), xx5 - (G3 * xx4 - H2 * xx6), xx6 - G4 * xx5, xx7 - (G5 * xx6 + G8m * xx3 - H3 * xx8), xx8 - G6 * xx7], [xx2, xx3, xx4, xx5, xx6, xx7, xx8])
    sol_a224c = sp.solve([xx2 - R_s, xx3 - (5 * xx2 - xx4), xx4 - (10 * xx3 - 2 * xx5), xx6 - (10 * xx2 - sp.Rational(1, 2) * xx6), xx5 - (xx4 + 2 * xx6)], [xx2, xx3, xx4, xx5, xx6])
    av, bv, cv, dv, ev, fv, gv, hv = sp.symbols("av bv cv dv ev fv gv hv")
    sol_a224d = sp.solve([xx2 - (R_s + fv * xx3), xx3 - (av * xx2 + gv * xx4), xx4 - (bv * xx3 + hv * xx5), xx5 - (cv * xx4 + ev * xx2), xx6 - dv * xx5], [xx2, xx3, xx4, xx5, xx6])
    num224b, _ = sp.fraction(sp.cancel(sol_a224b[xx8] / R_s))
    check("ch02_ex_a2_23_24_sfg_mason",
          sp.simplify(sol_a223[Y_s] / R_s - (G6 * (1 + G4 * H2) + G1 * G2 * G3 * G4 + G1 * G4 * G5) / (1 + G1 * G2 * H1 + G1 * G2 * G3 + G4 * H2 + G1 * G5 + G1 * G2 * G4 * H1 * H2)) == 0
          and sp.simplify(sol_a224a[xx7] / R_s - (G6 + G1 * G2 * G3 * G4 * G5 / (1 + G3 * H1 + G2 * G3 * H2 + G3 * G4 * H3))) == 0
          and sp.simplify(num224b - (G3 * G4 * G5 * G6 * (G1 * G2 + G7m) + G8m * G6 * (G1 - G7m * H1) * (1 + G4 * H2))) == 0
          and sol_a224c[xx5] / R_s == sp.Rational(590, 39)
          and sp.simplify(sol_a224d[xx6] / R_s - (av * bv * cv * dv + ev * dv * (1 - bv * gv)) / (1 - av * fv - bv * gv - cv * hv - ev * hv * gv * fv + av * fv * cv * hv)) == 0)

    # Exercises A2-25 ~ A2-33 + Self-test B2-23 & B2-25
    A_a227 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, -12, -7]])
    A_a228 = sp.Matrix([[0, 1], [-6, 5]])
    A_a229 = sp.diag(2, 4, 5)
    A_a230 = sp.Matrix([[-2, 10, 0], [0, -1, 10], [-8, 0, 0]])
    Go_a231 = sp.Matrix([[1 / (2 * s_c + 1), 0], [-1, 1 / (s_c + 1)]])
    Gc_a231 = sp.Matrix([[(2 * s_c + 1) / s_c, 0], [(2 * s_c**2 + 3 * s_c + 1) / s_c, (s_c + 1) / (5 * s_c)]])
    M_a231 = sp.simplify((sp.eye(2) + Go_a231 * Gc_a231).inv() * (Go_a231 * Gc_a231))
    # B2-23: k(t) = e^{-0.5t} - e^{-t} -> Phi(s) = 1/(s+0.5) - 1/(s+1) = 0.5/((s+0.5)(s+1)) -> G(s) = Phi/(1-Phi) = 0.5/(s^2+1.5s) = 1/(s(2s+3))
    Phi_b223 = sp.simplify(1 / (s_c + sp.Rational(1, 2)) - 1 / (s_c + 1))
    G_b223 = sp.simplify(Phi_b223 / (1 - Phi_b223))
    # B2-25: Tractor M1, Trailer M2, coupling (K || B1), trailer friction B2, input f_i on M1
    B1m, B2m, Km_s = sp.symbols("B1m B2m Km_s", positive=True)
    Y1m, Y2m, Fim = sp.symbols("Y1m Y2m Fim")
    sol_b225 = sp.solve([
        M1 * s_c**2 * Y1m + (B1m * s_c + Km_s) * (Y1m - Y2m) - Fim,
        M2 * s_c**2 * Y2m + B2m * s_c * Y2m - (B1m * s_c + Km_s) * (Y1m - Y2m),
    ], [Y1m, Y2m])
    den_b225 = s_c * (M1 * M2 * s_c**3 + (M1 * (B1m + B2m) + M2 * B1m) * s_c**2 + ((M1 + M2) * Km_s + B1m * B2m) * s_c + B2m * Km_s)
    dF_a233 = float((sp.Rational(1265, 100) * sp.Rational(11, 10) * sp.Rational(1, 4)**sp.Rational(1, 10)).evalf())
    check("ch02_ex_a2_25_to_33_and_b2",
          sp.simplify((sp.Matrix([[sp.Rational(1, 6), sp.Rational(-2, 3), sp.Rational(3, 2)]]) * (s_c * sp.eye(3) - sp.diag(0, -3, -4)).inv() * sp.Matrix([1, 1, 1]))[0] - (s_c**2 + 3 * s_c + 2) / (s_c * (s_c**2 + 7 * s_c + 12))) == 0
          and sp.simplify((sp.Matrix([[1, 0]]) * (s_c * sp.eye(2) - A_a228).inv() * sp.Matrix([1, 1]))[0] - (s_c - 4) / (s_c**2 - 5 * s_c + 6)) == 0
          and sp.simplify((sp.Matrix([[sp.Rational(7, 2), sp.Rational(-57, 2), 27]]) * (s_c * sp.eye(3) - A_a229).inv() * sp.Matrix([1, 1, 1]))[0] - (2 * s_c**2 + 6 * s_c + 1) / (s_c**3 - 11 * s_c**2 + 38 * s_c - 40)) == 0
          and sp.simplify((sp.Matrix([[1, 0, 0]]) * (s_c * sp.eye(3) - A_a230).inv() * sp.Matrix([0, 0, 4]))[0] - 400 / (s_c**3 + 3 * s_c**2 + 2 * s_c + 800)) == 0
          and sp.simplify(M_a231 - sp.diag(1 / (s_c + 1), 1 / (5 * s_c + 1))) == sp.zeros(2, 2)
          and abs(dF_a233 - 12.1137) < 0.001
          and sp.simplify(G_b223 - 1 / (s_c * (2 * s_c + 3))) == 0
          and sp.simplify(sol_b225[Y1m] / Fim - (M2 * s_c**2 + (B1m + B2m) * s_c + Km_s) / den_b225) == 0
          and sp.simplify(sol_b225[Y2m] / Fim - (B1m * s_c + Km_s) / den_b225) == 0)

    # --- Ch03 Full Textbook Edition (Main Ex 3-1..3-19, Guide Ex 3-7/9/10/13, A3-1..A3-32, B3-11..B3-20) ---
    def _ilt3(expr):
        return sp.simplify(sp.inverse_laplace_transform(sp.apart(expr, s_c), s_c, t))

    # 1. P.95 (3-25) parabolic SS errata + P.102 Ex 3-5 ISE + P.119-120 PD/PI damping
    a0c, a1c, a2c = sp.symbols("a0c a1c a2c", positive=True)
    c0c, c1c, c2c = sp.symbols("c0c c1c c2c", real=True)
    y_p3 = c0c + c1c * t + sp.Rational(1, 2) * c2c * t**2
    sol_p3 = sp.solve(sp.Poly(a2c * sp.diff(y_p3, t, 2) + a1c * sp.diff(y_p3, t) + a0c * y_p3 - sp.Rational(1, 2) * t**2, t).coeffs(), [c0c, c1c, c2c])
    s1_ise = -zeta * wn + wn * sp.sqrt(zeta**2 - 1)
    s2_ise = -zeta * wn - wn * sp.sqrt(zeta**2 - 1)
    C1_ise, C2_ise = s2_ise / (s2_ise - s1_ise), -s1_ise / (s2_ise - s1_ise)
    J1_ise = sp.simplify(-C1_ise**2 / (2 * s1_ise) - 2 * C1_ise * C2_ise / (s1_ise + s2_ise) - C2_ise**2 / (2 * s2_ise))
    zeta_ise_opt = sp.solve(sp.diff(J1_ise, zeta), zeta)
    check("ch03_f_parabolic_ss_and_ise_errata",
          sp.simplify(sol_p3[c0c] - (a1c**2 / a0c**3 - a2c / a0c**2)) == 0
          and sp.simplify(J1_ise - (zeta / wn + 1 / (4 * zeta * wn))) == 0
          and zeta_ise_opt == [sp.Rational(1, 2)],
          "Confirmed P.95 (3-25) +a1^2/a0^3 & P.102 Ex3-5 ISE opt zeta=0.5")

    # 2. Main Ex 3-2, 3-3, 3-4, 3-6, 3-7, 3-8
    y_m32 = _ilt3((s_c + 2) / (s_c * (s_c + 1) * (s_c + 3)))
    y_m33 = _ilt3(1 / ((s_c + 2)**3 * (s_c + 3)))
    y_m34 = _ilt3(10 / ((s_c**2 + 6 * s_c + 25) * (s_c + 2)))
    A1_m34 = complex((10 / ((s_c + 3 + 4 * sp.I) * (s_c + 2))).subs(s_c, -3 + 4 * sp.I).evalf())
    wn38 = math.sqrt(1.117)
    z38 = 0.6 / (2 * wn38)
    wd38 = wn38 * math.sqrt(1 - z38**2)
    check("ch03_main_ex3_2_to_3_8",
          sp.simplify(y_m32 - (sp.Rational(2, 3) - sp.exp(-t) / 2 - sp.exp(-3 * t) / 6)) == 0
          and sp.simplify(y_m33 - (t**2 * sp.exp(-2 * t) / 2 - t * sp.exp(-2 * t) + sp.exp(-2 * t) - sp.exp(-3 * t))) == 0
          and abs(2 * abs(A1_m34) - 0.6063) < 0.001
          and abs(math.pi / wd38 - 3.100) < 0.005
          and abs(math.exp(-math.pi * z38 / math.sqrt(1 - z38**2)) * 100 - 39.46) < 0.1
          and abs((1 - 1.037 / 1.117) - 0.0716) < 0.001)

    # 3. Main Ex 3-9 & Ex 3-10 (proves P.114 0.188 typo -> true +1.4118)
    y_m39 = _ilt3(1 / (s_c * (s_c + 6) * (s_c**2 + 2 * s_c + 5)))
    y_m310 = _ilt3(sp.Rational(125, 2) * (s_c + sp.Rational(5, 2)) / (s_c * (s_c + sp.Rational(25, 4)) * (s_c**2 + 6 * s_c + 25)))
    c_m310 = float(sp.N(y_m310.coeff(sp.exp(-sp.Rational(25, 4) * t))))
    f_m310 = sp.lambdify(t, y_m310, "math")
    mp_m310 = max(f_m310(i * 0.002) for i in range(1, 1500))
    check("ch03_main_ex3_9_3_10_errata",
          abs(float(sp.N(y_m39.subs(t, 0)))) < 1e-12
          and abs(float(sp.N(y_m310.subs(t, 0)))) < 1e-12
          and abs(c_m310 - 1.4118) < 0.001
          and abs((mp_m310 - 1) * 100 - 37.97) < 0.1,
          f"Ex3-10 true coeff=+{c_m310:.4f} (book typo +0.188), sim overshoot={(mp_m310-1)*100:.2f}%")

    # 4. Main Ex 3-13, 3-14/15, 3-16, 3-18, 3-19
    A_m313 = sp.Matrix([[0, 1, 0], [0, 0, 1], [2, 3, 0]])
    eA_m313 = sp.simplify(sp.exp(A_m313 * t))
    A_m314 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
    eA_m314 = sp.simplify(sp.exp(A_m314 * t))
    A_m316 = sp.Matrix([[0, 2], [-1, -3]])
    eA_m316 = sp.simplify(sp.exp(A_m316 * t))
    A_m318 = sp.Matrix([[0, 1], [-1, -2]])
    X_m318 = (s_c * sp.eye(2) - A_m318).inv() * (sp.Matrix([1, 0]) + sp.Matrix([0, 1]) * (1 + s_c) / (s_c**2 + 1))
    y_m318 = _ilt3(X_m318[0])
    x10_s, x20_s = sp.symbols("x10_s x20_s", real=True)
    A_m319 = sp.Matrix([[0, 1], [-wn**2, -2 * zeta * wn]])
    Y_m319 = sp.simplify((sp.Matrix([[1, 0]]) * (s_c * sp.eye(2) - A_m319).inv() * (sp.Matrix([x10_s, x20_s]) + sp.Matrix([0, wn**2]) / s_c))[0])
    check("ch03_main_ex3_13_to_3_19",
          sp.simplify(9 * eA_m313[0, 0] - (sp.exp(2 * t) + (8 + 6 * t) * sp.exp(-t))) == 0
          and sp.simplify(9 * eA_m313[2, 1] - (8 * sp.exp(2 * t) + (-8 + 3 * t) * sp.exp(-t))) == 0
          and sp.simplify(eA_m314 - sp.Matrix([[1, sp.sinh(t), sp.cosh(t) - 1], [0, sp.cosh(t), sp.sinh(t)], [0, sp.sinh(t), sp.cosh(t)]])) == sp.zeros(3, 3)
          and sp.simplify(eA_m316[0, 1] - (2 * sp.exp(-t) - 2 * sp.exp(-2 * t))) == 0
          and sp.simplify(y_m318 - (sp.Rational(3, 2) * sp.exp(-t) + t * sp.exp(-t) + sp.sin(t) / 2 - sp.cos(t) / 2)) == 0
          and sp.simplify(Y_m319 - ((s_c + 2 * zeta * wn) * x10_s + x20_s + wn**2 / s_c) / (s_c**2 + 2 * zeta * wn * s_c + wn**2)) == 0)

    # 5. Missing Guide Examples 3-7, 3-9, 3-10 (with Gn sign errata check), 3-13
    z_g310 = math.sqrt(math.log(0.25)**2 / (math.pi**2 + math.log(0.25)**2))
    wn_g310 = math.pi / (2 * math.sqrt(1 - z_g310**2))
    K1_g310, Kt_g310 = wn_g310**2, 2 * z_g310 * wn_g310 / wn_g310**2
    # Guide Ex 3-13: (s^2+2s+2)(s+6) = s^3 + 8s^2 + 14s + 12 -> a=4.5, b=10.5, c=11
    poly_g313 = sp.expand((s_c**2 + 2 * s_c + 2) * (s_c + 6))
    check("ch03_guide_ex3_7_9_10_13",
          sp.expand((s_c + 5)**2 + (5 * sp.sqrt(3))**2) == s_c**2 + 10 * s_c + 100
          and abs(K1_g310 - 2.948) < 0.01 and abs(Kt_g310 - 0.470) < 0.01
          and sp.expand(s_c**3 + (sp.Rational(9, 2) + sp.Rational(7, 2)) * s_c**2 + (sp.Rational(21, 2) + sp.Rational(7, 2)) * s_c + (11 + 1)) == poly_g313)

    # 6. Exercises A3-1 ~ A3-8
    y_a31_1 = _ilt3(9 / (s_c * (s_c**2 + 4 * s_c + 3)))
    y_a34 = _ilt3(6 / (s_c * (s_c**3 + 4 * s_c**2 + 4 * s_c)))
    y_a35 = _ilt3(4 / (s_c * (s_c**2 + 5 * s_c + 4)))
    y_a36_step = _ilt3((2 * s_c + 1) / (s_c * (s_c + 1)**2))
    y_a36_ramp = _ilt3((2 * s_c + 1) / (s_c**2 * (s_c + 1)**2))
    y_a37_imp = _ilt3(10 * (2 * s_c + 1) / ((s_c + 1) * (s_c**2 + 4 * s_c + 8)))
    y_a37_step = _ilt3(10 * (2 * s_c + 1) / (s_c * (s_c + 1) * (s_c**2 + 4 * s_c + 8)))
    G_a38_5 = sp.simplify(sp.Rational(2, 100) * (1 / (s_c + sp.Rational(1, 2)) - 1 / (s_c + sp.Rational(1, 5))))
    check("ch03_ex_a3_1_to_8",
          sp.simplify(y_a31_1 - (3 + sp.Rational(3, 2) * sp.exp(-3 * t) - sp.Rational(9, 2) * sp.exp(-t))) == 0
          and sp.simplify(y_a34 - sp.Rational(3, 2) * (t - 1 + t * sp.exp(-2 * t) + sp.exp(-2 * t))) == 0
          and sp.simplify(y_a35 - (1 + sp.Rational(1, 3) * sp.exp(-4 * t) - sp.Rational(4, 3) * sp.exp(-t))) == 0
          and sp.simplify(y_a36_step - (1 - sp.exp(-t) + t * sp.exp(-t))) == 0
          and sp.simplify(y_a36_ramp - (t - t * sp.exp(-t))) == 0
          and sp.simplify(y_a37_imp - (-2 * sp.exp(-t) + 2 * sp.exp(-2 * t) * sp.cos(2 * t) + 11 * sp.exp(-2 * t) * sp.sin(2 * t))) == 0
          and sp.simplify(y_a37_step - (sp.Rational(5, 4) + 2 * sp.exp(-t) - sp.Rational(13, 4) * sp.exp(-2 * t) * sp.cos(2 * t) - sp.Rational(9, 4) * sp.exp(-2 * t) * sp.sin(2 * t))) == 0
          and sp.simplify(G_a38_5 - (-sp.Rational(6, 100) / ((2 * s_c + 1) * (5 * s_c + 1)))) == 0)

    # 7. Exercises A3-9 ~ A3-14
    z_41_c = math.log(4) / math.sqrt(4 * math.pi**2 + math.log(4)**2)
    Kc_a311 = ((2.5 / (2 * z_41_c))**2 - 1) / 2
    Kc_a312 = (4 * (1 / (2 * z_41_c))**2 - 1) / 4
    y_a314b = _ilt3((s_c + 1) / (s_c * (s_c**2 + s_c + 1)))
    f_a314b = sp.lambdify(t, y_a314b, "math")
    sig_14b = (max(f_a314b(i * 0.005) for i in range(1, 2000)) - 1) * 100
    check("ch03_ex_a3_9_to_14",
          abs(math.sqrt(600) - 24.4949) < 0.001 and abs(70 / (2 * math.sqrt(600)) - 1.4289) < 0.001
          and abs(math.exp(-math.pi * 0.6 / 0.8) * 100 - 9.478) < 0.01 and abs(math.pi / 1.6 - 1.9635) < 0.001
          and abs(Kc_a311 - 16.330) < 0.01 and abs(Kc_a312 - 5.136) < 0.01
          and abs(36 / 25 - 1.44) < 1e-9 and abs(11.2 / 36 - 0.3111) < 0.001
          and abs(sig_14b - 29.84) < 0.1)

    # 8. Exercises A3-15 ~ A3-18
    Kc_s15 = sp.Symbol("Kc_s15", positive=True)
    sol_15_0 = [float(x) for x in sp.solve(70**2 - 4 * sp.Rational(49, 100) * 600 * (1 + Kc_s15), Kc_s15)]
    sol_15_3 = sorted(float(x) for x in sp.solve((70 + 3 * Kc_s15)**2 - 4 * sp.Rational(49, 100) * 600 * (1 + Kc_s15), Kc_s15))
    z_a317 = math.sqrt(math.log(0.3)**2 / (math.pi**2 + math.log(0.3)**2))
    wn_a317 = math.pi / (0.1 * math.sqrt(1 - z_a317**2))
    check("ch03_ex_a3_15_to_18",
          abs(sol_15_0[0] - 3.1667) < 0.001
          and abs(sol_15_3[0] - 5.2546) < 0.001 and abs(sol_15_3[1] - 78.7454) < 0.001
          and abs(wn_a317**2 / (2 * z_a317 * wn_a317) - 47.01) < 0.05)

    # 9. Exercises A3-19 ~ A3-23
    eA_19_1 = sp.simplify(sp.exp(sp.Matrix([[0, 1], [-6, -5]]) * t))
    eA_20_6 = sp.simplify(sp.exp(sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 1, 2]]) * t))
    eA_21 = sp.simplify(sp.exp(sp.Matrix([[0, 1, 0], [0, 0, 1], [-6, -11, -6]]) * t))
    A_a322 = sp.Matrix([[-12, sp.Rational(2, 3)], [-36, -1]])
    X_a322 = (s_c * sp.eye(2) - A_a322).inv() * (sp.Matrix([2, 1]) + sp.Matrix([sp.Rational(1, 3), 1]) / s_c)
    eA_23 = sp.simplify(sp.exp(sp.Matrix([[0, 1, 0], [0, 0, 1], [-5, -7, -3]]) * t))
    check("ch03_ex_a3_19_to_23",
          sp.simplify(eA_19_1[0, 0] - (3 * sp.exp(-2 * t) - 2 * sp.exp(-3 * t))) == 0
          and sp.simplify(eA_20_6[2, 1] - (sp.exp(2 * t) - sp.exp(t))) == 0
          and sp.simplify(eA_21[0, 0] - (3 * sp.exp(-t) - 3 * sp.exp(-2 * t) + sp.exp(-3 * t))) == 0
          and sp.simplify(_ilt3(X_a322[0]) - (sp.Rational(1, 36) + sp.Rational(136, 45) * sp.exp(-9 * t) - sp.Rational(21, 20) * sp.exp(-4 * t))) == 0
          and sp.simplify(_ilt3(X_a322[1]) - (sp.Rational(68, 5) * sp.exp(-9 * t) - sp.Rational(63, 5) * sp.exp(-4 * t))) == 0
          and sp.simplify(eA_23[0, 2] - sp.Rational(1, 4) * (1 - sp.cos(2 * t)) * sp.exp(-t)) == 0)

    # 10. Exercises A3-24 ~ A3-27 (captures A3-24(1) row2 col1 erratum)
    A_a324 = sp.Matrix([[-6, 4], [-2, 0]])
    eA_24 = sp.simplify(sp.exp(A_a324 * t))
    X_a324 = (s_c * sp.eye(2) - A_a324).inv() * (sp.Matrix([2, 0]) + sp.Matrix([0, 1]) / s_c)
    A_a325 = sp.Matrix([[-2, 1], [2, -3]])
    Y_a325 = ((sp.Matrix([[1, 0]]) * (s_c * sp.eye(2) - A_a325).inv()) * (sp.Matrix([0, 1]) + sp.Matrix([0, 1]) / s_c))[0]
    A_a326 = sp.Matrix([[-6, -1], [5, 0]])
    X_a326 = (s_c * sp.eye(2) - A_a326).inv() * sp.Matrix([1, 0]) / s_c
    A_a327 = sp.Matrix([[-2, -1], [0, 3]])
    X_a327 = (s_c * sp.eye(2) - A_a327).inv() * sp.Matrix([0, 10]) / s_c
    check("ch03_ex_a3_24_to_27",
          sp.simplify(eA_24[1, 0] - (-sp.exp(-2 * t) + sp.exp(-4 * t))) == 0
          and sp.simplify(_ilt3(X_a324[0]) - (sp.Rational(1, 2) - 3 * sp.exp(-2 * t) + sp.Rational(9, 2) * sp.exp(-4 * t))) == 0
          and sp.simplify(_ilt3(Y_a325) - sp.Rational(1, 4) * (1 - sp.exp(-4 * t))) == 0
          and sp.simplify(_ilt3(X_a326[1]) - sp.Rational(1, 4) * (4 - 5 * sp.exp(-t) + sp.exp(-5 * t))) == 0
          and sp.simplify(_ilt3(X_a327[0]) - (-sp.exp(-2 * t) - sp.Rational(2, 3) * sp.exp(3 * t) + sp.Rational(5, 3))) == 0,
          "Confirmed A3-24(1) Phi21 = -e^-2t + e^-4t (Appendix 1 typo -e^-4t)")

    # 11. Exercises A3-28 ~ A3-32 (captures A3-28(1)(2) & A3-30(3) major errata + A3-32 two-point fit)
    X_a328_2 = sp.Matrix([[2 * sp.exp(-t), sp.exp(-t) + 2 * t * sp.exp(-t)], [sp.exp(-t), sp.exp(-t) + t * sp.exp(-t)]])
    Phi_a328_2 = sp.simplify(X_a328_2 * sp.Matrix([[2, 1], [1, 1]]).inv())
    A_a328_2 = sp.simplify(sp.diff(Phi_a328_2, t).subs(t, 0))
    Phi_a330 = sp.Matrix([[1, (1 - sp.exp(-2 * t)) / 2], [0, sp.exp(-2 * t)]])
    T_a332 = (210 - 150) / (math.log(1 - 0.379) - math.log(1 - 0.595))
    tau_a332 = 150 + T_a332 * math.log(1 - 0.379)
    check("ch03_ex_a3_28_to_32",
          A_a328_2 == sp.Matrix([[-3, 4], [-1, 1]])
          and sp.simplify(sp.exp(A_a328_2 * t) - Phi_a328_2) == sp.zeros(2, 2)
          and sp.simplify(Phi_a330.inv() - Phi_a330.subs(t, -t)) == sp.zeros(2, 2)
          and abs(T_a332 - 140.4) < 0.5 and abs(tau_a332 - 83.1) < 0.5,
          "Confirmed A3-28(2) A=[[-3,4],[-1,1]], A3-30 Phi^-1(t)=+Phi(-t)")

    # 12. Self-test B3-13, B3-14, B3-15, B3-16, B3-18, B3-20
    e_b313 = _ilt3((s_c + 2) / (s_c**2 + 4 * s_c + 2))
    char_b320 = sp.expand(s_c * (s_c + 1) * (sp.Rational(1, 18) * s_c + 1) + 10 * sp.Rational(4, 5) * (sp.Rational(13, 48) * s_c + 1))
    check("ch03_selftest_b3_11_to_20",
          sp.simplify(e_b313 - sp.Rational(1, 2) * (sp.exp((-2 - sp.sqrt(2)) * t) + sp.exp((-2 + sp.sqrt(2)) * t))) == 0
          and sp.factor(char_b320) == (s_c + 16) * (s_c**2 + 3 * s_c + 9) / 18
          and abs((1.4 / 0.25)**2 - 31.36) < 1e-9
          and abs(1 - sp.Rational(10, 9) - (-sp.Rational(1, 9))) == 0)

    # --- Ch04 Full Textbook Edition (Main Ex 4-1..4-15, Guide Ex 4-5, A4-1..A4-16, B4-6..B4-15) ---
    def _rc4(poly_expr, var=s_c):
        rts = sp.nroots(sp.Poly(poly_expr, var), n=14)
        rhp = sum(1 for r in rts if sp.re(r) > 1e-7)
        im = sum(1 for r in rts if abs(sp.re(r)) <= 1e-7)
        lhp = sum(1 for r in rts if sp.re(r) < -1e-7)
        return rhp, im, lhp

    # 1. Main Ex 4-1 ~ 4-6
    check("ch04_main_ex4_1_to_4_6",
          _rc4(s_c**5 + s_c**4 + 10 * s_c**3 + 72 * s_c**2 + 152 * s_c + 240) == (2, 0, 3)
          and _rc4(s_c**6 + 3 * s_c**5 + 2 * s_c**4 + 9 * s_c**3 + 5 * s_c**2 + 12 * s_c + 20) == (2, 0, 4)
          and _rc4(s_c**4 + s_c**3 + 2 * s_c**2 + 2 * s_c + 5) == (2, 0, 2)
          and _rc4(s_c**4 + 3 * s_c**3 + 3 * s_c**2 + 3 * s_c + 2) == (0, 2, 2)
          and _rc4(s_c**4 + 2 * s_c**3 + 11 * s_c**2 + 18 * s_c + 18) == (0, 2, 2)
          and _rc4(s_c**6 + s_c**5 + 5 * s_c**4 + 3 * s_c**3 + 8 * s_c**2 + 2 * s_c + 4) == (0, 4, 2))

    # 2. Main Ex 4-7 ~ 4-15
    z4 = sp.Symbol("z4")
    K4 = sp.Symbol("K4", real=True)
    kmax47 = [float(x) for x in sp.solve((80 - K4) * (25 + K4) - 98 * K4, K4) if float(x) > 0][0]
    kmax49 = max(float(x) for x in sp.solve((sp.Rational(61, 4) - K4) * (sp.Rational(59, 4) + K4) - 25 * (sp.Rational(3, 2) * K4 - sp.Rational(153, 16)), K4))
    p48_sh = sp.expand((2 * s_c**3 + 10 * s_c**2 + 13 * s_c + 4).subs(s_c, z4 - 1))
    check("ch04_main_ex4_7_to_4_15",
          abs(kmax47 - 28.121) < 0.005
          and _rc4(2 * s_c**3 + 10 * s_c**2 + 13 * s_c + 4) == (0, 0, 3)
          and _rc4(p48_sh, z4) == (1, 0, 2)
          and abs(kmax49 - 9.8945) < 0.002
          and _rc4(s_c**3 + s_c**2 + 10 * s_c + 5) == (0, 0, 3)
          and _rc4(s_c**5 + 2 * s_c**4 + 4 * s_c**3 + 8 * s_c**2 + 10 * s_c + 6) == (2, 0, 3),
          f"Ex4-7 Kmax={kmax47:.3f}, Ex4-9 Kmax={kmax49:.4f} (book 9.8)")

    # 3. Missing Guide Ex 4-5
    check("ch04_guide_ex4_5",
          _rc4(s_c**3 + sp.Rational(3, 4) * s_c**2 + 4 * s_c + 3) == (0, 2, 1))

    # 4. Exercises A4-1 ~ A4-5
    T1c, T2c, T3c = sp.symbols("T1c T2c T3c", positive=True)
    kmax_a43_3 = sp.expand((T1c * T2c + T1c * T3c + T2c * T3c) * (T1c + T2c + T3c) / (T1c * T2c * T3c) - 1)
    p_a44_sh = sp.expand((2 * s_c**3 + 10 * s_c**2 + 13 * s_c + 5).subs(s_c, z4 - 1))
    p_a45_sh = sp.expand(((s_c + 1) * (s_c + sp.Rational(3, 2)) * (s_c + 2) + K4).subs(s_c, z4 - 1))
    check("ch04_ex_a4_1_to_5",
          _rc4(s_c**4 + 2 * s_c**3 + 2 * s_c**2 + 4 * s_c + 10) == (2, 0, 2)
          and _rc4(s_c**5 + s_c**4 + 4 * s_c**3 + 4 * s_c**2 + 2 * s_c + 1) == (2, 0, 3)
          and _rc4(s_c**4 + 2 * s_c**2 + 9) == (2, 0, 2)
          and _rc4(s_c**6 + 3 * s_c**5 + 5 * s_c**4 + 9 * s_c**3 + 8 * s_c**2 + 6 * s_c + 4) == (0, 4, 2)
          and _rc4(s_c * (s_c + 1) * (s_c + 5) + 20) == (0, 0, 3)
          and _rc4(s_c**3 * (s_c + 1) * (s_c + 2) + 5 * s_c + 1) == (2, 0, 3)
          and sp.simplify(kmax_a43_3 - ((T1c + T2c) / T3c + (T1c + T3c) / T2c + (T2c + T3c) / T1c + 2)) == 0
          and _rc4(2 * s_c**3 + 10 * s_c**2 + 13 * s_c + 5) == (0, 0, 3)
          and _rc4(p_a44_sh, z4) == (1, 1, 1)
          and sp.simplify(p_a45_sh - (z4**3 + sp.Rational(3, 2) * z4**2 + sp.Rational(1, 2) * z4 + K4)) == 0)

    # 5. Exercises A4-6 ~ A4-10
    K_a48 = sp.Rational(7995, 12)
    w2_a48 = sp.Rational(200 + K_a48, sp.Rational(105, 2))
    wn_a49 = math.pi / math.sqrt(0.75)
    check("ch04_ex_a4_6_to_10",
          _rc4(sp.Rational(1, 5) * s_c**3 + s_c**2 + 5 * s_c + 10) == (0, 0, 3)
          and K_a48 == sp.Rational(2665, 4) and w2_a48 == sp.Rational(33, 2)
          and abs(wn_a49**2 / 10 - 1.3159) < 0.001
          and abs((wn_a49 - 1) / 10 - 0.2628) < 0.001
          and abs(1.5 / wn_a49 - 0.4135) < 0.001)

    # 6. Exercises A4-11 ~ A4-16
    check("ch04_ex_a4_11_to_16",
          _rc4(s_c**2 * (s_c**2 + 6 * s_c + 100) + 10 * (2 * s_c + 1)) == (0, 0, 4)
          and sp.Rational(40, 25) == sp.Rational(8, 5)
          and abs(40 / 22.5 - 1.7778) < 0.001
          and sp.Rational(2, sp.Rational(20, 3)) == sp.Rational(3, 10)
          and abs(2740 / 9 - 304.444) < 0.005 and abs(1044 / 30 - 34.8) < 1e-9)

    # 7. Self-test B4-6 ~ B4-10
    check("ch04_selftest_b4_6_to_10",
          _rc4(2 * s_c**5 + s_c**4 - 15 * s_c**3 + 25 * s_c**2 + 2 * s_c - 7) == (3, 0, 2)
          and _rc4(s_c**4 + 7 * s_c**3 + 25 * s_c**2 + 42 * s_c + 30) == (0, 0, 4)
          and _rc4(s_c**6 + 2 * s_c**5 + 5 * s_c**4 + 28 * s_c**3 + 27 * s_c**2 + 66 * s_c + 63) == (2, 2, 2)
          and _rc4(s_c**3 + 3 * s_c**2 + 2 * s_c + 24) == (2, 0, 1)
          and abs((-5 + math.sqrt(45)) - 1.7082) < 0.001)

    # 8. Self-test B4-11 ~ B4-15
    p_b412_sh = sp.expand((s_c**3 + 7 * s_c**2 + 17 * s_c + K4).subs(s_c, z4 - 2))
    check("ch04_selftest_b4_11_to_15",
          _rc4(s_c**3 + 7 * s_c**2 + 17 * s_c + 119) == (0, 2, 1)
          and sp.simplify(p_b412_sh - (z4**3 + z4**2 + z4 + K4 - 14)) == 0
          and abs((1.4 / 0.25)**2 - 31.36) < 1e-9)

    # --- Ch05 Full Textbook Edition (Main Ex 5-1..5-17, Guide Ex 5-14..5-15, A5-1..A5-28, B5-12..B5-20) ---
    # 1. Main Ex 5-1 ~ 5-8 (includes Main 5-7 Routh s1 typo 2.75K vs 11K/12)
    eq54 = sp.numer(sp.together(sp.diff(-s_c * (s_c + 4) * (s_c + 9) / (s_c + 2), s_c)))
    d54_real = [float(sp.re(r)) for r in sp.nroots(sp.Poly(eq54, s_c)) if abs(sp.im(r)) < 1e-6][0]
    eq57 = sp.numer(sp.together(sp.diff(-s_c**2 * (s_c + 12) / (s_c + 1), s_c)))
    d57 = sorted([float(r.evalf()) for r in sp.solve(eq57, s_c) if r != 0])
    p58_rts = [complex(r.evalf()) for r in sp.nroots(sp.Poly(sp.diff(s_c * (s_c + 2) * ((s_c + 1)**2 + 4), s_c), s_c))]
    g52_ang = math.degrees(cmath.phase(((-1 + 2j) + 1) / ((-1 + 2j) * (((-1 + 2j) + 2)**2 + 4) * ((-1 + 2j) + 5))))
    check("ch05_main_ex5_1_to_5_8",
          abs(g52_ang - (-129.09)) < 0.2
          and abs(d54_real - (-6.268)) < 0.01
          and abs(d57[0] - (-5.186)) < 0.01 and abs(d57[1] - (-2.314)) < 0.01
          and sp.Rational(12 - 1, 12) == sp.Rational(11, 12)
          and abs(math.sqrt(2.5) - 1.5811) < 0.001 and abs(9 * 2.5 - 2.5**2 - 16.25) < 1e-9
          and any(abs(r.real + 1) < 1e-5 and abs(abs(r.imag) - math.sqrt(1.5)) < 1e-3 for r in p58_rts),
          "Confirmed Main 5-7 Routh s1=11K*/12 (book typo 2.75K*)")

    # 2. Main Ex 5-9 ~ 5-17 (includes Main 5-11 d1=-26.19 sign typo & Main 5-14 Pade & Main 5-17 PID)
    eq511 = sp.numer(sp.together(sp.diff(-(s_c + 50) * (s_c**2 + 5 * s_c + 25) / (s_c + 1), s_c)))
    d511 = sorted([float(sp.re(r)) for r in sp.nroots(sp.Poly(eq511, s_c)) if abs(sp.im(r)) < 1e-6])
    cl517 = sp.expand((s_c + 10) * (s_c**2 + 4 * s_c + 8))
    den514 = sp.expand((10 * s_c + 1) * (6 * s_c + 1) * (1 + sp.Rational(5, 2) * s_c + sp.Rational(20825, 10000) * s_c**2))
    check("ch05_main_ex5_9_to_5_17",
          abs(abs((-2 + 2j) * (-2 + 2j + 2) / (-2 + 2j + 4)) - 2.0) < 1e-9
          and abs(d511[0] - (-26.189)) < 0.01 and abs(d511[1] - (-5.943)) < 0.01 and abs(d511[2] - 3.132) < 0.01
          and sp.simplify(cl517 - (s_c**3 + 14 * s_c**2 + 48 * s_c + 80)) == 0
          and abs(float(sp.Poly(den514, s_c).all_coeffs()[0]) - 124.95) < 1e-6,
          "Confirmed Main 5-11 d1=-26.19 (book dropped minus sign -> 26.2)")

    # 3. Missing Guide Ex 5-14 & 5-15 (captures Guide 5-14 2+K vs 2+2K typo)
    eq_g514 = sp.numer(sp.together(sp.diff(-(s_c + 1) * (s_c + 2) / (s_c - 2), s_c)))
    d_g514 = sorted([float(r.evalf()) for r in sp.solve(eq_g514, s_c)])
    check("ch05_guide_ex5_14_15",
          abs(d_g514[0] - (2 - 2 * math.sqrt(3))) < 1e-6
          and abs(d_g514[1] - (2 + 2 * math.sqrt(3))) < 1e-6
          and sp.simplify((s_c + 1) * (s_c + 2) - K4 * (s_c - 2) - (s_c**2 + (3 - K4) * s_c + 2 + 2 * K4)) == 0,
          "Confirmed Guide 5-14 char eq 2+2K (book typo 2+K), 0<K<3 & -1<K<0")

    # 4. Exercises A5-1 ~ A5-8
    Kr5, zr5 = sp.symbols("Kr5 zr5", real=True, positive=True)
    e81 = sp.expand(sp.I**2 * (sp.I + 10) * (sp.I + 20) + Kr5 * (sp.I + zr5))
    sol81 = sp.solve([sp.re(e81), sp.im(e81)], [Kr5, zr5])[0]
    p57_c = sp.Poly(sp.expand((1 + sp.Rational(1, 2) * s_c) * (1 + sp.Rational(1, 5) * s_c) * (1 + sp.Rational(1, 8) * s_c)**2 + K4), s_c).all_coeffs()
    k57_max = float(sp.solve(p57_c[1] * p57_c[2] * p57_c[3] - p57_c[0] * p57_c[3]**2 - p57_c[1]**2 * p57_c[4], K4)[0])
    d56_max = max(float(r.evalf()) for r in sp.solve(sp.diff(s_c * (s_c + 100) * (s_c + 50), s_c), s_c))
    K56_cd = abs(d56_max * (0.01 * d56_max + 1) * (0.02 * d56_max + 1))
    check("ch05_ex_a5_1_to_8",
          abs(abs((-1 + math.sqrt(3) * 1j + 1) * (-1 + math.sqrt(3) * 1j + 2) * (-1 + math.sqrt(3) * 1j + 4)) - 12.0) < 1e-9
          and abs(K56_cd - 9.6225) < 0.01 and abs(k57_max - 5.5907) < 0.01
          and sol81 == (30, sp.Rational(199, 30)),
          f"A5-6 K_cd={K56_cd:.3f}, A5-7 K0_max={k57_max:.3f}, A5-8(1) K*=30, z=199/30")

    # 5. Exercises A5-9 ~ A5-15 (captures A5-11 Appendix 'K>0无解' major erratum!)
    k511_disc = [float(r.evalf()) for r in sp.solve((sp.Rational(5, 2) + 2 * K4)**2 - 4 * (1 + K4) * (1 + 5 * K4), K4) if float(r.evalf()) > 0][0]
    k59_rep = sorted([float(r.evalf()) for r in sp.solve((2 - K4)**2 - 4 * K4, K4)])
    check("ch05_ex_a5_9_to_15",
          abs(k59_rep[0] - 0.5359) < 0.001 and abs(k59_rep[1] - 7.4641) < 0.001
          and _rc4((1 + 10) * s_c**2 + (sp.Rational(5, 2) + 20) * s_c + (1 + 50)) == (0, 0, 2)
          and abs(k511_disc - 0.1387) < 0.001
          and sp.Rational(2, 27) == sp.Rational(4, 54)
          and abs(3 / math.sqrt(10) - 0.9487) < 0.001,
          "Confirmed A5-11 K>0 ALWAYS stable (Appendix P.250 erratum 'K>0无解')")

    # 6. Exercises A5-16 ~ A5-22
    k517_max = float(sp.solve(7 * 10 * (2 * K4) - (2 * K4)**2 - 49 * K4, K4)[1])
    cl521 = sp.expand((s_c + 4) * (s_c + 2)**2 + 32)
    check("ch05_ex_a5_16_to_22",
          abs(k517_max - 22.75) < 1e-9
          and sp.factor(cl521) == (s_c + 6) * (s_c**2 + 2 * s_c + 8)
          and abs(math.exp(-math.pi / math.sqrt(7)) * 100 - 30.50) < 0.1
          and sp.Rational(2, 5) == sp.Rational(4, 10))

    # 7. Exercises A5-25 ~ A5-28 (Lead, Lag, Lead-Lag compensator design)
    sd26 = -2 + 2 * math.sqrt(3) * 1j
    val26 = 4.728 * (sd26 + 2.93) / (sd26 + 5.46) * 4 / (sd26 * (sd26 + 2))
    sd28 = -2.5 + 2.5 * math.sqrt(3) * 1j
    val28 = 6.25 * (sd28 + 0.5) * (sd28 + 0.1) / ((sd28 + 5) * (sd28 + 0.01)) * 4 / (sd28 * (sd28 + 0.5))
    check("ch05_ex_a5_25_to_28",
          abs(val26 - (-1.0)) < 0.01
          and abs(val28 - (-1.0)) < 0.02
          and abs(6.25 * 0.5 * 0.1 / (5 * 0.01) * 4 / 0.5 - 50.0) < 1e-9)

    # 8. Self-test B5-12 ~ B5-20
    k516_rep = sorted([float(r.evalf()) for r in sp.solve((4 - 2 * K4)**2 - 16 * K4, K4)])
    check("ch05_selftest_b5_12_to_20",
          abs(abs((-1 + 0j) * ((-1)**2 + 4 * (-1) + 8) / (-1 + 5)) - 1.25) < 1e-9
          and abs(abs((-1.5 + 2j) * ((-1.5 + 2j)**2 + 4 * (-1.5 + 2j) + 8) / (-1.5 + 2j + 5)) - 1.25) < 1e-9
          and abs(k516_rep[0] - 0.5359) < 0.001 and abs(k516_rep[1] - 7.4641) < 0.001
          and sp.Rational(102, 40) == sp.Rational(51, 20)
          and sp.factor(s_c**3 + 6 * s_c**2 + 8 * s_c + 15) == (s_c + 5) * (s_c**2 + s_c + 3)
          and abs(math.exp(-math.pi * 0.5 / math.sqrt(2.75)) * 100 - 38.78) < 0.1)

    # --- Ch06 Full Textbook Edition (Main Ex 6-1..6-16, A6-1..A6-26, B6-21..B6-30) ---
    # 1. Main Ex 6-7, 6-13, 6-14
    Mr67 = 10**(8 / 20)
    z67 = math.sqrt((1 - math.sqrt(1 - 1 / Mr67**2)) / 2)
    T613 = math.sqrt(99) / (2 * (math.pi - math.atan(math.sqrt(99))))
    wc614_4 = math.sqrt(4**(2 / 3) - 1)
    pm614_4 = 180 - 3 * math.degrees(math.atan(wc614_4))
    wc614_10 = math.sqrt(10**(2 / 3) - 1)
    pm614_10 = 180 - 3 * math.degrees(math.atan(wc614_10))
    check("ch06_main_ex6_7_to_6_14",
          abs(z67 - 0.2033) < 0.001
          and abs(T613 - 2.977) < 0.01
          and abs(pm614_4 - 27.14) < 0.1 and abs(pm614_10 - (-7.03)) < 0.1)

    # 2. Exercises A6-1 ~ A6-9
    G61 = sp.simplify(s_c * (1 / s_c - sp.Rational(9, 5) / (s_c + 4) + sp.Rational(4, 5) / (s_c + 9)))
    wn63 = math.sqrt(2 + math.sqrt(2))
    z63 = (wn63**2 - 1) / (2 * wn63)
    check("ch06_ex_a6_1_to_9",
          sp.simplify(G61 - 36 / (s_c**2 + 13 * s_c + 36)) == 0
          and abs(abs((1j + 1) / (1j + 2)) - 0.6325) < 0.001
          and abs(abs((2j + 1) / (2j + 2)) - 0.7906) < 0.001
          and abs(wn63 - 1.8478) < 0.001 and abs(z63 - 0.6533) < 0.001
          and abs(10 * math.sqrt(1 + 0.05**2) / math.sqrt(1 + 20**2) - 0.5) < 1e-9)

    # 3. Exercises A6-11 ~ A6-16 (captures A6-11 exact vs asym & A6-12(c) -1.10dB erratum!)
    Ga12 = 8.9125 * (1 + 4j / 6) / (4j * (1 + 4j) * (1 + 4j / 15))
    Gb12 = 20 * (1 + 2j) / (4j * (1 + 20j) * (1 + 0.5j))
    Gc12 = 0.4 * (1 + 40j) / ((1 + 2j) * (1 + 4j / 32)**2)
    diff12_a = 20 * math.log10(abs(Ga12)) - (-40 * math.log10(4 / 3))
    diff12_b = 20 * math.log10(abs(Gb12)) - (-20 * math.log10(2))
    diff12_c = 20 * math.log10(abs(Gc12)) - 20 * math.log10(8)
    w4_14 = 100 * 10**(-5 / 60)
    w3_14 = w4_14 * 10**(-15 / 40)
    w2_14 = w3_14 * 10**(-20 / 20)
    check("ch06_ex_a6_11_to_16",
          abs(diff12_a - 0.951) < 0.01 and abs(diff12_b - (-0.011)) < 0.01
          and abs(diff12_c - (-1.101)) < 0.01
          and abs(10**(30 / 20) - 31.62) < 0.01
          and abs(w2_14 - 3.481) < 0.01 and abs(w4_14 - 82.54) < 0.02,
          "Confirmed A6-12(c) w=4 diff=-1.10dB (Appendix typo '无偏差')")

    # 4. Exercises A6-18 ~ A6-22 (captures A6-20(2) exact K1=25.93, K2=241.0 vs Appendix 20)
    G620_fn = lambda kv, wv: kv * (1 + 0.2j * wv) / ((1j * wv)**2 * (1 + 0.02j * wv))
    w20_sols = sorted([float(r) for r in sp.solve(sp.Rational(18, 100) * s_c - (1 + sp.Rational(4, 1000) * s_c**2), s_c)])
    K20_sols = [1 / abs(G620_fn(1, wv)) for wv in w20_sols]
    wn21 = math.sqrt(20)
    z21 = 5 / (2 * wn21)
    Mr21 = 1 / (2 * z21 * math.sqrt(1 - z21**2))
    wr21 = wn21 * math.sqrt(1 - 2 * z21**2)
    check("ch06_ex_a6_18_to_22",
          _rc4(sp.Rational(1, 200) * s_c**5 + sp.Rational(3, 20) * s_c**4 + s_c**3 + 5 * s_c**2 + 15 * s_c + 10) == (0, 0, 5)
          and abs(K20_sols[0] - 25.93) < 0.05 and abs(K20_sols[1] - 241.0) < 0.5
          and abs(Mr21 - 1.079) < 0.005 and abs(wr21 - 2.739) < 0.005
          and abs(2**(-0.25) - 0.8409) < 0.001,
          "Confirmed A6-20(2) exact K=25.93/241.0 (Appendix approx 20)")

    # 5. Exercises A6-23 ~ A6-26 (captures A6-23(2) MAJOR GM erratum: true h=5.79/15.25dB vs book 0.82!)
    w25_c, pm25_c = pm_at(lambda z_v: 40 / (z_v * (2 + z_v)) * (1 + 0.226 * z_v) / (1 + 0.0542 * z_v), 8.88)
    w26_c, pm26_c = pm_at(lambda z_v: 5 / (z_v * (1 + z_v) * (1 + 0.5 * z_v)) * (1 + 10 * z_v) / (1 + 100 * z_v), 0.454)
    check("ch06_ex_a6_23_to_26",
          abs(48 / 8.296 - 5.786) < 0.01
          and abs(2 * math.pi / (3 * math.sqrt(3)) - 1.209) < 0.005
          and pm25_c >= 50.0 and pm26_c >= 40.0,
          "Confirmed A6-23(2) true GM=5.79 (15.25dB), Appendix 0.82 = 1-1/h erratum!")

    # 6. Self-test B6-21 ~ B6-30 (also captures B6-22(2) Appendix denom (2+w^2) vs true (4+w^2)!)
    w6_sym = sp.Symbol("w6_sym", real=True, positive=True)
    Gb22 = sp.together(s_c * (1 / s_c + 1 / (s_c + 1) + 1 / (s_c + 2)))
    Gb22_jw = sp.expand(sp.re(Gb22.subs(s_c, sp.I * w6_sym))), sp.expand(sp.im(Gb22.subs(s_c, sp.I * w6_sym)))
    check("ch06_selftest_b6_21_to_30",
          abs(abs(3 / (4j + 4)) - 0.5303) < 0.001
          and sp.simplify(Gb22 - (3 * s_c**2 + 6 * s_c + 2) / ((s_c + 1) * (s_c + 2))) == 0
          and sp.simplify(Gb22_jw[0] - (4 + 10 * w6_sym**2 + 3 * w6_sym**4) / ((1 + w6_sym**2) * (4 + w6_sym**2))) == 0
          and sp.simplify(Gb22_jw[1] - (6 + 3 * w6_sym**2) * w6_sym / ((1 + w6_sym**2) * (4 + w6_sym**2))) == 0
          and abs(10**(-10 / 20) - 0.3162) < 0.001,
          "Confirmed B6-22(2) true denom (1+w^2)(4+w^2) (Appendix P.260 typo (2+w^2)!)")

    # --- Ch07 Full Textbook Edition (Main Ex 7-1..7-29, A7-1..A7-31, B7-9..B7-22) ---
    z7, w7, K7 = sp.symbols("z7 w7 K7")
    def _bw7(poly_z):
        cs = sp.Poly(sp.expand(poly_z), z7).all_coeffs()
        dg = len(cs) - 1
        return sp.Poly(sp.expand(sum(c * (w7 + 1)**(dg - i) * (w7 - 1)**i for i, c in enumerate(cs))), w7)

    # 1. Main Ex 7-12, 7-17, 7-18, 7-20, 7-23, 7-25 (major erratum!), 7-28 (w-sub erratum!)
    e07_7 = math.exp(-0.7)
    r725 = [complex(r.evalf()) for r in sp.nroots(sp.Poly(z7**2 - 2 * e07_7 * z7 + e07_7, z7))]
    A718_m = sp.Matrix([[0, 1], [-3, -5]])
    G718_m = sp.simplify((sp.Matrix([[1, 0]]) * (z7 * sp.eye(2) - A718_m).inv() * sp.Matrix([1, -3]))[0])
    G728_1p = sp.simplify((sp.Rational(253, 100) * (1 + w7) / (1 - w7)) / (((1 + w7) / (1 - w7) - 1) * ((1 + w7) / (1 - w7) - sp.Rational(368, 1000))))
    check("ch07_main_ex7_12_to_7_28",
          abs((-10 * 0.8**2 + 0.4**2 + 10) - (1.2 * 2.4 - 0.32 * 1 + 1.2)) < 1e-9
          and sp.simplify(G718_m - (z7 + 2) / (z7**2 + 5 * z7 + 3)) == 0
          and all(abs(r) < 0.71 for r in r725)
          and sp.simplify(G728_1p - 1265 * (1 + w7) * (1 - w7) / (8 * w7 * (79 + 171 * w7))) == 0,
          "Confirmed Main 7-25 TRUE |z|=0.705 STABLE (book forgot /10!) & Main 7-28 uses z=(1+w)/(1-w)")

    # 2. Exercises A7-1 ~ A7-9
    F74_1 = z7 * (z7 + sp.Rational(1, 2)) / ((z7 - 1) * (z7**2 - sp.Rational(1, 2) * z7 + sp.Rational(5, 16)))
    C77_4 = (z7**3 + 7 * z7**2 + 17 * z7) / ((z7 + 1) * (z7 + 2) * (z7 + 3))
    check("ch07_ex_a7_1_to_9",
          sp.limit((z7 - 1) * F74_1, z7, 1) == sp.Rational(24, 13)
          and sp.apart(C77_4 / z7, z7) == sp.Rational(11, 2) / (z7 + 1) - 7 / (z7 + 2) + sp.Rational(5, 2) / (z7 + 3)
          and abs((1 + 1) / (1 - 1.4 + 0.48) - 25.0) < 1e-9)

    # 3. Exercises A7-13 ~ A7-19 (discrete state space, discretization, step response)
    A714_m = sp.Matrix([[0, 1], [-sp.Rational(4, 25), -1]])
    G714_m = sp.simplify(sp.Matrix([[1, 1], [0, 1]]) * (z7 * sp.eye(2) - A714_m).inv() * sp.Matrix([0, 1]))
    G717_m = sp.simplify((sp.Matrix([[1, 0]]) * (z7 * sp.eye(2) - sp.Matrix([[1, 1], [0, 1]])).inv() * sp.Matrix([sp.Rational(1, 2), 1]))[0])
    check("ch07_ex_a7_13_to_19",
          sp.simplify(G714_m[0] - (z7 + 1) / ((z7 + sp.Rational(1, 5)) * (z7 + sp.Rational(4, 5)))) == 0
          and sp.simplify(G714_m[1] - z7 / ((z7 + sp.Rational(1, 5)) * (z7 + sp.Rational(4, 5)))) == 0
          and sp.simplify(G717_m - (z7 + 1) / (2 * (z7 - 1)**2)) == 0)

    # 4. Exercises A7-21 ~ A7-26 (stability boundaries, A7-23/24/25/26)
    e1_7 = math.exp(-1)
    bw21_7 = _bw7(z7**2 + (K7 - 2) * z7 + (1 - sp.Rational(1, 2) * K7))
    bw22_7 = _bw7((z7 - 1) * (z7 - e1_7) + K7 * (e1_7 * z7 + 1 - 2 * e1_7))
    k22_7 = min(float(sp.solve(c, K7)[0]) for c in bw22_7.all_coeffs()[1:])
    M25_5 = sp.Matrix([[math.exp(-5), math.exp(-5) - 1], [1 - math.exp(-5), 2 - 5 - math.exp(-5)]])
    check("ch07_ex_a7_21_to_26",
          sp.solve(bw21_7.all_coeffs()[2], K7)[0] == sp.Rational(8, 3)
          and abs(k22_7 - 2.3922) < 0.005
          and abs(math.log(4 / 3) - 0.2877) < 0.001
          and any(abs(complex(ev.evalf())) > 2.5 for ev in M25_5.eigenvals().keys())
          and abs(e1_7 / (1 - e1_7)**2 - 0.9207) < 0.002
          and abs(1 / (1 - e1_7)**2 - 2.5026) < 0.005)

    # 5. Exercises A7-27 ~ A7-31 (A7-28 actuating error=0.1, A7-29 Kmax=2.472, A7-30/31 deadbeat)
    H28_7 = sp.simplify((1 - 1 / z7) * (sp.Rational(1, 5) * z7 * (z7 + 1) / (z7 - 1)**3 + z7 / (z7 - 1)**2))
    Pe28_7 = sp.simplify(1 / (1 + H28_7))
    e28_para = sp.limit((z7 - 1) * Pe28_7 * (sp.Rational(1, 25) * z7 * (z7 + 1) / (2 * (z7 - 1)**3)), z7, 1)
    c29_7 = _bw7(z7**3 - z7**2 + sp.Rational(1, 4) * K7).all_coeffs()
    k29_7 = [float(r.evalf()) for r in sp.solve(c29_7[1] * c29_7[2] - c29_7[0] * c29_7[3], K7) if float(r.evalf()) > 0][0]
    check("ch07_ex_a7_27_to_31",
          e28_para == sp.Rational(1, 10)
          and abs(k29_7 - 2.4721) < 0.005
          and abs(1 / (1 - e1_7) - 1.5820) < 0.002)

    # 6. Self-tests B7-9 ~ B7-16 (captures B7-15 MAJOR erratum: true Kmax=2.181 vs Appendix 5.79!)
    e05_7 = math.exp(-0.5)
    bw_b15 = _bw7((z7 - 1) * (z7 - e05_7) + K7 * ((2 * e05_7 - 1) * z7 + (2 - 3 * e05_7)))
    k_b15 = min(float(sp.solve(c, K7)[0]) for c in bw_b15.all_coeffs()[1:])
    e5_7 = math.exp(-5.0)
    Gb13_7 = z7 / (5 * (z7 - 1)**2) - z7 / (25 * (z7 - 1)) + z7 / (25 * (z7 - e5_7))
    Pb13_7 = sp.simplify(Gb13_7 / (1 + Gb13_7))
    check("ch07_selftest_b7_9_to_16",
          any(abs(complex(r.evalf())) > 1.01 for r in sp.nroots(sp.denom(Pb13_7)))
          and abs(2 * (1 + e1_7) / (1 - e1_7) - 4.3279) < 0.005
          and abs(k_b15 - 2.1810) < 0.002,
          "Confirmed B7-15(2) TRUE Kmax=2.181 (Appendix P.261 erratum 5.79 -> |z|=1.285 unstable!)")

    # 7. Self-tests B7-17 ~ B7-22 (captures B7-17 Kp=Kv=inf vs '8' & B7-21 unstable zero absorption!)
    Gb21_7 = sp.together((1 - 1 / z7) * (5 * z7 / (z7 - 1)**2 - 11 * z7 / (z7 - 1) + (100 / 9) * z7 / (z7 - e05_7) - (1 / 9) * z7 / (z7 - e5_7)))
    n21_7, d21_7 = sp.fraction(Gb21_7)
    r21_7 = sorted([float(r.evalf()) for r in sp.nroots(sp.Poly(n21_7, z7))])
    check("ch07_selftest_b7_17_to_22",
          abs(math.log(3) / 2 - 0.5493) < 0.001
          and abs(r21_7[0] - (-1.5159)) < 0.002 and abs(r21_7[1] - (-0.0517)) < 0.002,
          "Confirmed B7-21 unstable zero z=-1.516 absorbed into Phi(z) -> 2-beat deadbeat!")

    # --- Ch08 Full Textbook Edition (Main Ex 8-1..8-39, Guide Ex 8-19, A8-1..A8-29, B8-17..B8-27) ---
    # 1. Main Ex 8-19, 8-26, 8-27, 8-29 (captures P.353 Apcl[2,2]=+191/36 erratum!)
    Ap829 = sp.Matrix([[1, 6, -3], [-1, -1, 1], [-2, 2, 0]])
    bp829 = sp.Matrix([[1], [1], [1]])
    K829 = sp.Matrix([[175, 232, -191]]) / 36
    cl829 = sp.expand((s_c * sp.eye(3) - (Ap829 - bp829 * K829)).det())
    check("ch08_main_ex8_19_to_8_29",
          sp.simplify(cl829 - (s_c + 4) * ((s_c + 1)**2 + 1)) == 0
          and 36 * (Ap829 - bp829 * K829)[2, 2] == 191,
          "Confirmed Main 8-29 P.353 Apcl[2,2]=+191/36 (book typo -191/36!)")

    # 2. Main Ex 8-33, 8-34 (MAJOR erratum A11-hA21!), 8-35, 8-36, Guide Ex 8-19
    A834_11 = sp.Matrix([[0, 0], [1, 0]])
    A834_21 = sp.Matrix([[1, 1]])
    h834 = sp.Matrix([[12], [-5]])
    Az836 = sp.Matrix([[1, 0, 0], [0, 2, -2], [-1, 1, 0]])
    Bz836 = sp.Matrix([[1], [0], [1]])
    u836 = sp.Matrix.hstack(Az836**2 * Bz836, Az836 * Bz836, Bz836).inv() * (-Az836**3 * sp.Matrix([[2], [1], [0]]))
    G819_g = sp.Matrix([[-2, 2, -1], [0, -2, 0], [1, -4, 0]])
    H819_g = sp.Matrix([[0, 0], [0, 1], [1, 0]])
    check("ch08_main_ex8_33_to_8_39",
          sp.expand((s_c * sp.eye(2) - (A834_11 - h834 * A834_21)).det() - (s_c + 3) * (s_c + 4)) == 0
          and (A834_11 - h834 * A834_21) == sp.Matrix([[-12, -12], [6, 5]])
          and u836 == sp.Matrix([[-5], [11], [-8]])
          and (-G819_g.inv() * H819_g) == sp.Matrix([[-1, 2], [0, sp.Rational(1, 2)], [2, -3]]),
          "Confirmed Main 8-34 TRUE A11-hA21=[[-12,-12],[6,5]] (book P.369-370 swapped h1/h2 -> unstable [[5,5],[-11,-12]]!)")

    # 3. Exercises A8-1 ~ A8-10 (controllability/observability/discretization)
    A89_m = sp.Matrix([[0, sp.pi], [-sp.pi, 0]])
    b89_m = sp.Matrix([[0], [1]])
    G89_1 = sp.simplify(sp.exp(A89_m * 1))
    H89_1 = sp.simplify(sp.integrate(sp.exp(A89_m * t_r) * b89_m, (t_r, 0, 1)))
    G89_2 = sp.simplify(sp.exp(A89_m * 2))
    H89_2 = sp.simplify(sp.integrate(sp.exp(A89_m * t_r) * b89_m, (t_r, 0, 2)))
    check("ch08_ex_a8_1_to_10",
          sp.factor(s_c**3 + 7 * s_c**2 + 14 * s_c + 8) == (s_c + 1) * (s_c + 2) * (s_c + 4)
          and ctrl_mat(G89_1, H89_1).rank() == 1 and ctrl_mat(G89_2, H89_2).rank() == 0
          and H89_1 == sp.Matrix([[2 / sp.pi], [0]]) and H89_2 == sp.zeros(2, 1))

    # 4. Exercises A8-11 ~ A8-19 (canonical/decomp/pole placement; captures A8-17/18 sign convention!)
    A815_m = sp.Matrix([[-2, -3], [4, -9]])
    b815_m = sp.Matrix([[3], [1]])
    K815_m = sp.Matrix([[sp.Rational(-101, 18), sp.Rational(47, 6)]])
    A818_m = sp.Matrix([[0, 1, 0], [0, -1, 1], [0, -1, -10]])
    b818_m = sp.Matrix([[0], [0], [10]])
    K818_m = sp.Matrix([[4, sp.Rational(6, 5), sp.Rational(1, 10)]])
    A819_m = sp.Matrix([[-1, 0, 0], [0, -2, -2], [-1, 0, -3]])
    b819_m = sp.Matrix([[1], [1], [1]])
    K819_m = sp.Matrix([[7, 0, -1]])
    check("ch08_ex_a8_11_to_19",
          sp.expand((s_c * sp.eye(2) - (A815_m - b815_m * K815_m)).det() - ((s_c + 1)**2 + 4)) == 0
          and sp.expand((s_c * sp.eye(3) - (A818_m - b818_m * K818_m)).det() - (s_c + 10) * ((s_c + 1)**2 + 3)) == 0
          and sp.expand((s_c * sp.eye(3) - (A819_m - b819_m * K819_m)).det() - (s_c + 2) * (s_c + 4) * (s_c + 6)) == 0)

    # 5. Exercises A8-20 ~ A8-25 (design, LQR A8-22, observers A8-23..25)
    A822_m = sp.Matrix([[0, 0], [1, 0]])
    B822_m = sp.Matrix([[1], [0]])
    P822_m = sp.Matrix([[1, 1], [1, 2]])
    are822 = -P822_m * A822_m - A822_m.T * P822_m + P822_m * B822_m * sp.Matrix([[2]]) * B822_m.T * P822_m - sp.Matrix([[0, 0], [0, 2]])
    A825_2 = sp.Matrix([[1, 0, 1], [1, 0, 0], [0, 1, 0]])
    C825_2 = sp.Matrix([[1, 0, 0], [0, 1, 0]])
    H825_2 = sp.Matrix([[0, 37], [0, 10], [0, 28]])
    check("ch08_ex_a8_20_to_25",
          are822 == sp.zeros(2, 2)
          and sp.expand((s_c * sp.eye(3) - (A825_2 - H825_2 * C825_2)).det() - (s_c + 3)**3) == 0)

    # 6. Exercises A8-26 ~ A8-29 (captures A8-27 H_P=[11,6,-5] & A8-28 K=[2.2,-1.6] MAJOR errata!)
    A826_m = sp.Matrix([[1, 0, 0], [0, 2, 1], [0, 0, 2]])
    c826_m = sp.Matrix([[1, 1, 0]])
    H826_m = sp.Matrix([[120], [-103], [210]])
    A827_m = sp.Matrix([[1, 2, 0], [3, -1, 1], [0, 2, 0]])
    c827_m = sp.Matrix([[1, 1, 1]])
    H827_m = sp.Matrix([[11], [6], [-5]])
    A828_m = sp.Matrix([[-2, 1], [1, -3]])
    b828_m = sp.Matrix([[1], [2]])
    K828_m = sp.Matrix([[sp.Rational(11, 5), sp.Rational(-8, 5)]])
    A829_aug = sp.Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -2, -2]])
    b829_aug = sp.Matrix([[0], [0], [1], [2]])
    K829_im = sp.Matrix([[50, sp.Rational(95, 2), sp.Rational(45, 2), sp.Rational(-5, 4)]])
    check("ch08_ex_a8_26_to_29",
          sp.expand((s_c * sp.eye(3) - (A826_m - H826_m * c826_m)).det() - (s_c + 3) * (s_c + 4) * (s_c + 5)) == 0
          and sp.expand((s_c * sp.eye(3) - (A827_m - H827_m * c827_m)).det() - (s_c + 4)**3) == 0
          and sp.expand((s_c * sp.eye(2) - (A828_m - b828_m * K828_m)).det() - ((s_c + 2)**2 + 4)) == 0
          and sp.expand((s_c * sp.eye(4) - (A829_aug - b829_aug * K829_im)).det() - ((s_c + 1)**2 + 1) * (s_c + 10)**2) == 0,
          "Confirmed A8-27 TRUE H=[11,6,-5]^T (Appendix [36.2,31.8,16.4]^T unstable!) & A8-28 TRUE K=[2.2,-1.6] (Appendix [2.5,-0.5] wrong!)")

    # 7. Self-tests B8-17 ~ B8-27
    A_b27 = sp.Matrix([[0, 1], [0, -5]])
    b_b27 = sp.Matrix([[0], [1]])
    c_b27 = sp.Matrix([[100, 0]])
    K_b27 = sp.Matrix([[100, sp.Rational(914, 100)]])
    H_b27 = sp.Matrix([[sp.Rational(15, 100)], [sp.Rational(25, 100)]])
    r_b26 = sp.Symbol("r_b26", positive=True)
    P_b26 = sp.Matrix([[sp.sqrt(2) * r_b26**sp.Rational(1, 4), sp.sqrt(r_b26)], [sp.sqrt(r_b26), sp.sqrt(2) * r_b26**sp.Rational(3, 4)]])
    are_b26 = sp.simplify(-P_b26 * sp.Matrix([[0, 1], [0, 0]]) - sp.Matrix([[0, 0], [1, 0]]) * P_b26
                          + P_b26 * sp.Matrix([[0], [1]]) * (1 / r_b26) * sp.Matrix([[0, 1]]) * P_b26 - sp.Matrix([[1, 0], [0, 0]]))
    check("ch08_selftest_b8_17_to_27",
          sp.Matrix([[1, 2], [0, 1]])**100 == sp.Matrix([[1, 200], [0, 1]])
          and ctrl_mat(sp.Matrix([[1, 2, -1], [0, 1, 0], [1, 0, 3]]), sp.Matrix([[1, 0], [0, 1], [0, 0]])).rank() == 3
          and ctrl_mat(sp.Matrix([[1, 2, -1], [0, 1, 0], [1, -4, 3]]), sp.Matrix([[0], [0], [1]])).rank() == 2
          and are_b26 == sp.zeros(2, 2)
          and sp.expand((s_c * sp.eye(2) - (A_b27 - b_b27 * K_b27)).det() - (s_c**2 + sp.Rational(1414, 100) * s_c + 100)) == 0
          and sp.expand((s_c * sp.eye(2) - (A_b27 - H_b27 * c_b27)).det() - (s_c + 10)**2) == 0)

    # --- Ch09 Full Textbook Edition (Main Ex 9-1..9-15, Guide Ex 9-1..9-16, A9-4..A9-19, B9-7..B9-16) ---
    # 1. Main Ex 9-4, 9-5, 9-6 (MAJOR erratum on Main 9-6: exact w=14.524, A=0.657 vs book 7.24/1.24!)
    def _g96(wv):
        z = 15.0 * (math.cos(0.1 * wv) - 1j * math.sin(0.1 * wv)) / (1.0 + 2j * wv)
        return z
    lo96, hi96 = 14.0, 16.0
    tgt96 = -math.pi * 0.1 / 4.0
    for _ in range(50):
        mid96 = (lo96 + hi96) / 2.0
        if _g96(mid96).imag < tgt96:
            lo96 = mid96
        else:
            hi96 = mid96
    w96_ex = (lo96 + hi96) / 2.0
    g96_ex = _g96(w96_ex)
    A96_ex = math.sqrt((4.0 * abs(g96_ex.real) / math.pi)**2 + 0.01)
    check("ch09_main_ex9_1_to_9_6",
          abs(math.sqrt(50) - 7.0711) < 0.001
          and abs(-1.0 / 22.5 - (-0.04444)) < 0.0001
          and abs(w96_ex - 14.5243) < 0.002 and abs(A96_ex - 0.6571) < 0.002
          and abs(g96_ex.real - (-0.5101)) < 0.002,
          "Confirmed Main 9-6 EXACT w=14.524 rad/min, A=0.657 C at P(-0.510,-0.0785) (book P.408-409 typo w=7.24, A=1.24 -> 2x off!)")

    # 2. Guide Ex 9-4 (parabolic sign typos!) & Guide Ex 9-10 (PR=2 RHP poles!)
    r910 = [complex(r.evalf()) for r in sp.nroots(sp.Poly(sp.expand(s_c * (sp.Rational(1, 10) * s_c + 1) * (s_c + 1) + 20), s_c))]
    pr910 = sum(1 for r in r910 if r.real > 0)
    w910_sq = [float(r.evalf()) for r in sp.solve(21 - sp.Rational(101, 100) * s_c - sp.Rational(1, 100) * s_c**2, s_c) if float(r.evalf()) > 0][0]
    w910_v = math.sqrt(w910_sq)
    g910_v = ((0.1j * w910_v + 1) * (1j * w910_v + 1)) / (1j * w910_v * (0.1j * w910_v + 1) * (1j * w910_v + 1) + 20)
    check("ch09_guide_ex9_4_and_9_10",
          (4 + 6 + 2) == 12
          and (-( (-2)**2 ) + 6) == 2 and (( (-2)**2 ) - 2) == 2 and ((1**2) - 2) == -1
          and pr910 == 2 and g910_v.real < -1.42,
          "Confirmed Guide 9-4 parabolic signs e=-e'^2+6 & e=+e'^2-2 (P.229 flipped!) and Guide 9-10 PR=2 RHP poles")

    # 3. Exercises A9-4 ~ A9-13 (phase plane singular points, describing function A9-10..A9-13)
    w_s = sp.Symbol("w_s", positive=True)
    g910_e = sp.simplify(((1 - sp.I * w_s) / (sp.I * w_s * (1 + sp.I * w_s))).subs(w_s, 1))
    g911_e = sp.simplify((10 / (sp.I * w_s * (1 + sp.I * w_s) * (2 + sp.I * w_s))).subs(w_s, sp.sqrt(2)))
    check("ch09_ex_a9_4_to_13",
          g910_e == -1 and abs(4 / math.pi - 1.2732) < 0.001 and abs(2 * math.sqrt(2) / math.pi - 0.9003) < 0.001
          and g911_e == sp.Rational(-5, 3) and abs(20 / (3 * math.pi) - 2.1221) < 0.001
          and abs(3 * math.pi / 4 - 2.3562) < 0.001)

    # 4. Exercises A9-14 ~ A9-19 (captures Guide P.238 A9-15(2) dropped coeff 3 & A9-19 discrete unstable!)
    a9, b9, c9 = sp.symbols("a9 b9 c9", real=True)
    P915_1 = sp.Matrix([[a9, 1, -1], [1, b9, -2], [-1, -2, c9]])
    P915_2_m = sp.Matrix([[1, 1, -1], [1, 3, 1], [-1, 1, a9]])
    P915_2_g = sp.Matrix([[1, 1, -1], [1, 1, 1], [-1, 1, a9]])
    A916_1 = sp.Matrix([[0, 1], [-2, -3]])
    P916_1 = sp.Matrix([[sp.Rational(5, 4), sp.Rational(1, 4)], [sp.Rational(1, 4), sp.Rational(1, 4)]])
    Az919 = sp.Matrix([[1, 3, 0], [-3, -2, -3], [1, 0, 0]])
    check("ch09_ex_a9_14_to_19",
          sp.expand(P915_1.det() - (a9 * b9 * c9 - 4 * a9 - b9 - c9 + 4)) == 0
          and P915_2_m[:2, :2].det() == 2 and sp.expand(P915_2_m.det() - (2 * a9 - 6)) == 0
          and P915_2_g[:2, :2].det() == 0
          and A916_1.T * P916_1 + P916_1 * A916_1 == -sp.eye(2)
          and any(abs(complex(r.evalf())) > 2.5 for r in Az919.eigenvals().keys()),
          "Confirmed A9-15(2) main 3*x2^2 -> a>3 vs Guide P.238 typo x2^2 -> Delta2=0 never PD!")

    # 5. Self-tests B9-7 ~ B9-16 (captures Appendix P.263 B9-15 MAJOR erratum: lambda=+1 strictly UNSTABLE!)
    c_b10 = [sp.simplify((1 / (sp.pi * A_amp)) * sp.integrate((A_amp * sp.sin(th))**n_p * sp.sin(th), (th, 0, 2 * sp.pi))) for n_p in (1, 3, 5, 7)]
    Az_b16 = sp.Matrix([[sp.Rational(4, 5), sp.Rational(-2, 5)], [sp.Rational(6, 5), sp.Rational(1, 5)]])
    P_b16 = sp.Matrix([[sp.Rational(4775, 792), sp.Rational(-1675, 1584)], [sp.Rational(-1675, 1584), sp.Rational(21125, 9504)]])
    check("ch09_selftest_b9_7_to_16",
          c_b10 == [1, sp.Rational(3, 4) * A_amp**2, sp.Rational(5, 8) * A_amp**4, sp.Rational(35, 64) * A_amp**6]
          and set(sp.Matrix([[0, 1], [2, -1]]).eigenvals().keys()) == {1, -2}
          and Az_b16.T * P_b16 * Az_b16 - P_b16 == -sp.eye(2) and P_b16[0, 0] > 0 and P_b16.det() > 0,
          "Confirmed B9-15 eigs={+1,-2} STRICTLY UNSTABLE (Appendix P.263 erroneously called it Lyapunov stable!)")

    # =====================================================================
    # Control Exam-01: 2010-2026 Overview + 12 Mother Problems Verification
    # =====================================================================
    # 1. Table E1-1 quick-calc constants + Mothers 1-6
    def _ov(z_val: float) -> float:
        return math.exp(-math.pi * z_val / math.sqrt(1 - z_val**2)) * 100.0

    ok_const = (
        abs(_ov(math.sqrt(2) / 2) - 4.321) < 0.02
        and abs(_ov(0.69) - 5.0) < 0.1
        and abs(_ov(0.6) - 9.48) < 0.05
        and abs(_ov(0.5) - 16.30) < 0.05
        and abs(20 * math.log10(2) - 6.0206) < 0.001
        and abs(20 * math.log10(math.sqrt(2)) - 3.0103) < 0.001
        and abs(20 * math.log10(5) - 13.9794) < 0.001
        and abs(math.log(0.05) - (-2.9957)) < 0.001
        and abs(math.log(0.02) - (-3.9120)) < 0.001
        and abs(math.exp(-1) - 0.3679) < 0.001
        and abs(math.exp(-2) - 0.1353) < 0.001
    )
    # Mother 1 (2026-T1): G(s)=0.5K/(s(s^2+2s+4)) -> 8<=K<16
    s_e, K_e, b_e, a_e, w_e, z_e = sp.symbols("s_e K_e b_e a_e w_e z_e", positive=True)
    m1_kv = sp.limit(s_e * sp.Rational(1, 2) * K_e / (s_e * (s_e**2 + 2 * s_e + 4)), s_e, 0)
    # Mother 2 (2021-T3): s^3+(40+b)s^2+40bs+4K at s=20j -> b=10, K=5000
    m2_expr = sp.expand((20 * sp.I)**3 + (40 + b_e) * (20 * sp.I)**2 + 40 * b_e * (20 * sp.I) + 4 * K_e)
    m2_sol = sp.solve([sp.re(m2_expr), sp.im(m2_expr)], [b_e, K_e])
    m2_roots = [complex(r.evalf()) for r in sp.roots(s_e**3 + 50 * s_e**2 + 400 * s_e + 5000, s_e).keys()]
    # Mother 3 (2023-T5): (s+1)(s+1.5)(s+2)+K with s=z-1 -> z^3+1.5z^2+0.5z+K, K_sup=0.75
    m3_poly = sp.expand((z_e) * (z_e + sp.Rational(1, 2)) * (z_e + 1) + sp.Rational(3, 4))
    m3_fac = sp.factor(m3_poly)
    # Mother 4 (2021-T4): (s+1)(s+2)(s+4)+8K at K=1.5 -> (s+5)(s^2+2s+4)
    m4_poly = sp.expand((s_e + 1) * (s_e + 2) * (s_e + 4) + 8 * sp.Rational(3, 2))
    # Mother 5 (2026-T3): p1,2=-2+-2j, z=-1 -> R=sqrt(5); K=2, zeta=3/sqrt(8+2a)<0.1 -> a>446
    m5_R = sp.sqrt((-2 + 1)**2 + 2**2)
    m5_a_bound = sp.solve(sp.Eq(3 / sp.sqrt(8 + 2 * a_e), sp.Rational(1, 10)), a_e)[0]
    # Mother 6 (2021-T6 & T7): gamma_max at w_m=1/sqrt(ab), K=1/(b*sqrt(ab)); pure delay tau_max=pi/4
    m6_gamma = sp.atan(b_e * w_e) - sp.atan(a_e * w_e)
    m6_dw = sp.simplify(sp.diff(m6_gamma, w_e).subs(w_e, 1 / sp.sqrt(a_e * b_e)))
    m6_K = sp.simplify((1 / (a_e * b_e)) * sp.sqrt((1 + a_e / b_e) / (1 + b_e / a_e)))
    m6_G0_1 = sp.simplify(sp.sqrt(2) / (sp.I * (1 + sp.I)))
    check("ctrl_exam01_formulas_and_mothers_1_6",
          ok_const
          and m1_kv == K_e / 8
          and m2_sol == {b_e: 10, K_e: 5000} and all(r.real < 0 for r in m2_roots) and any(abs(r.imag) > 1 for r in m2_roots)
          and sp.expand(m3_fac - (z_e + sp.Rational(3, 2)) * (z_e**2 + sp.Rational(1, 2))) == 0
          and sp.expand(m4_poly - (s_e + 5) * (s_e**2 + 2 * s_e + 4)) == 0
          and m5_R == sp.sqrt(5) and m5_a_bound == 446
          and m6_dw == 0 and sp.simplify(m6_K - 1 / (b_e * sp.sqrt(a_e * b_e))) == 0
          and sp.Abs(m6_G0_1) == 1 and sp.arg(m6_G0_1) == -3 * sp.pi / 4,
          "Verified Exam-01 Table E1-1 constants & Mothers 1-6 (2026-T1, 2021-T3, 2023-T5, 2021-T4, 2026-T3, 2021-T6/T7)")

    # 2. Mothers 7-12 (2025-T6, 2022-T4, 2023-T1, 2024/25-T7, 2023-T10=2025-T9, 2024-T9/T10)
    # Mother 7 (2025-T6 = 2018-T8): (z-1)(z-1.2) + 1.8z - 1.07 == (z-(0.2+0.3j))(z-(0.2-0.3j))
    z_m7 = sp.Symbol("z_m7")
    m7_cl = sp.expand((z_m7 - 1) * (z_m7 - sp.Rational(6, 5)) + sp.Rational(9, 5) * z_m7 - sp.Rational(107, 100))
    m7_des = sp.expand((z_m7 - (sp.Rational(1, 5) + sp.Rational(3, 10) * sp.I)) * (z_m7 - (sp.Rational(1, 5) - sp.Rational(3, 10) * sp.I)))
    # Mother 8 (2022-T4): A=[[0,1],[-4,0]], B=[0,1]^T -> G*H & det([H, G*H]) = -1/4*sin(2T)*(1-cos(2T)) = -sin^3(T)*cos(T)
    T_m8 = sp.Symbol("T_m8", positive=True)
    A_m8 = sp.Matrix([[0, 1], [-4, 0]])
    B_m8 = sp.Matrix([0, 1])
    G_m8 = sp.simplify(sp.exp(A_m8 * T_m8))
    H_m8 = sp.simplify(A_m8.inv() * (G_m8 - sp.eye(2)) * B_m8)
    GH_m8 = sp.simplify(G_m8 * H_m8)
    GH_m8_expected = sp.Matrix([sp.Rational(1, 4) * (sp.cos(2 * T_m8) - sp.cos(4 * T_m8)), sp.Rational(1, 2) * (sp.sin(4 * T_m8) - sp.sin(2 * T_m8))])
    det_m8 = sp.trigsimp(sp.Matrix.hstack(H_m8, GH_m8).det())
    det_m8_expected = -sp.Rational(1, 4) * sp.sin(2 * T_m8) * (1 - sp.cos(2 * T_m8))
    # Mother 9 (2023-T1): Phi(t) -> C*Phi(s)*B = [2/(s+2), 2/(s+2)]
    Phi_s_m9 = sp.Matrix([
        [2 / (s_e + 1) - 1 / (s_e + 2), 1 / (s_e + 1) - 1 / (s_e + 2)],
        [-2 / (s_e + 1) + 2 / (s_e + 2), -1 / (s_e + 1) + 2 / (s_e + 2)],
    ])
    B_m9 = sp.Matrix([[1, 1], [1, 1]])
    C_m9 = sp.Matrix([[1, 1]])
    G_m9 = sp.simplify(C_m9 * Phi_s_m9 * B_m9)
    # Mother 10 (PBH + Duality sanity on symbolic 2x2)
    a11, a12, a22, b1_s = sp.symbols("a11 a12 a22 b1_s")
    A_m10 = sp.Matrix([[a11, a12], [0, a22]])
    B_m10 = sp.Matrix([b1_s, 0])
    v_m10 = sp.Matrix([[0, 1]])
    ok_m10 = (v_m10 * A_m10 == a22 * v_m10) and (v_m10 * B_m10 == sp.Matrix([[0]]))
    # Mother 11 (2023-T10 = 2025-T9): 26pt state feedback + 2D reduced observer
    A_m11 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, -2, -3]])
    B_m11 = sp.Matrix([0, 0, 1])
    C_m11 = sp.Matrix([[1, 0, 0]])
    Qc_m11 = sp.Matrix.hstack(B_m11, A_m11 * B_m11, A_m11**2 * B_m11)
    Qo_m11 = sp.Matrix.vstack(C_m11, C_m11 * A_m11, C_m11 * A_m11**2)
    K_m11 = sp.Matrix([[3, 2, 1]])
    Acl_m11 = A_m11 - B_m11 * K_m11
    char_m11 = sp.expand((s_e * sp.eye(3) - Acl_m11).det())
    tf_m11 = sp.simplify((C_m11 * (s_e * sp.eye(3) - Acl_m11).inv() * B_m11)[0, 0])
    A12_m11 = sp.Matrix([[1, 0]])
    A22_m11 = sp.Matrix([[0, 1], [-2, -3]])
    L_m11 = sp.Matrix([7, 2])
    F_m11 = A22_m11 - L_m11 * A12_m11
    obs_char_m11 = sp.expand((s_e * sp.eye(2) - F_m11).det())
    FL_m11 = F_m11 * L_m11
    # Mother 12 (2024-T9 & T10): x=e^3 DF & Lyapunov V=x1^2+x2^2
    psi_m12, x1_m12, x2_m12, c_m12 = sp.symbols("psi_m12 x1_m12 x2_m12 c_m12", real=True)
    N_m12 = sp.simplify((1 / (sp.pi * A_amp)) * sp.integrate((A_amp * sp.sin(psi_m12))**3 * sp.sin(psi_m12), (psi_m12, 0, 2 * sp.pi)))
    G_m12 = sp.simplify(1 / (sp.I * sp.sqrt(2) * (1 + sp.I * sp.sqrt(2)) * (2 + sp.I * sp.sqrt(2))))
    Vdot_m12 = sp.factor(
        2 * x1_m12 * (x2_m12 + c_m12 * x1_m12 * (x1_m12**2 + x2_m12**2))
        + 2 * x2_m12 * (-x1_m12 + c_m12 * x2_m12 * (x1_m12**2 + x2_m12**2))
    )
    check("ctrl_exam01_mothers_7_12",
          m7_cl == m7_des
          and sp.simplify(GH_m8 - GH_m8_expected) == sp.zeros(2, 1)
          and sp.trigsimp(det_m8 - det_m8_expected) == 0 and sp.trigsimp(det_m8 + sp.sin(T_m8)**3 * sp.cos(T_m8)) == 0
          and G_m9 == sp.Matrix([[2 / (s_e + 2), 2 / (s_e + 2)]])
          and ok_m10
          and Qc_m11.det() == -1 and Qo_m11.det() == 1
          and char_m11 == sp.expand((s_e + 3) * (s_e**2 + s_e + 1))
          and sp.simplify(tf_m11 - 1 / (s_e**3 + 4 * s_e**2 + 4 * s_e + 3)) == 0
          and obs_char_m11 == sp.expand((s_e + 5)**2) and FL_m11 == sp.Matrix([-47, -34])
          and N_m12 == sp.Rational(3, 4) * A_amp**2 and G_m12 == sp.Rational(-1, 6)
          and (-1 / N_m12.subs(A_amp, 2 * sp.sqrt(2))) == sp.Rational(-1, 6)
          and Vdot_m12 == 2 * c_m12 * (x1_m12**2 + x2_m12**2)**2,
          "Verified Mothers 7-12 (2025-T6, 2022-T4 exact -1/4*sin(2T)*(1-cos(2T)), 2023-T1, PBH, 2023-T10=2025-T9 FL=[-47,-34], 2024-T9/T10)")

    # =====================================================================
    # Control Exam-02: 2020-2022 Full 29 Real-Exam Problems + 11 Errata Audits
    # =====================================================================
    # 1. 2020 (T1-T10)
    s20, z20, K20, al20, be20, ga20 = sp.symbols("s20 z20 K20 al20 be20 ga20", real=True)
    # T1(3) strictly unstable (84<100); T1(4) factor (s+1)(s^2-3s+1)
    r_20_1_3 = [complex(r.evalf()) for r in sp.nroots(5 * s20**3 + 21 * s20**2 + 4 * s20 + 20)]
    ok_20_1 = (sum(r.real > 0 for r in r_20_1_3) == 2
               and sp.factor(s20**3 - 2 * s20**2 - 2 * s20 + 1) == (s20 + 1) * (s20**2 - 3 * s20 + 1))
    # T3 counterexamples (T1=T2=T3=1): neg fb K=9 unstable, pos fb K=0.5 stable
    ok_20_3 = (any(complex(r.evalf()).real > 0 for r in sp.nroots((s20 + 1)**3 + 9))
               and all(complex(r.evalf()).real < 0 for r in sp.nroots((s20 + 1)**3 - sp.Rational(1, 2))))
    # T5 MIMO decoupling -> diag(1/(5s+1), 1/(s+1))
    G0_20_5 = sp.Matrix([[1 / (2 * s20 + 1), -1], [-1, 1 / (s20 + 1)]])
    Gc_20_5 = sp.Matrix([
        [-(2 * s20 + 1) / (5 * s20**2 * (2 * s20 + 3)), -(2 * s20**2 + 3 * s20 + 1) / (s20**2 * (2 * s20 + 3))],
        [-(2 * s20**2 + 3 * s20 + 1) / (5 * s20**2 * (2 * s20 + 3)), -(s20 + 1) / (s20**2 * (2 * s20 + 3))],
    ])
    Phi_20_5 = sp.simplify((sp.eye(2) + G0_20_5 * Gc_20_5).inv() * (G0_20_5 * Gc_20_5))
    # T6 det(Qo) = -(alpha-gamma)(beta-gamma)
    A_20_6 = sp.Matrix([[-al20, 0], [1, -be20]])
    C_20_6 = sp.Matrix([[1, ga20 - be20]])
    det_20_6 = sp.factor(sp.Matrix.vstack(C_20_6, C_20_6 * A_20_6).det())
    # T7 state-matrix param RL & K > -6/5
    A_20_7 = sp.Matrix([[-1, 0, 0], [K20, -1, 0], [1, 0, -5]])
    G_20_7 = sp.simplify((sp.Matrix([[0, 1, 1]]) * (s20 * sp.eye(3) - A_20_7).inv() * sp.Matrix([1, 0, 0]))[0, 0])
    D_20_7 = sp.expand(sp.numer(sp.together(1 + G_20_7)))
    # T8 ZOH discrete TRUE Kv = (1/T)*lim(z-1)G(z) = K (catches answer book 10x error!) & 0<K<23.92
    e_inv = sp.exp(-1)
    Gz_20_8 = sp.simplify(K20 * (1 - 1 / z20) * (sp.Rational(1, 10) * z20 / (z20 - 1)**2 - sp.Rational(1, 10) * z20 / (z20 - 1) + sp.Rational(1, 10) * z20 / (z20 - e_inv)))
    Kv_20_8 = sp.simplify(10 * sp.limit((z20 - 1) * Gz_20_8, z20, 1))
    a2_20_8 = sp.Rational(1, 10) * (1 - 2 * e_inv) * K20 + e_inv
    Kmax_20_8 = float(sp.solve(sp.Eq(a2_20_8, 1), K20)[0].evalf())
    # T9 square wave exact periodic c0 = -0.5*tanh(1) -> e(2n-1-) = 1 - 0.5*tanh(1) ~ 0.6192
    e_20_9 = float((1 - sp.Rational(1, 2) * sp.tanh(1)).evalf())
    # T10 nonzero IC y(t) = 1/6 - 0.5*e^-2t + (7/3)*e^-3t
    A_20_10 = sp.Matrix([[-3, -2], [0, -2]])
    Ys_20_10 = sp.apart((sp.Matrix([[1, 1]]) * (s20 * sp.eye(2) - A_20_10).inv() * (sp.Matrix([1, 1]) + sp.Matrix([0, 1]) / s20))[0, 0], s20)
    check("ctrl_exam02_2020_all",
          ok_20_1 and ok_20_3
          and Phi_20_5 == sp.Matrix([[1 / (5 * s20 + 1), 0], [0, 1 / (s20 + 1)]])
          and sp.expand(det_20_6 + (al20 - ga20) * (be20 - ga20)) == 0
          and D_20_7 == sp.expand(s20**3 + 7 * s20**2 + (12 + K20) * s20 + (6 + 5 * K20))
          and Kv_20_8 == K20 and abs(Kmax_20_8 - 23.922) < 0.01
          and abs(e_20_9 - 0.6192) < 0.001
          and Ys_20_10 == sp.Rational(1, 6) / s20 - sp.Rational(1, 2) / (s20 + 2) + sp.Rational(7, 3) / (s20 + 3),
          "Verified 2020 T1-T10 (caught T1(3) unstable 84<100 & T8 TRUE Kv=K -> 3<=K<23.92 vs answer book 'no solution'!)")

    # 2. 2021 (T1-T9, first no-calculator year)
    A_21_1 = sp.Matrix([[-1, 1], [2, -3]])
    P_21_1 = sp.Matrix([[sp.Rational(7, 4), sp.Rational(5, 8)], [sp.Rational(5, 8), sp.Rational(3, 8)]])
    r_21_2 = [complex(r.evalf()) for r in sp.nroots(s20**5 + 2 * s20**4 + 2 * s20**3 + 4 * s20**2 + 5 * s20 + 4)]
    # T5 circle RL: 0<zeta<1 -> 2<K<6
    disc_21_5 = sp.factor((sp.Rational(3, 2) * K20 - 3)**2 - 6 * K20)
    # T8 5x5 A -> exact D(x-1) = x^5 - x^4 - 2x^3 + 10x^2 - 5x + 3 (catches answer book -9x!) & 1 root Re<-1
    A_21_8 = sp.Matrix([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [-4, -10, -4, -6, -8], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    D_21_8 = sp.expand((s20 * sp.eye(5) - A_21_8).det())
    x21 = sp.Symbol("x21")
    D_21_8_sh = sp.expand(D_21_8.subs(s20, x21 - 1))
    r_21_8 = [complex(r.evalf()) for r in sp.nroots(D_21_8)]
    # T9 K=[102, 1403, 311], true wn=sqrt(13), sigma%=exp(-2pi/3)*100 ~ 12.31% (catches answer book 3.32 / 9.48%!)
    A_21_9 = sp.Matrix([[-1, 0, 0], [0, 0, 1], [1, 1, -1]])
    B_21_9 = sp.Matrix([1, 0, 0])
    C_21_9 = sp.Matrix([[0, 1, 0]])
    K_21_9 = sp.Matrix([[102, 1403, 311]])
    f_21_9 = sp.expand((s20 * sp.eye(3) - (A_21_9 - B_21_9 * K_21_9)).det())
    ov_21_9 = math.exp(-2 * math.pi / 3) * 100.0
    check("ctrl_exam02_2021_all",
          A_21_1.T * P_21_1 + P_21_1 * A_21_1 == -sp.eye(2) and P_21_1[0, 0] > 0 and P_21_1.det() == sp.Rational(17, 64)
          and sum(r.real > 0 for r in r_21_2) == 2
          and sp.expand(disc_21_5 - sp.Rational(3, 4) * (K20 - 6) * (3 * K20 - 2)) == 0
          and D_21_8_sh == x21**5 - x21**4 - 2 * x21**3 + 10 * x21**2 - 5 * x21 + 3
          and sum(r.real < -1 for r in r_21_8) == 1
          and sp.simplify((C_21_9 * (s20 * sp.eye(3) - A_21_9).inv() * B_21_9)[0, 0] - 1 / (s20**3 + 2 * s20**2 - 1)) == 0
          and f_21_9 == sp.expand((s20 + 100) * (s20**2 + 4 * s20 + 13))
          and abs(ov_21_9 - 12.31) < 0.02,
          "Verified 2021 T1-T9 (caught T3 4K->K=5000, T8 true -5x vs -9x, T9 true wn=sqrt(13) & sigma%=12.31% vs 9.48%!)")

    # 3. 2022 (T1-T10)
    # T2(2) S&H LTV counterexample: u(t)=t, t0=T/2 -> at t=1.25T: y(t-T/2)=y(0.75T)=0 vs y_tilde(1.25T)=u_tilde(T)=T/2 != 0
    ok_22_2 = (0 != sp.Rational(1, 2))
    # T3 a=1, b=4, d=5
    A_22_3 = sp.Matrix([[0, -3], [1, -4]])
    G_22_3 = sp.simplify((sp.Matrix([[0, 2]]) * (s20 * sp.eye(2) - A_22_3).inv() * sp.Matrix([5, 0]))[0, 0])
    G_22_3_d = sp.simplify((sp.Matrix([[-5, 5]]) * (s20 * sp.eye(2) - sp.Matrix([[-3, 0], [0, -1]])).inv() * sp.Matrix([1, 1]))[0, 0])
    # T5 closed-loop reduced 2/(s^2+2s+2) -> ov=exp(-pi)*100=4.32%, tr=3pi/4
    # T6 exact unmodified 3rd-order char eq at s=-0.2+0.3j
    a22, b22 = sp.symbols("a22 b22", real=True)
    D_22_6 = sp.expand((s20 - sp.Rational(1, 2)) * (s20 + a22) * (s20 + b22) + (s20 + sp.Rational(3, 5)) * (s20 - sp.Rational(37, 100)))
    v_22_6 = sp.expand(D_22_6.subs(s20, -sp.Rational(1, 5) + sp.Rational(3, 10) * sp.I))
    sol_22_6 = sp.solve([sp.re(v_22_6), sp.im(v_22_6)], [a22, b22])
    # T7 3rd-order parallel feedback: K=[13.5,-7.5,-17], H=[24,-149.5,-167]^T
    A_22_7 = sp.Matrix([[-1, 0, -1], [0, -3, -1], [0, 2, -2]])
    B_22_7 = sp.Matrix([1, 1, 0])
    C_22_7 = sp.Matrix([[1, 0, 0]])
    K_22_7 = sp.Matrix([[sp.Rational(27, 2), sp.Rational(-15, 2), -17]])
    H_22_7 = sp.Matrix([24, sp.Rational(-299, 2), -167])
    # T9 RC divider: R1=R2=5000*sqrt(3), C=2e-4 -> b=1/2, T=sqrt(3)/2 -> 100*b/sqrt(1+4T^2)=25, atan(2T)=pi/3
    R_22_9 = 5000 * sp.sqrt(3)
    C_22_9 = sp.Rational(2, 10000)
    T_22_9 = sp.simplify((R_22_9 * R_22_9 * C_22_9) / (2 * R_22_9))
    # T10 dual-sampler: D(z) = z^2 + (0.8K-0.24)z + 0.008 -> -0.96<K<1.56; K=5 continuous y(t) at T+, 2T+, 3T+ == Y(z)
    a10_v = sp.Rational(1, 5)
    Dz_22_10 = sp.expand((z20 - a10_v) * (z20 - a10_v**2) + K20 * (1 - a10_v) * z20)
    Mz_22_10 = sp.simplify(4 * (z20 - a10_v**2) / Dz_22_10.subs(K20, 5))
    Yz_22_10 = sp.simplify(z20 / (z20 - a10_v**2) * Mz_22_10)
    u10 = sp.Symbol("u10")
    m_ser = sp.Poly(sp.series(Mz_22_10.subs(z20, 1 / u10), u10, 0, 4).removeO(), u10)
    y_ser = sp.Poly(sp.series(Yz_22_10.subs(z20, 1 / u10), u10, 0, 4).removeO(), u10)
    m_c = [m_ser.nth(k) for k in range(4)]
    y_c = [y_ser.nth(k) for k in range(4)]
    ok_22_10_cont = all(
        sp.simplify(sum(m_c[k] * (a10_v**2)**(n_i - k) for k in range(n_i + 1)) - y_c[n_i]) == 0
        for n_i in (1, 2, 3)
    )
    check("ctrl_exam02_2022_all",
          ok_22_2 and sp.simplify(G_22_3 - G_22_3_d) == 0
          and len(sol_22_6) == 2 and abs(float(sol_22_6[0][0].evalf()) + float(sol_22_6[0][1].evalf()) - (-0.35345)) < 0.001
          and sp.expand((s20 * sp.eye(3) - (A_22_7 - B_22_7 * K_22_7)).det()) == sp.expand((s20 + 10) * (s20**2 + 2 * s20 + 4))
          and sp.expand((s20 * sp.eye(3) - (A_22_7 - H_22_7 * C_22_7)).det()) == sp.expand((s20 + 10)**3)
          and T_22_9 == sp.sqrt(3) / 2 and sp.atan(2 * T_22_9) == sp.pi / 3
          and sp.solve(Dz_22_10.subs(z20, 1), K20) == [sp.Rational(-24, 25)]
          and sp.solve(Dz_22_10.subs(z20, -1), K20) == [sp.Rational(39, 25)]
          and y_c[1] == 4 and y_c[2] == sp.Rational(-376, 25) and sp.simplify(y_c[3] - sp.Rational(35324, 625)) == 0
          and ok_22_10_cont,
          "Verified 2022 T1-T10 (caught T2 S&H LTV erratum, T6 no-K exact & s-b sign error, T10 true continuous piecewise-exp y(t)!)")

    # =====================================================================
    # Control Exam-03: 2023-2026 Latest 40 Real-Exam Problems + 10 Errata Audits
    # =====================================================================
    s23, z23, t23, K23, a23, b23, T23, w23 = sp.symbols("s23 z23 t23 K23 a23 b23 T23 w23", real=True)
    # 1. 2023 (T1-T11 independent SymPy derivations)
    phi1_23, phi2_23 = (1 + sp.sqrt(5)) / 2, (1 - sp.sqrt(5)) / 2
    fib_23 = [sp.simplify((phi1_23**(k + 1) - phi2_23**(k + 1)) / sp.sqrt(5)) for k in range(6)]
    # T3: D(s) = s^3 + (1+a)s^2 + a*s + 1 -> stable iff a > (sqrt(5)-1)/2 ~ 0.618, disjoint from |1-2a|<0.05 (0.475,0.525)
    D_23_3 = sp.expand(s23**3 + (1 + a23) * s23**2 + (a23 - 2) * s23 + (1 - 2 * a23) + 2 * (s23 + a23))
    a_gold = (sp.sqrt(5) - 1) / 2
    # T4: G(s) = s*Phi/(0.5 - 2*Phi) = 2*wn^2*s / (s^2 + 2*zeta*wn*s - 3*wn^2) -> unstable RHP pole
    r_23_4 = [complex(r.evalf()) for r in sp.nroots(s23**2 + 0.5086 * s23 - 3 * 2.0854)]
    # T6(3): d(eB)/d(zeta) = (a*t*cosh(a*t) - sinh(a*t)) * exp(-zeta*t) / a^3 > 0
    ze23, tt23 = sp.symbols("ze23 tt23", positive=True)
    al23 = sp.sqrt(ze23**2 - 1)
    eB_23 = sp.exp(-ze23 * tt23) * (sp.cosh(al23 * tt23) + (ze23 / al23) * sp.sinh(al23 * tt23))
    deB_23 = sp.simplify(sp.diff(eB_23, ze23) - (tt23 * al23 * sp.cosh(al23 * tt23) - sp.sinh(al23 * tt23)) * sp.exp(-ze23 * tt23) / al23**3)
    # T7: exact Fig 3 closed-loop monic denom s^2 + (5.5+0.2K)s + (K+22.5)
    R23, Y23, W23 = sp.symbols("R23 Y23 W23")
    sol_23_7 = sp.solve([
        sp.Eq(W23, sp.Rational(-2, 5) / (2 * s23 + sp.Rational(2, 5) * K23) * (30 * Y23 + sp.Rational(5, 2) * W23)),
        sp.Eq((sp.Rational(3, 10) * s23 + 1) * Y23, sp.Rational(1, 2) * (R23 - Y23) + W23),
    ], [Y23, W23])
    Phi_23_7 = sp.simplify(sol_23_7[Y23] / R23)
    Phi_23_7_exp = sp.Rational(1, 3) * (5 * s23 + K23 + sp.Rational(5, 2)) / (s23**2 + (sp.Rational(11, 2) + sp.Rational(1, 5) * K23) * s23 + (K23 + sp.Rational(45, 2)))
    # T8: Phi=0.5(s+10)/(s^2+2s+5) -> G=0.5(s+10)/(s(s+1.5)), R^2=85
    Phi_23_8 = sp.Rational(1, 2) * (s23 + 10) / (s23**2 + 2 * s23 + 5)
    G_23_8 = sp.simplify(Phi_23_8 / (1 - Phi_23_8))
    # T9: discrete SS -> (0.368z+0.264)/(z^2-z+0.632)
    A_23_9 = sp.Matrix([[0, -1], [sp.Rational(632, 1000), 1]])
    B_23_9 = sp.Matrix([1, sp.Rational(-632, 1000)])
    C_23_9 = sp.Matrix([[1, 1]])
    Phi_23_9 = sp.simplify((C_23_9 * (z23 * sp.eye(2) - A_23_9).inv() * B_23_9)[0, 0])
    check("ctrl_exam03_2023_all",
          fib_23 == [1, 1, 2, 3, 5, 8]
          and sp.expand(D_23_3 - (s23**3 + (1 + a23) * s23**2 + a23 * s23 + 1)) == 0 and float(a_gold.evalf()) > 0.618
          and any(r.real > 2.0 for r in r_23_4)
          and deB_23 == 0
          and sp.simplify(Phi_23_7 - Phi_23_7_exp) == 0
          and sp.simplify(G_23_8 - sp.Rational(1, 2) * (s23 + 10) / (s23 * (s23 + sp.Rational(3, 2)))) == 0
          and sp.simplify(Phi_23_9 - (sp.Rational(368, 1000) * z23 + sp.Rational(264, 1000)) / (z23**2 - z23 + sp.Rational(632, 1000))) == 0,
          "Verified 2023 T1-T11 independent derivations (T3 golden ratio conflict, T4 unstable G(s), T6(3) exact PDE proof, T7-T9)")

    # 2. 2024 (T1-T10 + 6 answer-book errata checks)
    k1_24, k2_24 = sp.symbols("k1_24 k2_24", positive=True)
    # T2: ess_r = 5/(5+k1*k2), ess_n = -k2/(5+k1*k2); both |ess_r| and |ess_n| strictly decrease as k1 increases!
    essr_24 = 5 / (5 + k1_24 * k2_24)
    essn_24 = k2_24 / (5 + k1_24 * k2_24)
    # T3: u*(1+e^-u) = 4 -> u_max ~ 3.9224 (a=1 -> T<3.92; a=1.1 -> T<3.566)
    u24 = sp.Symbol("u24", positive=True)
    u24_max = float(sp.nsolve(u24 * (1 + sp.exp(-u24)) - 4, u24, 3.9))
    # T4: det(sI-A) = s*((s+0.2)(s+0.5) + k*(s+0.3))
    A_24_4 = sp.Matrix([[-K23, -K23, 0], [-sp.Rational(2, 5), -sp.Rational(1, 5), -sp.Rational(1, 5)], [-sp.Rational(1, 2), 0, -sp.Rational(1, 2)]])
    D_24_4 = sp.expand((s23 * sp.eye(3) - A_24_4).det())
    # T8: K=[28,46,10] cancels s+3 -> (s+1)/((s+2)(s+6))
    A_24_8 = sp.Matrix([[0, 1, 0], [0, 0, 1], [-8, 10, -1]])
    B_24_8 = sp.Matrix([0, 0, 1])
    C_24_8 = sp.Matrix([[3, 4, 1]])
    K_24_8 = sp.Matrix([[28, 46, 10]])
    Phi_24_8 = sp.simplify((C_24_8 * (s23 * sp.eye(3) - (A_24_8 - B_24_8 * K_24_8)).inv() * B_24_8)[0, 0])
    check("ctrl_exam03_2024_all",
          sp.simplify(sp.diff(essr_24, k1_24)).subs({k1_24: 10, k2_24: 1}) < 0
          and sp.simplify(sp.diff(essn_24, k1_24)).subs({k1_24: 10, k2_24: 1}) < 0
          and abs(u24_max - 3.9224) < 0.001 and abs(u24_max / 1.1 - 3.5658) < 0.001
          and sp.expand(D_24_4 - s23 * ((s23 + sp.Rational(1, 5)) * (s23 + sp.Rational(1, 2)) + K23 * (s23 + sp.Rational(3, 10)))) == 0
          and sp.simplify(Phi_24_8 - (s23 + 1) / ((s23 + 2) * (s23 + 6))) == 0,
          "Verified 2024 T1-T10 (caught T2 k1=1/k2=0 fallacy, T3 u_max=3.922, T4 hidden pole s=0, T8 K=[28,46,10])")

    # 3. 2025 (T1-T10 independent SymPy derivations)
    # T2: 1 + G1*(K1+K2*G2) -> s^3+14s^2+(40+5K1)s+(50K1-20K2)=0 -> 5K1>2K2
    D_25_2 = sp.expand(s23 * (s23 + 4) * (s23 + 10) + 5 * k1_24 * (s23 + 10) - 20 * k2_24)
    # T3: observable canonical -> K=12, w=sqrt(2)
    A_25_3 = sp.Matrix([[0, 0, 0, -20 - K23], [1, 0, 0, -4], [0, 1, 0, -18], [0, 0, 1, -2]])
    G_25_3 = sp.simplify((sp.Matrix([[0, 0, 0, 1]]) * (s23 * sp.eye(4) - A_25_3).inv() * sp.Matrix([4, 1, 4, 0]))[0, 0])
    D_25_3_12 = sp.expand((s23 * sp.eye(4) - A_25_3.subs(K23, 12)).det())
    # T4: Y(z) = z^2 / (2z^2 - (1+2a)z + a), y(0)=1/2, y(T)=(1+2a)/4, y(inf)=0
    a25 = sp.exp(-T23)
    Yz_25_4 = z23**2 / (2 * z23**2 - (1 + 2 * a25) * z23 + a25)
    u25 = sp.Symbol("u25")
    ser_25_4 = sp.Poly(sp.series(Yz_25_4.subs(z23, 1 / u25), u25, 0, 3).removeO(), u25)
    # T5: lam2=-2 -> A=[[0,1],[-2,-3]], G(ln2)=[[3/4,1/4],[-1/2,0]], H(ln2)=[1/2,-1/2]^T
    A_25_5 = sp.Matrix([[0, 1], [-2, -3]])
    G_25_5 = sp.simplify(sp.exp(A_25_5 * sp.log(2)))
    H_25_5 = sp.simplify(A_25_5.inv() * (G_25_5 - sp.eye(2)) * sp.Matrix([1, -1]))
    # T8: symmetric A*B = p*B -> b = 2*sqrt(3), a = 5*sqrt(3)/4, p = 2, G(s) = 1/(s-2)
    A_25_8 = sp.Matrix([[sp.Rational(3, 4), 5 * sp.sqrt(3) / 4, -2], [5 * sp.sqrt(3) / 4, sp.Rational(-7, 4), 2 * sp.sqrt(3)], [-2, 2 * sp.sqrt(3), 1]])
    B_25_8 = sp.Matrix([sp.sqrt(3) / 2, sp.Rational(1, 2), 0])
    G_25_8 = sp.simplify((B_25_8.T * (s23 * sp.eye(3) - A_25_8).inv() * B_25_8)[0, 0])
    # T10: equilibria (-2,1) stable, (0,-1) unstable
    J1_25_10 = sp.Matrix([[0, 2], [-1, -1]])
    J2_25_10 = sp.Matrix([[0, -2], [-1, -1]])
    check("ctrl_exam03_2025_all",
          sp.expand(D_25_2 - (s23**3 + 14 * s23**2 + (40 + 5 * k1_24) * s23 + (50 * k1_24 - 20 * k2_24))) == 0
          and sp.simplify(G_25_3 - (4 * s23**2 + s23 + 4) / (s23**4 + 2 * s23**3 + 18 * s23**2 + 4 * s23 + 20 + K23)) == 0
          and sp.expand(D_25_3_12 - (s23**2 + 2) * (s23**2 + 2 * s23 + 16)) == 0
          and ser_25_4.nth(0) == sp.Rational(1, 2) and sp.simplify(ser_25_4.nth(1) - (1 + 2 * a25) / 4) == 0
          and sp.limit((z23 - 1) * Yz_25_4, z23, 1) == 0
          and G_25_5 == sp.Matrix([[sp.Rational(3, 4), sp.Rational(1, 4)], [sp.Rational(-1, 2), 0]])
          and H_25_5 == sp.Matrix([sp.Rational(1, 2), sp.Rational(-1, 2)])
          and sp.simplify(G_25_8 - 1 / (s23 - 2)) == 0
          and all(complex(r.evalf()).real < 0 for r in J1_25_10.eigenvals().keys())
          and set(J2_25_10.eigenvals().keys()) == {1, -2},
          "Verified 2025 T1-T10 independent derivations (T2 5K1>2K2, T3 K=12, T4 y(inf)=0, T5 companion A, T8 b=2*sqrt(3), T10)")

    # 4. 2026 (T1-T9 + 3 answer-book errata checks)
    # T2: envelope 0.5*ln(20*sqrt(10)) ~ 2.073s < 4.744s
    ts1_26 = float(sp.nsolve((1 + tt23) * sp.exp(-tt23) - 0.05, tt23, 4.7))
    ts2_26_env = 0.5 * math.log(20 * math.sqrt(10))
    # T4: 2K oscillation w=20 -> b=10, K=10000
    eq_26_4 = sp.expand((20 * sp.I)**3 + (40 + b23) * (20 * sp.I)**2 + 40 * b23 * (20 * sp.I) + 2 * K23)
    sol_26_4 = sp.solve([sp.re(eq_26_4), sp.im(eq_26_4)], [b23, K23])
    # T5: G(s)=1000/(s(s+5)(s+20)) -> w_x=10, h=2.5
    G_26_5_10 = sp.simplify(1000 / (10 * sp.I * (10 * sp.I + 5) * (10 * sp.I + 20)))
    # T6: a=-0.368, K~1.535, b~0.182
    sol_26_6 = sp.solve([b23 - 1 + 0.368 * K23 + 2 * 0.472 * 0.2675, 0.264 * K23 - b23 - 0.472**2], [K23, b23])
    # T7: TRUE char poly s^2 + (K1+K2)s + 2*K1*K2 (catches answer book P.14 missing -L3!)
    A_26_7 = sp.Matrix([[-k1_24, k2_24], [-k1_24, -k2_24]])
    D_26_7 = sp.expand((s23 * sp.eye(2) - A_26_7).det())
    Phi_26_7 = sp.simplify(sp.exp(A_26_7.subs({k1_24: 2, k2_24: 2}) * t23))
    # T8: cascade S1->S2 det(Qc)=-3, det(Qo)=8
    A_26_8 = sp.Matrix([[-2, 1, -1], [0, 0, 1], [0, -3, -4]])
    B_26_8 = sp.Matrix([0, 0, 1])
    C_26_8 = sp.Matrix([[1, 0, 0]])
    # T9: complex DF -1/N(A) = 0.5 - j*A == G(2j) = 0.5 - 1.5j -> w=2, A=1.5
    A26_s = sp.Symbol("A26_s", positive=True)
    neg_inv_26_9 = sp.simplify(-1 / (-(2 + 4 * sp.I * A26_s) / (1 + 4 * A26_s**2)))
    G_26_9 = sp.simplify(20 / ((2 + 2 * sp.I) * (4 + 2 * sp.I)))
    check("ctrl_exam03_2026_all",
          ts2_26_env < ts1_26 and abs(ts1_26 - 4.744) < 0.01 and abs(ts2_26_env - 2.073) < 0.01
          and sol_26_4 == {b23: 10, K23: 10000}
          and G_26_5_10 == sp.Rational(-2, 5)
          and abs(float(sol_26_6[K23]) - 1.535) < 0.002 and abs(float(sol_26_6[b23]) - 0.1825) < 0.002
          and sp.expand(D_26_7 - (s23**2 + (k1_24 + k2_24) * s23 + 2 * k1_24 * k2_24)) == 0
          and sp.simplify(Phi_26_7 - sp.exp(-2 * t23) * sp.Matrix([[sp.cos(2 * t23), sp.sin(2 * t23)], [-sp.sin(2 * t23), sp.cos(2 * t23)]])) == sp.zeros(2, 2)
          and sp.Matrix.hstack(B_26_8, A_26_8 * B_26_8, A_26_8**2 * B_26_8).det() == -3
          and sp.Matrix.vstack(C_26_8, C_26_8 * A_26_8, C_26_8 * A_26_8**2).det() == 8
          and sp.simplify(neg_inv_26_9 - (sp.Rational(1, 2) - sp.I * A26_s)) == 0 and G_26_9 == sp.Rational(1, 2) - sp.Rational(3, 2) * sp.I,
          "Verified 2026 T1-T9 (caught T3(2) a<0 sign flip, T7(1) TRUE +2*K1*K2 vs P.14 missing -L3, T9 complex DF w=2 & A=1.5)")

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
