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
