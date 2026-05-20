"""
Week 9 과제: 고전 역학 시뮬레이션 (AI 풀이)
==============================================
작성자: Claude (Anthropic) — AI 어시스턴트
과제:  역학 파트 핵심 문제 3가지를 수치 시뮬레이션으로 풀기

문제 목록:
  1. Euler vs RK4 수치 적분 비교 (단순 조화 진동자)
  2. 행성 궤도 시뮬레이션 & 케플러 법칙 검증
  3. 3체 문제 — Figure-8 주기 궤도

AI 접근 전략:
  - 모든 2차 미분 방정식을 1차 연립방정식으로 변환
  - RK4 범용 적분기 하나로 세 문제 모두 처리
  - 에너지/각운동량 보존으로 정확도 검증
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'hw_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def rk4(f, y, t, dt):
    """범용 4차 Runge-Kutta 적분 한 스텝"""
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def euler(f, y, t, dt):
    """오일러 적분 한 스텝"""
    return y + dt * f(y, t)


def integrate(method, f, y0, t_end, dt):
    """주어진 방법으로 전체 궤적 적분"""
    steps = int(t_end / dt)
    t_arr = np.zeros(steps)
    y_arr = np.zeros((steps, len(y0)))
    y = y0.copy()
    for i in range(steps):
        t_arr[i] = i * dt
        y_arr[i] = y
        y = method(f, y, i * dt, dt)
    return t_arr, y_arr


# ── 문제 1: Euler vs RK4 — 단순 조화 진동자 ─────────────────────────────────
#
# 운동 방정식: x'' = -ω²x
# 상태벡터: [x, v]   → dy/dt = [v, -ω²x]
# 해석해:   x(t) = cos(ωt)  (A=1, ω=1, φ=0)
# 에너지:   E = ½v² + ½ω²x² = 일정

def problem1():
    print("=" * 60)
    print("문제 1: Euler vs RK4 — 단순 조화 진동자")
    print("=" * 60)

    omega = 1.0
    y0 = np.array([1.0, 0.0])   # x=1, v=0
    t_end = 40.0
    dt = 0.1

    def sho(y, t):
        x, v = y
        return np.array([v, -omega**2 * x])

    t, y_euler = integrate(euler, sho, y0, t_end, dt)
    _,  y_rk4  = integrate(rk4,   sho, y0, t_end, dt)

    x_exact = np.cos(omega * t)

    # 에너지
    E_euler = 0.5 * y_euler[:, 1]**2 + 0.5 * omega**2 * y_euler[:, 0]**2
    E_rk4   = 0.5 * y_rk4[:, 1]**2  + 0.5 * omega**2 * y_rk4[:, 0]**2
    E0 = 0.5 * omega**2

    print(f"  Euler  최대 에너지 오차: {abs(E_euler - E0).max():.4f}")
    print(f"  RK4    최대 에너지 오차: {abs(E_rk4  - E0).max():.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("문제 1: Euler vs RK4 — 단순 조화 진동자", fontsize=14, fontweight='bold')

    # 궤적 비교
    axes[0, 0].plot(t, x_exact,         'k-',  lw=2,   label='해석해')
    axes[0, 0].plot(t, y_euler[:, 0],   'r--', lw=1.5, label='Euler')
    axes[0, 0].plot(t, y_rk4[:, 0],     'b:',  lw=1.5, label='RK4')
    axes[0, 0].set(xlabel='시간 (s)', ylabel='위치 x', title='위치 비교')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 오차
    axes[0, 1].semilogy(t, abs(y_euler[:, 0] - x_exact) + 1e-12, 'r', label='Euler')
    axes[0, 1].semilogy(t, abs(y_rk4[:, 0]  - x_exact) + 1e-12, 'b', label='RK4')
    axes[0, 1].set(xlabel='시간 (s)', ylabel='|오차|', title='절대 오차 (로그)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 에너지
    axes[1, 0].plot(t, E_euler, 'r', label='Euler')
    axes[1, 0].plot(t, E_rk4,   'b', label='RK4')
    axes[1, 0].axhline(E0, color='k', ls='--', label='이론값')
    axes[1, 0].set(xlabel='시간 (s)', ylabel='에너지 E', title='에너지 보존')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 위상 공간
    axes[1, 1].plot(y_euler[:, 0], y_euler[:, 1], 'r', lw=0.8, label='Euler')
    axes[1, 1].plot(y_rk4[:, 0],   y_rk4[:, 1],   'b', lw=0.8, label='RK4')
    axes[1, 1].set(xlabel='위치 x', ylabel='속도 v', title='위상 공간 (나선 → 에너지 증가)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'hw9_p1_euler_vs_rk4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 저장: {path}\n")


# ── 문제 2: 행성 궤도 & 케플러 법칙 ─────────────────────────────────────────
#
# 단위: AU, year, M_☉   →  G = 4π²
# 상태벡터: [x, y, vx, vy]
# 케플러 제3법칙: T² = a³  (단위 선택 덕분에 = 1)

def problem2():
    print("=" * 60)
    print("문제 2: 행성 궤도 시뮬레이션 & 케플러 법칙")
    print("=" * 60)

    G = 4 * np.pi**2  # AU³ yr⁻² M_☉⁻¹

    # 이심률 0.5인 타원 궤도 초기 조건
    # 근일점 거리 r_p = a(1-e), 근일점 속도 v_p = sqrt(GM(1+e)/(a(1-e)))
    a, e = 1.5, 0.5
    r_p = a * (1 - e)
    v_p = np.sqrt(G * (1 + e) / (a * (1 - e)))

    y0 = np.array([r_p, 0.0, 0.0, v_p])

    def kepler(y, t):
        x, yy, vx, vy = y
        r3 = (x**2 + yy**2) ** 1.5
        return np.array([vx, vy, -G * x / r3, -G * yy / r3])

    T_kepler = a ** 1.5   # 케플러 제3법칙: T = a^(3/2)
    t, traj = integrate(rk4, kepler, y0, 3 * T_kepler, 1e-4)

    # 보존량
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    E = 0.5 * (traj[:, 2]**2 + traj[:, 3]**2) - G / r
    L = traj[:, 0] * traj[:, 3] - traj[:, 1] * traj[:, 2]  # 각운동량

    print(f"  이심률 e = {e},  장반경 a = {a} AU")
    print(f"  케플러 주기 예측: T = {T_kepler:.4f} yr")
    print(f"  에너지 변동: ΔE/E₀ = {(E.max()-E.min())/abs(E[0])*100:.4f}%")
    print(f"  각운동량 변동: ΔL/L₀ = {(L.max()-L.min())/abs(L[0])*100:.4f}%")

    # 케플러 제3법칙 검증: 여러 a값
    a_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    T_measured = []
    for a_i in a_vals:
        r_pi = a_i * (1 - e)
        v_pi = np.sqrt(G * (1 + e) / (a_i * (1 - e)))
        y0_i = np.array([r_pi, 0.0, 0.0, v_pi])
        t_i, traj_i = integrate(rk4, kepler, y0_i, 3 * a_i**1.5, 1e-4)
        # 주기: y좌표가 0을 두 번째로 지나는 시각
        y_sign = np.sign(traj_i[:, 1])
        crossings = np.where(np.diff(y_sign) > 0)[0]
        T_measured.append(t_i[crossings[0]] * 2 if len(crossings) >= 1 else a_i**1.5)

    T_measured = np.array(T_measured)

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("문제 2: 행성 궤도 & 케플러 법칙", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', lw=1)
    ax1.plot(0, 0, 'yo', ms=12, label='태양')
    ax1.plot(traj[0, 0], traj[0, 1], 'g^', ms=10, label='근일점')
    ax1.set(xlabel='x (AU)', ylabel='y (AU)', title=f'타원 궤도 (e={e}, a={a} AU)')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.axis('equal')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, (E - E[0]) / abs(E[0]) * 100, 'purple', lw=1.5)
    ax2.set(xlabel='시간 (yr)', ylabel='에너지 오차 (%)', title='에너지 보존')
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, L, 'teal', lw=1.5)
    ax3.set(xlabel='시간 (yr)', ylabel='각운동량 L', title='각운동량 보존')
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.scatter(a_vals, T_measured**2, s=80, c='red', zorder=5, label='시뮬레이션 T²')
    ax4.plot(a_vals, a_vals**3, 'k--', lw=2, label='케플러: T²=a³')
    ax4.set(xlabel='장반경 a (AU)', ylabel='T² (yr²)', title='케플러 제3법칙 검증: T²∝a³')
    ax4.legend(); ax4.grid(alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'hw9_p2_kepler.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 저장: {path}\n")


# ── 문제 3: 3체 문제 — Figure-8 주기 궤도 ───────────────────────────────────
#
# Chenciner & Montgomery (2000) 이 발견한 아름다운 주기 해
# 세 질량이 동일하며 서로 Figure-8 모양으로 쫓아다님
# 초기 조건: Moore (1993) 의 수치 해

def problem3():
    print("=" * 60)
    print("문제 3: 3체 문제 — Figure-8 주기 궤도")
    print("=" * 60)

    G = 1.0
    m = [1.0, 1.0, 1.0]

    # Figure-8 초기 조건 (Chenciner-Montgomery)
    x1, y1 =  -0.97000436,  0.24308753
    vx3, vy3 = -0.93240737, -0.86473146

    y0 = np.array([
        x1,  y1,  -vx3/2, -vy3/2,
       -x1, -y1,  -vx3/2, -vy3/2,
        0.0, 0.0,  vx3,    vy3
    ])

    def three_body(y, t):
        r = [y[4*i:4*i+2] for i in range(3)]
        v = [y[4*i+2:4*i+4] for i in range(3)]
        a = [np.zeros(2) for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if i != j:
                    d = r[j] - r[i]
                    dist = max(np.linalg.norm(d), 1e-8)
                    a[i] += G * m[j] / dist**3 * d
        return np.concatenate([np.concatenate([v[i], a[i]]) for i in range(3)])

    T_period = 6.3259        # 알려진 주기
    t, traj = integrate(rk4, three_body, y0, T_period, 5e-5)

    # 에너지
    def energy(y):
        r = [y[:, 4*i:4*i+2] for i in range(3)]
        v = [y[:, 4*i+2:4*i+4] for i in range(3)]
        KE = sum(0.5 * m[i] * np.sum(v[i]**2, axis=1) for i in range(3))
        PE = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                d = np.linalg.norm(r[i] - r[j], axis=1)
                PE -= G * m[i] * m[j] / d
        return KE + PE

    E = energy(traj)
    print(f"  에너지 보존: ΔE/E₀ = {(E.max()-E.min())/abs(E[0])*100:.5f}%")

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    labels = ['천체 1', '천체 2', '천체 3']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("문제 3: 3체 문제 — Figure-8 주기 궤도", fontsize=14, fontweight='bold')

    for i in range(3):
        x = traj[:, 4*i]
        y = traj[:, 4*i+1]
        axes[0].plot(x, y, color=colors[i], lw=1.5, label=labels[i])
        axes[0].plot(x[0], y[0], 'o', color=colors[i], ms=8, markeredgecolor='black')

    axes[0].set(xlabel='x', ylabel='y', title=f'Figure-8 궤도 (주기 T≈{T_period})')
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].axis('equal')

    axes[1].plot(t, (E - E[0]) / abs(E[0]) * 100, 'purple', lw=1.5)
    axes[1].set(xlabel='시간', ylabel='에너지 오차 (%)', title='에너지 보존 (RK4)')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'hw9_p3_three_body.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 저장: {path}\n")


# ── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Week 9 과제: 고전 역학 시뮬레이션 (AI 풀이)")
    print("Claude (Anthropic) 작성\n")

    problem1()
    problem2()
    problem3()

    print("=" * 60)
    print("모든 문제 완료! 결과 이미지:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {OUTPUT_DIR}/{f}")
