"""순수 물리/수학 함수 단위 테스트"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Lab 4: 진자 주기 ──────────────────────────────────────────────────
class TestCalculateTruePeriod:
    def test_small_angle_formula(self):
        """작은 각도(θ→0)에서 T ≈ 2π√(L/g)"""
        from week4_app import calculate_true_period
        g = 9.81
        L = 1.0
        expected = 2 * np.pi * np.sqrt(L / g)
        result = calculate_true_period(L, theta0_deg=1.0)
        assert abs(result - expected) < 0.01  # 1% 이내

    def test_longer_pendulum_longer_period(self):
        """길이가 4배 → 주기 2배"""
        from week4_app import calculate_true_period
        T1 = calculate_true_period(1.0, 10.0)
        T2 = calculate_true_period(4.0, 10.0)
        assert abs(T2 / T1 - 2.0) < 0.01

    def test_larger_angle_longer_period(self):
        """큰 각도는 작은 각도보다 주기가 길다"""
        from week4_app import calculate_true_period
        T_small = calculate_true_period(1.0, 5.0)
        T_large = calculate_true_period(1.0, 75.0)
        assert T_large > T_small

    def test_positive_period(self):
        """주기는 항상 양수"""
        from week4_app import calculate_true_period
        for L in [0.5, 1.0, 2.0]:
            for theta in [10, 45, 70]:
                assert calculate_true_period(L, theta) > 0


# ── Lab 3: 과적합 목표 함수 ───────────────────────────────────────────
class TestTrueFunction:
    def test_at_zero(self):
        """x=0 이면 sin(0) + 0 = 0"""
        from week4_app import overfitting_true_function
        assert overfitting_true_function(np.array([0.0])) == pytest.approx(0.0)

    def test_shape_preserved(self):
        """입력 shape = 출력 shape"""
        from week4_app import overfitting_true_function
        x = np.linspace(-2, 2, 100)
        y = overfitting_true_function(x)
        assert y.shape == x.shape


# ── Lab 2: 포물선 데이터 생성 ─────────────────────────────────────────
class TestGenerateProjectileData:
    def test_output_shapes(self):
        """X shape=(N,3), Y shape=(N,2)"""
        from week4_app import generate_projectile_data
        X, Y = generate_projectile_data(n_samples=200)
        assert X.ndim == 2 and X.shape[1] == 3
        assert Y.ndim == 2 and Y.shape[1] == 2

    def test_y_nonnegative(self):
        """y 좌표는 항상 0 이상 (땅 위)"""
        from week4_app import generate_projectile_data
        _, Y = generate_projectile_data(n_samples=500, noise_level=0.0)
        assert np.all(Y[:, 1] >= -0.01)  # 노이즈 없음, 약간의 부동소수점 허용


# ── Lab 1: 함수 정의 ─────────────────────────────────────────────────
class TestLab1Functions:
    def test_sin_values(self):
        """sin 함수 값 검증"""
        from week4_app import lab1_get_function
        fn = lab1_get_function("sin(x)")
        x = np.array([0.0, np.pi / 2])
        y = fn(x)
        assert y[0] == pytest.approx(0.0)
        assert y[1] == pytest.approx(1.0, abs=1e-5)

    def test_x_sin_x(self):
        """x·sin(x) at x=0 → 0"""
        from week4_app import lab1_get_function
        fn = lab1_get_function("x·sin(x)")
        assert fn(np.array([0.0]))[0] == pytest.approx(0.0)
