import math
import pytest
import sympy as sp
from calculus_engine import CalculusGeometryEngine

@pytest.fixture
def engine():
    return CalculusGeometryEngine(variable_name="x")

def test_area_parabola_and_line(engine):
    r"""Tính S hình phẳng giữa y = x^2 và y = 2x.
    Nghiệm: x = 0, x = 2.
    S = \int_0^2 |x^2 - 2x| dx = 4/3.
    """
    res = engine.calculate_area(f_expr="x**2", g_expr="2*x")
    assert res.exact_value == sp.Rational(4, 3)
    assert pytest.approx(res.numeric_value, 1e-5) == 4 / 3

def test_rotation_volume_ox(engine):
    r"""Thể tích khi quay hình phẳng giới hạn bởi y = sqrt(x), y = 0, x = 4 quanh Ox.
    V = \pi \int_0^4 (sqrt(x))^2 dx = \pi [x^2/2]_0^4 = 8\pi.
    """
    res = engine.calculate_rotation_volume_ox(f_expr="sqrt(x)", g_expr="0", a=0, b=4)
    assert res.exact_value == 8 * sp.pi
    assert pytest.approx(res.numeric_value, 1e-5) == 8 * math.pi

def test_rotation_volume_oy_shell(engine):
    r"""Thể tích khi quay y = x^2 từ x=0 đến x=1 quanh Oy.
    V = 2\pi \int_0^1 x*(x^2) dx = 2\pi [x^4/4]_0^1 = \pi / 2.
    """
    res = engine.calculate_rotation_volume_oy(f_expr="x**2", g_expr="0", a=0, b=1)
    assert res.exact_value == sp.pi / 2
    assert pytest.approx(res.numeric_value, 1e-5) == math.pi / 2

if __name__ == "__main__":
    # Chạy thử nghiệm trực tiếp
    eng = CalculusGeometryEngine()
    print("--- 1. Diện tích giữa y = x^2 và y = x + 2 ---")
    r1 = eng.calculate_area("x**2", "x + 2")
    print(r1)

    print("\n--- 2. Thể tích tròn xoay quanh Ox: y = sin(x), x in [0, pi] ---")
    r2 = eng.calculate_rotation_volume_ox("sin(x)", 0, a=0, b=sp.pi)
    print(r2)