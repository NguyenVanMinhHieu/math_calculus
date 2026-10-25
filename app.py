"""Giao diện Web tương tác tính Toán giải tích hình học."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from calculus_engine import CalculusGeometryEngine

st.set_page_config(page_title="Giải tích Hình học", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("Hỗ trợ nghiên cứu và giảng dạy Toán học - Biên soạn bởi Thầy giáo Nguyễn Văn Minh Hiếu - Trường THPT TRUNG PHÚ.")

# Khởi tạo engine
engine = CalculusGeometryEngine()

# Sidebar: Chọn chế độ
mode = st.sidebar.selectbox(
    "Chọn bài toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Nhập dữ liệu")
    f_str = st.text_input("Hàm số f(x):", value="x**2")
    g_str = st.text_input("Hàm số g(x) (mặc định 0):", value="2*x")
    
    use_custom_bounds = st.checkbox("Chỉ định cận [a, b] thủ công", value=False)
    a_val: float | None = None
    b_val: float | None = None
    
    if use_custom_bounds:
        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Cận dưới (a):", value=0.0)
        with c2:
            b_val = st.number_input("Cận trên (b):", value=2.0)

    btn_calc = st.button("🚀 Bắt đầu tính toán", type="primary")

with col2:
    st.subheader("Kết quả & Đồ thị")
    if btn_calc:
        try:
            if "Diện tích" in mode:
                res = engine.calculate_area(f_str, g_str, a=a_val, b=b_val)
                st.success("Tính toán diện tích thành công!")
                st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
            else:
                res = engine.calculate_rotation_volume_ox(f_str, g_str, a=a_val, b=b_val)
                st.success("Tính toán thể tích quanh Ox thành công!")
                st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

            st.write(f"**Các khoảng phân hoạch:** `{res.intervals}`")

            # Vẽ đồ thị 2D minh họa
            fig, ax = plt.subplots(figsize=(6, 4))
            x_sym = sp.Symbol("x")
            f_expr = engine._parse_expression(f_str)
            g_expr = engine._parse_expression(g_str)

            f_num = sp.lambdify(x_sym, f_expr, "numpy")
            g_num = sp.lambdify(x_sym, g_expr, "numpy")

            # Xác định phạm vi vẽ đồ thị
            all_x = [pt for interval in res.intervals for pt in interval]
            min_x, max_x = min(all_x) - 1.0, max(all_x) + 1.0
            xs = np.linspace(min_x, max_x, 500)

            ax.plot(xs, f_num(xs), label=f"$f(x) = {sp.latex(f_expr)}$", color="royalblue")
            ax.plot(xs, g_num(xs), label=f"$g(x) = {sp.latex(g_expr)}$", color="darkorange", linestyle="--")

            # Tô màu miền diện tích tích phân
            for sub_a, sub_b in res.intervals:
                x_fill = np.linspace(sub_a, sub_b, 200)
                ax.fill_between(x_fill, f_num(x_fill), g_num(x_fill), color="skyblue", alpha=0.4)

            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Lỗi: {e}")