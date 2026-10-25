"""Web App Giải tích Hình học với kết xuất SVG và sinh mã TikZ."""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from calculus_engine import CalculusGeometryEngine

st.set_page_config(page_title="Giải tích Hình học", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Biên soạn bởi Thầy giáo Nguyễn Văn Minh Hiếu - Trường THPT TRUNG PHÚ.**")

engine = CalculusGeometryEngine()

st.sidebar.header("Tùy chọn bài toán")
mode = st.sidebar.selectbox(
    "Loại bài toán:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)

col_input, col_view = st.columns([1, 1.2])

with col_input:
    st.subheader("📝 Nhập dữ liệu")
    f_str = st.text_input("Hàm số f(x):", value="x**2")
    g_str = st.text_input("Hàm số g(x) (trục hoành đặt là 0):", value="2*x")

    use_custom_bounds = st.checkbox("Chỉ định cận [a, b] thủ công", value=False)
    a_val: float | None = None
    b_val: float | None = None

    if use_custom_bounds:
        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Cận dưới (a):", value=0.0)
        with c2:
            b_val = st.number_input("Cận trên (b):", value=2.0)

    btn_calc = st.button("🚀 Bắt đầu tính toán", type="primary", use_container_width=True)

with col_view:
    st.subheader("🎯 Kết quả & Bản vẽ")
    if btn_calc:
        try:
            if "Diện tích" in mode:
                res = engine.calculate_area(f_str, g_str, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Diện tích hình phẳng!")
                st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
            else:
                res = engine.calculate_rotation_volume_ox(f_str, g_str, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Thể tích khối tròn xoay quanh Ox!")
                st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

            st.caption(f"Các khoảng tích phân: `{res.intervals}`")

            tikz_code = engine.generate_tikz(f_str, g_str, a=a_val, b=b_val, intervals=res.intervals)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_sym = sp.Symbol("x")
            f_expr = engine._parse_expression(f_str)
            g_expr = engine._parse_expression(g_str)

            f_num = sp.lambdify(x_sym, f_expr, "numpy")
            g_num = sp.lambdify(x_sym, g_expr, "numpy")

            all_x = [pt for interval in res.intervals for pt in interval]
            min_x, max_x = min(all_x) - 1.0, max(all_x) + 1.0
            xs = np.linspace(min_x, max_x, 500)

            ax.plot(xs, f_num(xs), label=f"$f(x) = {sp.latex(f_expr)}$", color="#1f77b4", linewidth=2)
            ax.plot(xs, g_num(xs), label=f"$g(x) = {sp.latex(g_expr)}$", color="#ff7f0e", linestyle="--", linewidth=2)

            for sub_a, sub_b in res.intervals:
                x_fill = np.linspace(sub_a, sub_b, 300)
                ax.fill_between(x_fill, f_num(x_fill), g_num(x_fill), color="#aec7e8", alpha=0.5, hatch="//")

            ax.axhline(0, color="black", linewidth=1.0)
            ax.axvline(0, color="black", linewidth=1.0)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True)

            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            tab_graph, tab_tikz = st.tabs(["📊 Đồ thị (Vector SVG)", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_graph:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button(
                        label="📥 Tải ảnh SVG (Sắc nét vô hạn)",
                        data=svg_data,
                        file_name="hinh_giai_tich.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )
                with c_btn2:
                    st.download_button(
                        label="📥 Tải ảnh PNG (300 DPI)",
                        data=png_data,
                        file_name="hinh_giai_tich.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            with tab_tikz:
                st.info("💡 Thầy có thể sao chép đoạn mã TikZ dưới đây để dán trực tiếp vào đề thi LaTeX:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
