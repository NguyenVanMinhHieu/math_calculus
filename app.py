"""Web App Giải tích Hình học chuẩn Sư phạm Toán THPT."""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from calculus_engine import CalculusGeometryEngine

# Cấu hình thẩm mỹ toán học toàn cục
plt.rcParams["mathtext.fontset"] = "cm"  # Font Computer Modern chuẩn LaTeX
plt.rcParams["font.family"] = "serif"

st.set_page_config(page_title="Giải tích Hình học", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Biên soạn bởi Thầy giáo Nguyễn Văn Minh Hiếu - Trường THPT TRUNG PHÚ.**")

engine = CalculusGeometryEngine()

st.sidebar.header("Tùy chọn bài toán")
mode = st.sidebar.selectbox(
    "Loại bài toán:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)

col_input, col_view = st.columns([1, 1.25])

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
    st.subheader("🎯 Kết quả & Bản vẽ chuẩn Sư phạm")
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

            st.caption(f"Các mốc phân hoạch: `{res.intervals}`")

            # 1. Sinh mã TikZ
            tikz_code = engine.generate_tikz(f_str, g_str, a=a_val, b=b_val, intervals=res.intervals)

            # 2. Tạo hình vẽ chuẩn Toán học thực tế
            fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=120)

            # Ẩn khung hộp chữ nhật (Top & Right)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Đặt trục Ox, Oy cắt nhau tại gốc tọa độ (0, 0)
            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_color("#222222")
            ax.spines["bottom"].set_color("#222222")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            # Hàm toán học
            x_sym = sp.Symbol("x")
            f_expr = engine._parse_expression(f_str)
            g_expr = engine._parse_expression(g_str)
            f_num = sp.lambdify(x_sym, f_expr, "numpy")
            g_num = sp.lambdify(x_sym, g_expr, "numpy")

            # Xác định dải hiển thị x, y
            all_x = [pt for interval in res.intervals for pt in interval]
            span = max(all_x) - min(all_x)
            margin = max(span * 0.35, 1.0)
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            xs = np.linspace(x_min, x_max, 500)

            ys_f = f_num(xs)
            ys_g = g_num(xs)
            if np.isscalar(ys_f):
                ys_f = np.full_like(xs, ys_f)
            if np.isscalar(ys_g):
                ys_g = np.full_like(xs, ys_g)

            # Tính toán biên độ y
            y_eval = np.concatenate([ys_f, ys_g])
            y_eval = y_eval[np.isfinite(y_eval)]
            y_min, y_max = np.min(y_eval), np.max(y_eval)
            y_margin = max((y_max - y_min) * 0.25, 1.0)
            y_min_plot = min(y_min - y_margin, -0.5)
            y_max_plot = max(y_max + y_margin, 1.0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min_plot, y_max_plot)

            # Mũi tên trục Ox và Oy (Chuẩn SGK)
            ax.plot(x_max, 0, ">k", clip_on=False, markersize=5.5)
            ax.plot(0, y_max_plot, "^k", clip_on=False, markersize=5.5)

            # Ký hiệu chữ O, x, y
            dx = (x_max - x_min) * 0.02
            dy = (y_max_plot - y_min_plot) * 0.03
            ax.text(x_max, -dy * 1.5, r"$x$", fontsize=12, style="italic", ha="center")
            ax.text(-dx * 1.8, y_max_plot, r"$y$", fontsize=12, style="italic", va="center")
            ax.text(-dx * 1.6, -dy * 1.5, r"$O$", fontsize=11, style="italic")

            # Đường lưới phụ mờ
            ax.grid(True, linestyle=":", alpha=0.4, color="#888888")

            # Miền gạch chéo chuẩn SGK (hatch chéo //)
            for sub_a, sub_b in res.intervals:
                x_fill = np.linspace(sub_a, sub_b, 300)
                yf = f_num(x_fill)
                yg = g_num(x_fill)
                if np.isscalar(yf):
                    yf = np.full_like(x_fill, yf)
                if np.isscalar(yg):
                    yg = np.full_like(x_fill, yg)

                ax.fill_between(
                    x_fill,
                    yf,
                    yg,
                    facecolor="#e8f1f9",
                    edgecolor="#2a6f97",
                    hatch="///",
                    linewidth=0.6,
                    alpha=0.9,
                )

            # Vẽ các đường cong đồ thị
            ax.plot(xs, ys_f, label=rf"$y = {sp.latex(f_expr)}$", color="#005f73", linewidth=1.8)
            ax.plot(xs, ys_g, label=rf"$y = {sp.latex(g_expr)}$", color="#ca6702", linestyle="--", linewidth=1.8)

            # Vẽ đường nét đứt dóng tọa độ từ các điểm mốc
            unique_pts = sorted(list(set(all_x)))
            for pt in unique_pts:
                f_val = float(f_expr.subs(x_sym, pt).evalf())
                g_val = float(g_expr.subs(x_sym, pt).evalf())
                target_y = f_val if abs(f_val) > abs(g_val) else g_val

                # Đường dóng xuống Ox
                ax.plot([pt, pt], [0, target_y], color="#444444", linestyle=":", linewidth=0.9)
                # Chấm tròn giao điểm
                ax.plot(pt, target_y, marker="o", markersize=4, color="#1d3557")
                # Đánh dấu nhãn số trên trục Ox
                if abs(pt) > 1e-4:
                    ax.text(pt, -dy * 1.5, f"{pt:g}", fontsize=9, ha="center")

            ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

            # Xuất Vector SVG và PNG
            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            # Hiển thị giao diện Tabs
            tab_graph, tab_tikz = st.tabs(["📊 Đồ thị SGK (Vector SVG)", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_graph:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button(
                        label="📥 Tải ảnh SVG (In đề thi siêu nét)",
                        data=svg_data,
                        file_name="hinh_sgk.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )
                with c_btn2:
                    st.download_button(
                        label="📥 Tải ảnh PNG (Chèn Word 300 DPI)",
                        data=png_data,
                        file_name="hinh_sgk.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            with tab_tikz:
                st.info("💡 Mã TikZ với hệ trục chuẩn SGK, có thể dán trực tiếp vào đề thi LaTeX:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
