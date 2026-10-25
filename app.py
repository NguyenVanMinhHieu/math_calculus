"""Web App Giải tích Hình học THPT - Khối tròn xoay chuẩn mực Sách giáo khoa."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from calculus_engine import CalculusGeometryEngine

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

st.set_page_config(page_title="Giải tích Hình học THPT", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Thầy giáo Nguyễn Văn Minh Hiếu - THPT TRUNG PHÚ**")

engine = CalculusGeometryEngine()

def preprocess_math_str(s: str) -> str:
    """Tự động chuyển đổi: 2x -> 2*x, x^3 -> x**3."""
    if not s or s.strip() == "0":
        return "0"
    s = s.strip().replace("^", "**")
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'(\))(\()', r'\1*\2', s)
    return s

st.sidebar.header("⚙️ Tùy chọn bài toán")
mode = st.sidebar.radio(
    "Dạng toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)
is_volume = "Thể tích" in mode

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Ví dụ mẫu SGK:**")
if st.sidebar.button("📌 Khối tròn xoay hình chuông"):
    st.session_state["f_input"] = "1 + 0.3*(x-1)**2"
    st.session_state["g_input"] = "0"
    st.session_state["use_bounds"] = True
    st.session_state["a_val"] = 0.0
    st.session_state["b_val"] = 3.0
if st.sidebar.button("📌 y = x^3 và y = 2x"):
    st.session_state["f_input"] = "x^3"
    st.session_state["g_input"] = "2x"
    st.session_state["use_bounds"] = True
    st.session_state["a_val"] = -1.0
    st.session_state["b_val"] = 2.0

if "f_input" not in st.session_state:
    st.session_state["f_input"] = "1 + 0.3*(x-1)**2"
if "g_input" not in st.session_state:
    st.session_state["g_input"] = "0"
if "use_bounds" not in st.session_state:
    st.session_state["use_bounds"] = True
if "a_val" not in st.session_state:
    st.session_state["a_val"] = 0.0
if "b_val" not in st.session_state:
    st.session_state["b_val"] = 3.0

col_input, col_view = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("📝 Dữ liệu đầu vào")
    f_raw = st.text_input("Hàm số f(x):", value=st.session_state["f_input"])
    g_raw = st.text_input("Hàm số g(x) (trục hoành = 0):", value=st.session_state["g_input"])

    try:
        f_clean = preprocess_math_str(f_raw)
        g_clean = preprocess_math_str(g_raw)
        f_preview = engine._parse_expression(f_clean)
        g_preview = engine._parse_expression(g_clean)
        st.info(f"Xem trước: $f(x) = {sp.latex(f_preview)}$ và $g(x) = {sp.latex(g_preview)}$")
    except Exception:
        f_clean, g_clean = f_raw, g_raw

    use_custom_bounds = st.checkbox("Chỉ định cận [a, b] thủ công", value=st.session_state["use_bounds"])
    a_val: float | None = None
    b_val: float | None = None

    if use_custom_bounds:
        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Cận dưới (a):", value=float(st.session_state["a_val"]))
        with c2:
            b_val = st.number_input("Cận trên (b):", value=float(st.session_state["b_val"]))

    btn_calc = st.button("🚀 Bắt đầu tính toán & Dựng hình", type="primary", use_container_width=True)

with col_view:
    title_res = "🎯 Khối tròn xoay chuẩn SGK" if is_volume else "🎯 Hình phẳng chuẩn Đề thi"
    st.subheader(title_res)
    if btn_calc or "has_run" in st.session_state:
        st.session_state["has_run"] = True
        try:
            x_sym = engine.var
            f_expr = engine._parse_expression(f_clean)
            g_expr = engine._parse_expression(g_clean)

            if not is_volume:
                res = engine.calculate_area(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Diện tích hình phẳng!")
                st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
            else:
                res = engine.calculate_rotation_volume_ox(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Thể tích khối tròn xoay quanh Ox!")
                st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

            # Lấy cận mút chính xác
            all_pts = [pt for interval in res.intervals for pt in interval]
            x_a, x_b = min(all_pts), max(all_pts)
            label_a = f"{x_a:g}"
            label_b = f"{x_b:g}"
            if res.exact_intervals:
                label_a = sp.latex(res.exact_intervals[0][0])
                label_b = sp.latex(res.exact_intervals[-1][1])

            # Sinh mã TikZ
            tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals, exact_intervals=res.exact_intervals, is_volume=is_volume)

            # ==========================================
            # DỰNG HÌNH CHUẨN SÁCH GIÁO KHOA
            # ==========================================
            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            f_num = sp.lambdify(x_sym, f_expr, modules=["numpy"])
            span = max(x_b - x_a, 1.0)
            x_min = x_a - span * 0.3
            x_max = x_b + span * 0.35

            xs_curve = np.linspace(x_a, x_b, 300)
            ys_upper = np.abs(np.asarray(f_num(xs_curve), dtype=float))
            if ys_upper.ndim == 0: ys_upper = np.full_like(xs_curve, ys_upper)
            ys_lower = -ys_upper

            max_r = max(np.max(ys_upper), 1.0)
            y_lim = max_r * 1.4
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-y_lim, y_lim)

            if is_volume:
                # 1. Trục Ox: đoạn [a, b] bên trong vẽ nét đứt
                ax.plot([x_min, x_a], [0, 0], color="black", linewidth=1.2)
                ax.plot([x_a, x_b], [0, 0], color="black", linestyle="--", linewidth=1.2)
                ax.plot([x_b, x_max], [0, 0], color="black", linewidth=1.2)
                ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
                ax.text(x_max, -y_lim * 0.08, r"$x$", fontsize=12, style="italic")

                # Chấm tròn mốc a và b
                ax.plot(x_a, 0, "ko", markersize=4.5)
                ax.plot(x_b, 0, "ko", markersize=4.5)
                ax.text(x_a, -y_lim * 0.14, rf"${label_a}$", fontsize=11, ha="center")
                ax.text(x_b, -y_lim * 0.14, rf"${label_b}$", fontsize=11, ha="center")

                # 2. Miền phẳng f(x) ban đầu gạch sọc hồng/tím chuẩn SGK
                ax.fill_between(xs_curve, ys_upper, 0, facecolor="#fce4ec", edgecolor="#c2185b", hatch="///", linewidth=0.6, alpha=0.9)

                # Hai đoạn dóng thẳng đứng nét đứt
                fa = float(ys_upper[0])
                fb = float(ys_upper[-1])
                ax.plot([x_a, x_a], [0, fa], color="#c2185b", linestyle="--", linewidth=1.1)
                ax.plot([x_b, x_b], [0, fb], color="#c2185b", linestyle="--", linewidth=1.1)

                # 3. Hai đường sinh biên: Trên f(x) và dưới đối xứng -f(x)
                ax.plot(xs_curve, ys_upper, color="#005f73", linewidth=2.0)
                ax.plot(xs_curve, ys_lower, color="#005f73", linewidth=2.0)

                # 4. Hai Elip tại a và b
                rx = span * 0.045
                # Tại a: nửa trong nét đứt, nửa ngoài nét liền
                t_in = np.linspace(-np.pi/2, np.pi/2, 60)
                t_out = np.linspace(np.pi/2, 3*np.pi/2, 60)
                ax.plot(x_a + rx * np.cos(t_in), fa * np.sin(t_in), color="#005f73", linestyle="--", linewidth=1.1)
                ax.plot(x_a + rx * np.cos(t_out), fa * np.sin(t_out), color="#005f73", linestyle="-", linewidth=1.8)

                # Tại b: toàn bộ Elip nét liền
                t_full = np.linspace(0, 2*np.pi, 120)
                ax.plot(x_b + rx * np.cos(t_full), fb * np.sin(t_full), color="#005f73", linestyle="-", linewidth=1.8)

                # Nhãn f(x)
                mid_idx = len(xs_curve) // 2
                ax.text(xs_curve[mid_idx], ys_upper[mid_idx] * 1.12, r"$f(x)$", fontsize=12, style="italic", ha="center")
            else:
                # Hình 2D phẳng
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_visible(True)
                ax.spines["left"].set_position(("data", 0))
                ax.spines["bottom"].set_position(("data", 0))
                ax.plot(xs_curve, ys_upper, color="#005f73", linewidth=2.0, label=r"$y = f(x)$")
                ax.fill_between(xs_curve, ys_upper, 0, facecolor="#edf4f9", edgecolor="#1d3557", hatch="///", alpha=0.8)
                ax.legend()

            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            # Tabs hiển thị
            tab_img, tab_tikz = st.tabs(["📊 Hình vẽ chuẩn Sách Giáo Khoa", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_img:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button("📥 Tải ảnh SVG (Vector)", data=svg_data, file_name="khoi_tron_xoay_sgk.svg", mime="image/svg+xml", use_container_width=True)
                with c_btn2:
                    st.download_button("📥 Tải ảnh PNG (300 DPI)", data=png_data, file_name="khoi_tron_xoay_sgk.png", mime="image/png", use_container_width=True)

            with tab_tikz:
                st.info("💡 Mã TikZ chuẩn 100% hình minh họa SGK để Thầy copy vào đề thi:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
