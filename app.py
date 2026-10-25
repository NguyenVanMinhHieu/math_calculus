"""Web App Giải tích Hình học chuẩn Sư phạm Toán THPT - Tối ưu Layout & Typography."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from calculus_engine import CalculusGeometryEngine

# Cấu hình thẩm mỹ Toán học
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

st.set_page_config(page_title="Giải tích Hình học THPT", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Thầy giáo Nguyễn Văn Minh Hiếu - THPT TRUNG PHÚ**")

engine = CalculusGeometryEngine()

def preprocess_math_str(s: str) -> str:
    """Cho phép gõ tự nhiên: 2x -> 2*x, x^3 -> x**3."""
    if not s or s.strip() == "0":
        return "0"
    s = s.strip()
    s = s.replace("^", "**")
    # Tự động chèn dấu nhân: số đứng trước chữ cái (2x -> 2*x)
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    # Tự động chèn dấu nhân: đóng ngoặc trước mở ngoặc (x+1)(x-2) -> (x+1)*(x-2)
    s = re.sub(r'(\))(\()', r'\1*\2', s)
    return s

st.sidebar.header("⚙️ Tùy chọn bài toán")
mode = st.sidebar.radio(
    "Dạng toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Gợi ý bài toán hay:**")
if st.sidebar.button("📌 Bậc 3 và Tiếp tuyến"):
    st.session_state["f_input"] = "x^3 - 2*x^2 + 1"
    st.session_state["g_input"] = "2x"
    st.session_state["use_bounds"] = False
if st.sidebar.button("📌 Parabol & Trục hoành"):
    st.session_state["f_input"] = "4 - x^2"
    st.session_state["g_input"] = "0"
    st.session_state["use_bounds"] = False

# Thiết lập giá trị mặc định vào session state
if "f_input" not in st.session_state:
    st.session_state["f_input"] = "x^3 - 2*x^2 + 1"
if "g_input" not in st.session_state:
    st.session_state["g_input"] = "2x"
if "use_bounds" not in st.session_state:
    st.session_state["use_bounds"] = False

# BỐ CỤC CHÍNH 2 CỘT CÂN ĐỐI
col_input, col_view = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("📝 Dữ liệu đầu vào")
    
    f_raw = st.text_input("Hàm số f(x):", value=st.session_state["f_input"])
    g_raw = st.text_input("Hàm số g(x) (hoặc trục hoành = 0):", value=st.session_state["g_input"])

    # Hiển thị trực quan ngay công thức đang gõ (Live Preview)
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
            a_val = st.number_input("Cận dưới (a):", value=-1.0)
        with c2:
            b_val = st.number_input("Cận trên (b):", value=2.6)

    btn_calc = st.button("🚀 Bắt đầu tính toán & Dựng hình", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        """
        **Hướng dẫn thao tác:**
        * Nhập đa thức, căn thức hoặc lượng giác dễ dàng (`x^2`, `2x`, `sin(x)`).
        * Không cần gõ `2*x`, hệ thống tự nhận diện phép nhân ẩn.
        * Đồ thị SVG xuất ra có thể phóng to tùy ý không bị vỡ ảnh.
        """
    )

with col_view:
    st.subheader("🎯 Bản vẽ Sư phạm & Lời giải")
    if btn_calc or "has_run" in st.session_state:
        st.session_state["has_run"] = True
        try:
            x_sym = engine.var
            f_expr = engine._parse_expression(f_clean)
            g_expr = engine._parse_expression(g_clean)

            if "Diện tích" in mode:
                res = engine.calculate_area(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Diện tích hình phẳng!")
                st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
            else:
                res = engine.calculate_rotation_volume_ox(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Thể tích khối tròn xoay quanh Ox!")
                st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

            # Trình bày phân hoạch sư phạm
            parts_latex = []
            for sub_a, sub_b in res.intervals:
                parts_latex.append(rf"\int_{{{sub_a:.2f}}}^{{{sub_b:.2f}}} |f(x) - g(x)| dx")
            st.markdown("**Công thức tích phân từng đoạn:**")
            st.latex("S = " + " + ".join(parts_latex))

            # Sinh mã TikZ
            tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals)

            # VẼ ĐỒ THỊ CHUẨN MỰC
            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            f_num = sp.lambdify(x_sym, f_expr, modules=["numpy"])
            g_num = sp.lambdify(x_sym, g_expr, modules=["numpy"])

            all_x = [pt for interval in res.intervals for pt in interval]
            span = max(all_x) - min(all_x)
            margin_x = max(span * 0.25, 0.8)
            x_min, x_max = min(all_x) - margin_x, max(all_x) + margin_x
            xs = np.linspace(x_min, x_max, 500)

            # Thu thập giá trị y TRONG MIỀN TÍCH PHÂN để lấy nét trọng tâm
            x_focus = np.linspace(min(all_x), max(all_x), 300)
            yf_focus = np.asarray(f_num(x_focus), dtype=float)
            yg_focus = np.asarray(g_num(x_focus), dtype=float)
            if yf_focus.ndim == 0: yf_focus = np.full_like(x_focus, yf_focus)
            if yg_focus.ndim == 0: yg_focus = np.full_like(x_focus, yg_focus)

            y_f_min, y_f_max = min(np.min(yf_focus), np.min(yg_focus)), max(np.max(yf_focus), np.max(yg_focus))
            margin_y = max((y_f_max - y_f_min) * 0.35, 1.2)
            y_min_plot = min(y_f_min - margin_y, -0.6)
            y_max_plot = max(y_f_max + margin_y, 1.0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min_plot, y_max_plot)

            # Vẽ mũi tên trục
            ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
            ax.plot(0, y_max_plot, "^k", clip_on=False, markersize=5)

            # Chữ O, x, y tránh đè nhau
            dx = (x_max - x_min) * 0.02
            dy = (y_max_plot - y_min_plot) * 0.035
            ax.text(x_max, -dy * 1.3, r"$x$", fontsize=12, style="italic", ha="center")
            ax.text(-dx * 1.8, y_max_plot, r"$y$", fontsize=12, style="italic", va="center")
            ax.text(-dx * 1.8, -dy * 1.3, r"$O$", fontsize=11, style="italic")

            ax.grid(True, linestyle=":", alpha=0.35, color="#888888")

            # Tô miền diện tích gạch sọc
            for sub_a, sub_b in res.intervals:
                x_fill = np.linspace(sub_a, sub_b, 250)
                yf = np.asarray(f_num(x_fill), dtype=float)
                yg = np.asarray(g_num(x_fill), dtype=float)
                if yf.ndim == 0: yf = np.full_like(x_fill, yf)
                if yg.ndim == 0: yg = np.full_like(x_fill, yg)

                ax.fill_between(
                    x_fill, yf, yg,
                    facecolor="#edf4f9", edgecolor="#1d3557",
                    hatch="///", linewidth=0.5, alpha=0.9
                )

            # Vẽ đường cong
            ys_f = np.asarray(f_num(xs), dtype=float)
            ys_g = np.asarray(g_num(xs), dtype=float)
            if ys_f.ndim == 0: ys_f = np.full_like(xs, ys_f)
            if ys_g.ndim == 0: ys_g = np.full_like(xs, ys_g)

            ax.plot(xs, ys_f, label=rf"$y = {sp.latex(f_expr)}$", color="#005f73", linewidth=2.0)
            ax.plot(xs, ys_g, label=rf"$y = {sp.latex(g_expr)}$", color="#ca6702", linestyle="--", linewidth=1.8)

            # Đường dóng và nhãn số sạch sẽ
            unique_pts = sorted(list(set(all_x)))
            for pt in unique_pts:
                f_val = float(np.asarray(f_num(pt)).item())
                g_val = float(np.asarray(g_num(pt)).item())
                target_y = f_val if abs(f_val) > abs(g_val) else g_val

                ax.plot([pt, pt], [0, target_y], color="#555555", linestyle=":", linewidth=0.9)
                ax.plot(pt, target_y, marker="o", markersize=4, color="#1d3557")
                
                # Chỉ in nhãn nếu không quá gần gốc O (tránh đè chữ O)
                if abs(pt) > 0.18:
                    ax.text(pt, -dy * 1.3, f"{pt:.2f}", fontsize=9, ha="center", color="#111111")

            ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=9.5)

            # Xuất dữ liệu ảnh
            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            # Tabs hiển thị
            tab_graph, tab_tikz = st.tabs(["📊 Đồ thị SGK (Vector SVG)", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_graph:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button(
                        label="📥 Tải ảnh SVG (In đề sắc nét)",
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
                st.info("💡 Mã TikZ chuẩn SGK (đã có đường dóng tọa độ):")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
