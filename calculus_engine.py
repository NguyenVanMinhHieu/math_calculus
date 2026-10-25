"""Module tính toán Giải tích Hình học: Diện tích hình phẳng và Thể tích tròn xoay.

Được thiết kế tối ưu cho giảng dạy toán học, hỗ trợ cả nghiệm giải tích (SymPy)
và xấp xỉ số học.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import sympy as sp
from sympy.core.expr import Expr

# Cấu hình logging chuẩn mực
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CalculusEngine")


@dataclass(frozen=True)
class MathResult:
    """Dataclass chứa kết quả tính toán hình học giải tích.

    Attributes:
        exact_value: Biểu thức giải tích chính xác (SymPy Expr).
        numeric_value: Giá trị xấp xỉ thực thể dạng float.
        latex_str: Chuỗi mã hóa biểu thức LaTeX phục vụ trình bày bảng/slide.
        intervals: Danh sách các phân hoạch tích phân đã xử lý.
    """

    exact_value: Expr
    numeric_value: float
    latex_str: str
    intervals: List[Tuple[float, float]]

    def __str__(self) -> str:
        return (
            f"Kết quả chính xác: {self.exact_value}\n"
            f"Giá trị xấp xỉ   : {self.numeric_value:.6f}\n"
            f"Mã LaTeX         : {self.latex_str}"
        )


class CalculusGeometryEngine:
    """Engine xử lý các phép toán tích phân ứng dụng trong hình học phẳng và không gian."""

    def __init__(self, variable_name: str = "x") -> None:
        """Khởi tạo engine tính toán.

        Args:
            variable_name: Ký hiệu biến số độc lập (mặc định là 'x').
        """
        self.var: sp.Symbol = sp.Symbol(variable_name, real=True)
        logger.info("Khởi tạo CalculusGeometryEngine với biến độc lập: %s", self.var)

    def _parse_expression(self, expr_input: Union[str, Expr, float, int]) -> Expr:
        """Chuyển đổi dữ liệu đầu vào sang SymPy Expression an toàn.

        Args:
            expr_input: Biểu thức dạng chuỗi hoặc biểu thức toán học.

        Returns:
            Expr: Đối tượng biểu thức toán của SymPy.

        Raises:
            ValueError: Nếu biểu thức không hợp lệ.
        """
        if isinstance(expr_input, Expr):
            return expr_input
        try:
            parsed = sp.sympify(str(expr_input), locals={str(self.var): self.var})
            return parsed
        except (sp.SympifyError, TypeError, SyntaxError) as err:
            logger.error("Lỗi phân tích cú pháp biểu thức: %s", expr_input)
            raise ValueError(f"Biểu thức không hợp lệ: {expr_input}") from err

    def _find_real_roots(
        self,
        diff_expr: Expr,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> List[sp.Rational | sp.Number]:
        """Tìm tất cả các nghiệm thực của phương trình trong đoạn xác định.

        Args:
            diff_expr: Biểu thức f(x) - g(x).
            lower: Biên dưới (tùy chọn).
            upper: Biên trên (tùy chọn).

        Returns:
            List các điểm phân hoạch đã được sắp xếp tăng dần.
        """
        roots_set = sp.solveset(diff_expr, self.var, domain=sp.S.Reals)
        roots: List[Any] = []

        if isinstance(roots_set, sp.FiniteSet):
            roots = list(roots_set)
        else:
            # Fallback nếu giải phương trình siêu việt/lượng giác
            try:
                roots = sp.solve(diff_expr, self.var)
                roots = [r for r in roots if r.is_real]
            except NotImplementedError:
                logger.warning("Không thể tìm nghiệm giải tích hoàn toàn. Dùng nghiệm số.")
                roots = []

        # Lọc nghiệm nằm trong [lower, upper] nếu có chỉ định biên
        filtered_roots: List[Any] = []
        for r in roots:
            val = float(r.evalf())
            if lower is not None and val < lower - 1e-9:
                continue
            if upper is not None and val > upper + 1e-9:
                continue
            filtered_roots.append(r)

        # Sắp xếp các nghiệm tăng dần
        filtered_roots.sort(key=lambda x: float(x.evalf()))
        return filtered_roots

    def calculate_area(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> MathResult:
        """Tính diện tích hình phẳng giới hạn bởi y = f(x), y = g(x) và x = a, x = b.

        Công thức: S = \\int_{a}^{b} |f(x) - g(x)| dx

        Args:
            f_expr: Đường cong thứ nhất.
            g_expr: Đường cong thứ hai (mặc định = 0, tức trục Ox).
            a: Cận dưới. Nếu None, tự động tìm hoành độ giao điểm.
            b: Cận trên. Nếu None, tự động tìm hoành độ giao điểm.

        Returns:
            MathResult: Kết quả dạng tích phân chuẩn.
        """
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff = f - g

        roots = self._find_real_roots(diff, a, b)

        # Xác định các mốc phân hoạch tích phân
        partition: List[Any] = []
        if a is not None:
            partition.append(sp.sympify(a))
        partition.extend([r for r in roots if (a is None or float(r.evalf()) > a) and (b is None or float(r.evalf()) < b)])
        if b is not None:
            partition.append(sp.sympify(b))

        # Khử trùng lặp và sắp xếp
        partition = sorted(list({sp.nsimplify(p, rational=True) for p in partition}), key=lambda x: float(x.evalf()))

        if len(partition) < 2:
            raise ValueError(
                "Không đủ cận để tính tích phân. Cần chỉ định cận [a, b] "
                "hoặc đồ thị phải có ít nhất 2 giao điểm thực."
            )

        total_area: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        # Tích phân từng đoạn để khử dấu tuyệt đối |f(x) - g(x)|
        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            sub_integral = sp.integrate(diff, (self.var, sub_a, sub_b))
            total_area += sp.Abs(sub_integral)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_area = sp.simplify(total_area)
        numeric_val = float(total_area.evalf())

        return MathResult(
            exact_value=total_area,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_area),
            intervals=intervals_used,
        )

    def calculate_rotation_volume_ox(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> MathResult:
        """Tính thể tích khối tròn xoay khi quay quanh trục Ox.

        Công thức: V_Ox = \\pi \\int_{a}^{b} |f(x)^2 - g(x)^2| dx

        Args:
            f_expr: Đường biên trên.
            g_expr: Đường biên dưới (mặc định = 0).
            a: Cận dưới.
            b: Cận trên.

        Returns:
            MathResult: Kết quả thể tích tròn xoay.
        """
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff_sq = sp.simplify(f**2 - g**2)

        roots = self._find_real_roots(diff_sq, a, b)

        partition: List[Any] = []
        if a is not None:
            partition.append(sp.sympify(a))
        partition.extend([r for r in roots if (a is None or float(r.evalf()) > a) and (b is None or float(r.evalf()) < b)])
        if b is not None:
            partition.append(sp.sympify(b))

        partition = sorted(list({sp.nsimplify(p, rational=True) for p in partition}), key=lambda x: float(x.evalf()))

        if len(partition) < 2:
            raise ValueError("Không đủ cận xác định để tính thể tích quay quanh Ox.")

        total_integral: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            sub_val = sp.integrate(diff_sq, (self.var, sub_a, sub_b))
            total_integral += sp.Abs(sub_val)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_volume = sp.simplify(sp.pi * total_integral)
        numeric_val = float(total_volume.evalf())

        return MathResult(
            exact_value=total_volume,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_volume),
            intervals=intervals_used,
        )

    def calculate_rotation_volume_oy(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: float = 0.0,
        b: float = 1.0,
    ) -> MathResult:
        """Tính thể tích khối tròn xoay khi quay quanh trục Oy (PP Ống trụ - Shell method).

        Công thức: V_Oy = 2\\pi \\int_{a}^{b} x |f(x) - g(x)| dx (với 0 <= a < b)

        Args:
            f_expr: Đường cong thứ nhất.
            g_expr: Đường cong thứ hai.
            a: Cận dưới (yêu cầu a >= 0).
            b: Cận trên.

        Returns:
            MathResult: Kết quả thể tích tròn xoay quanh Oy.
        """
        if a < 0 or b < 0:
            raise ValueError("Phương pháp Shell yêu cầu dải tích phân x >= 0.")

        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff = f - g

        roots = self._find_real_roots(diff, a, b)

        partition = [sp.sympify(a)]
        partition.extend([r for r in roots if float(r.evalf()) > a and float(r.evalf()) < b])
        partition.append(sp.sympify(b))
        partition = sorted(list({sp.nsimplify(p, rational=True) for p in partition}), key=lambda x: float(x.evalf()))

        total_integral: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            kernel = self.var * diff
            sub_val = sp.integrate(kernel, (self.var, sub_a, sub_b))
            total_integral += sp.Abs(sub_val)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_volume = sp.simplify(2 * sp.pi * total_integral)
        numeric_val = float(total_volume.evalf())

        return MathResult(
            exact_value=total_volume,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_volume),
            intervals=intervals_used,
        )

    def generate_tikz(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        r"""Tự động sinh mã nguồn TikZ chuẩn LaTeX."""
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        f_tikz = str(f).replace(str(self.var), r"\x").replace("**", "^")
        g_tikz = str(g).replace(str(self.var), r"\x").replace("**", "^")

        if intervals and len(intervals) > 0:
            all_pts = [p for inv in intervals for p in inv]
            x_min = min(all_pts) - 1.0
            x_max = max(all_pts) + 1.0
        else:
            x_min, x_max = -2.0, 3.0

        shading_blocks = []
        if intervals:
            for sub_a, sub_b in intervals:
                shading_blocks.append(
                    f"    % Miền gạch chéo [{sub_a:.2f}, {sub_b:.2f}]\n"
                    f"    \\fill[pattern=north east lines, pattern color=blue!40] "
                    f"plot[domain={sub_a:.2f}:{sub_b:.2f}, samples=100] (\\x, {{{f_tikz}}}) -- "
                    f"plot[domain={sub_b:.2f}:{sub_a:.2f}, samples=100] (\\x, {{{g_tikz}}}) -- cycle;"
                )
        shading_code = "\n".join(shading_blocks)

        return (
            "% Yêu cầu: \\usepackage{tikz} và \\usetikzlibrary{patterns}\n"
            "\\begin{tikzpicture}[>=stealth, scale=1.0]\n"
            f"    \\draw[->] ({x_min:.1f}, 0) -- ({x_max + 0.5:.1f}, 0) node[below] {{$x$}};\n"
            f"    \\draw[->] (0, -1.5) -- (0, 5.0) node[left] {{$y$}};\n"
            f"    \\node[below left] at (0, 0) {{$O$}};\n\n"
            f"{shading_code}\n\n"
            f"    \\draw[thick, blue, domain={x_min:.2f}:{x_max:.2f}, samples=150] "
            f"plot (\\x, {{{f_tikz}}}) node[right] {{$y = {sp.latex(f)}$}};\n"
            f"    \\draw[thick, orange, dashed, domain={x_min:.2f}:{x_max:.2f}, samples=150] "
            f"plot (\\x, {{{g_tikz}}}) node[right] {{$y = {sp.latex(g)}$}};\n"
            "\\end{tikzpicture}"
        )
