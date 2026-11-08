<div align="center">

# ỨNG DỤNG GIẢI TÍCH SƯ PHẠM: DIỆN TÍCH & THỂ TÍCH TRÒN XOAY
### *Hệ thống phần mềm hỗ trợ Nghiên cứu và Giảng dạy Toán học THPT*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/managed%20by-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Math Engine](https://img.shields.io/badge/CAS-SymPy-brightgreen.svg)](https://www.sympy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Baseline](https://img.shields.io/badge/Release-v1.0.0%20(Locked)-success.svg)]()

**Tác giả:** Thầy giáo **Nguyễn Văn Minh Hiếu**  
*Đơn vị công tác:* **Trường THPT Trung Phú - TP. Hồ Chí Minh**

</div>

---

## 📖 GIỚI THIỆU TỔNG QUAN

**Math Calculus Engine** là ứng dụng Giải tích chuyên sâu được phát triển nhằm chuẩn hóa và trực quan hóa các bài toán **Ứng dụng tích phân tính Diện tích hình phẳng và Thể tích khối tròn xoay** trong chương trình Toán 12 (Chương trình GDPT mới & Đề thi Tốt nghiệp THPT).

Khác biệt với các công cụ tính toán số học thuần túy (numerical calculators), phần mềm vận hành trên nền tảng **Hệ đại số máy tính (CAS - Computer Algebra System)** kết hợp triết lý **Sư phạm giải tích**:
- **Bảo toàn tính chính xác tuyệt đối (Exact Symbolic):** Giữ nguyên các biểu thức phân số tối giản và căn thức vô tỉ (ví dụ $\sqrt{3}, \pi, \dots$), triệt tiêu hiện tượng làm tròn số thập phân xấu xí.
- **Trình bày chuẩn mực 4 bước Sư phạm:** Mô phỏng hoàn hảo các bước lập luận, phá dấu giá trị tuyệt đối và tính tích phân như trong giáo án giảng dạy chính khóa.
- **Hỗ trợ giáo viên soạn giảng:** Tự động xuất biểu thức dạng mã nguồn **LaTeX** và mã đồ họa **TikZ** đồng bộ 1-1 để chèn trực tiếp vào đề thi hay tài liệu bài giảng.

---

## 🎯 TÍNH NĂNG NỔI BẬT

### 1. Tính Diện tích hình phẳng chuẩn Sư phạm
- Tự động nhận diện và giải phương trình hoành độ giao điểm $f(x) - g(x) = 0$.
- Phân hoạch đoạn tích phân $[a; b]$ thành các đoạn con không đổi dấu theo các nghiệm $x_k \in (a; b)$.
- Tìm một nguyên hàm đại số $F(x) = \int [f(x) - g(x)] \, dx$.
- Thế cận, khử dấu giá trị tuyệt đối từng phần và xuất kết quả chính xác dạng căn thức đại số và xấp xỉ số thập phân.

### 2. Tính Thể tích tròn xoay quanh trục $Ox$ (Washer Method)
Khắc phục hoàn toàn các sai lầm kinh điển khi áp dụng máy móc công thức $|V_1 - V_2|$:
- **Bước 1 (Giao điểm bán kính):** Giải phương trình $f^2(x) - g^2(x) = 0 \iff [f(x) - g(x)][f(x) + g(x)] = 0$. Tự động nhận diện cả các nghiệm đối xứng qua trục hoành $f(x) = -g(x)$ (nơi hai đồ thị cắt nhau trong không gian quay).
- **Bước 2 (Công thức tổng quát):** Thiết lập tích phân hình vành khăn chuẩn mực:
  $$V_{Ox} = \pi \int_a^b \left| f^2(x) - g^2(x) \right| \, dx$$
- **Bước 3 (Nguyên hàm bình phương):** Xác định nguyên hàm của hiệu hai bình phương $H(x) = \int [f^2(x) - g^2(x)] \, dx$.
- **Bước 4 (Phân rã tích phân):** Tách tích phân theo các nghiệm giao điểm, nhân hệ số $\pi$ và rút gọn biểu thức đại số chính xác.

### 3. Trực quan hóa hình học & Đồng bộ TikZ
- Vẽ đồ thị 2D với hệ trục tọa độ $Oxy$ chuẩn tỉ lệ, gạch sọc / tô màu miền phẳng giới hạn.
- Vẽ khối tròn xoay 3D trực quan khi quay quanh $Ox$.
- Trình tạo mã **TikZ (LaTeX)** chuẩn hóa với lệnh `\clip` khóa viền, cho phép giáo viên copy-paste trực tiếp vào TeXstudio hoặc Overleaf.

---

## 📐 CƠ SỞ TOÁN HỌC & THUẬT TOÁN

| Tiêu chí | Diện tích Hình phẳng ($S$) | Thể tích Tròn xoay quanh $Ox$ ($V_{Ox}$) |
| :--- | :--- | :--- |
| **Bản chất hình học** | Miền phẳng giới hạn bởi 2 đường cong | Mặt cắt hình vành khăn (Washer) bán kính $R(x), r(x)$ |
| **Phương trình nghiệm** | $f(x) - g(x) = 0$ | $f^2(x) - g^2(x) = 0 \iff f(x) = \pm g(x)$ |
| **Công thức vi phân** | $dS = \|f(x) - g(x)\| \, dx$ | $dV = \pi \left( R^2(x) - r^2(x) \right) dx = \pi \|f^2(x) - g^2(x)\| \, dx$ |
| **Nguyên hàm phụ trợ** | $F(x) = \int [f(x) - g(x)] \, dx$ | $H(x) = \int [f^2(x) - g^2(x)] \, dx$ |
| **Định lượng phân đoạn** | $S = \sum_{k} \|F(x_{k+1}) - F(x_k)\|$ | $V = \pi \sum_{k} \|H(x_{k+1}) - H(x_k)\|$ |

---

## 📂 CẤU TRÚC DỰ ÁN

```text
math-calculus/
├── app.py                  # Giao diện người dùng Streamlit & Điều hướng Sư phạm
├── calculus_engine.py      # Động cơ Giải tích Đại số máy tính (SymPy Core Engine)
├── test_calculus.py        # Bộ kiểm thử tự động (Unit Tests) với Pytest
├── pyproject.toml          # Quản lý cấu hình dự án & Metadata tác giả (PEP 621)
├── uv.lock                 # Khóa toàn vẹn phiên bản dependencies
├── LICENSE                 # Giấy phép bản quyền mã nguồn mở MIT
└── README.md               # Tài liệu hướng dẫn & Bản quyền tác giả