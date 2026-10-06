# 🚀 Hướng Dẫn Chạy & Hoàn Thành Bài Lab Trên Google Colab

Tài liệu này hướng dẫn chi tiết cách mở, chạy, hoàn thành các bài tập và lưu kết quả bài lab **Kalman Filter Pilot** trên **Google Colab**.

---

## 1. Cách mở Notebook trên Google Colab

Bạn có 2 cách rất nhanh để mở:

### Cách 1: Mở trực tiếp 1-click từ GitHub (Khuyên dùng)
Nhấp trực tiếp vào liên kết bên dưới để Colab tải thẳng notebook từ repository của bạn:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenquangdao2004-glitch/K4-Track4-Day5-NguyenQuangDao-2A202602394-Kalman-Filter-Pilot/blob/main/Lab/kalman_fusion_lab_STUDENT.ipynb)

*URL trực tiếp:* `https://colab.research.google.com/github/nguyenquangdao2004-glitch/K4-Track4-Day5-NguyenQuangDao-2A202602394-Kalman-Filter-Pilot/blob/main/Lab/kalman_fusion_lab_STUDENT.ipynb`

### Cách 2: Upload thủ công file `.ipynb`
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **Upload** (Tải lên).
3. Chọn file `Lab/kalman_fusion_lab_STUDENT.ipynb` từ thư mục trên máy tính của bạn.

> [!NOTE]
> **Về dữ liệu & thư viện:** Toàn bộ dữ liệu của bài lab (quỹ đạo xe, cảm biến giả lập, nhiễu) đều được sinh trực tiếp bằng thuật toán trong RAM của Python. Google Colab đã cài sẵn `numpy`, `matplotlib`, `scipy` nên bạn **không cần upload thêm bất kỳ file dữ liệu nào** và **không cần cài đặt gì thêm**.

---

## 2. Kế hoạch làm bài trên Colab (120 phút)

| Thời gian | Mục | Nhiệm vụ | Điểm |
|:---|:---|:---|:---:|
| 5 phút | **Phần 0** | Chạy cell Setup | 0 |
| 25 phút | **Phần 1 – 3** | Chạy & đọc hiểu: Trung bình trượt $\rightarrow$ Kalman 1D | 0 |
| 10 phút | **Phần 4** | Chạy & đọc hiểu: Chỉ số NIS kỳ vọng $\approx 1-2$ | 0 |
| 25 phút | **Phần 5** | Tự viết code Bài **5.1** và **5.2** | **25** |
| 12 phút | **Phần 6** | Tự viết code Bài **6.1** | **15** |
| 10 phút | **Phần 7** | Tự viết code Bài **7.1** | **10** |
| 25 phút | **Phần 9** | Nhiệm vụ Lynx-07: Chẩn đoán lỗi, chọn cách sửa, viết báo cáo | **50** |
| 8 phút | **Phần 8** *(Bonus)* | Bộ lọc EKF phi tuyến cho Radar | *+4/5* |

---

## 3. Hướng dẫn & Lời giải tham khảo cho các bài tập tự code

### ✏️ Bài 5.1: `make_F(dt)` và `make_H()` (10 điểm)
Mô hình vận tốc không đổi (Constant Velocity) 4 chiều $x = [p_x, p_y, v_x, v_y]^\top$:

```python
def make_F(dt):
    """Ma trận chuyển trạng thái 4x4:
    x_k = x_{k-1} + vx * dt
    y_k = y_{k-1} + vy * dt
    vx_k = vx_{k-1}
    vy_k = vy_{k-1}
    """
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F

def make_H():
    """Ma trận đo lường 2x4 trích xuất vị trí [x, y] từ trạng thái [x, y, vx, vy]"""
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H
```

---

### ✏️ Bài 5.2: Class `KalmanFilter` (15 điểm)
Thực hiện các công thức đại số tuyến tính của bộ lọc Kalman:

```python
class KalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self, F, Q):
        # x^- = F * x
        self.x = F @ self.x
        # P^- = F * P * F.T + Q
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        # Innovation (phần dư)
        y = z - H @ self.x
        # Innovation covariance
        S = H @ self.P @ H.T + R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # Cập nhật trạng thái
        self.x = self.x + K @ y
        # Cập nhật hiệp phương sai
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        return y, S, K
```

---

### ✏️ Bài 6.1: Vòng lặp hợp nhất đa cảm biến `run_fusion` (15 điểm)
Hợp nhất các phép đo bất đồng bộ theo mốc thời gian:

```python
def run_fusion(meas, q, x0, P0, t0=0.0):
    kf = KalmanFilter(x0, P0)
    t_prev = t0
    log = []
    
    for ts, name, z, H, R in meas:
        dt = ts - t_prev
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        kf.update(z, H, R)
        t_prev = max(t_prev, ts)
        log.append((ts, kf.x.copy(), kf.P.copy()))
        
    return log
```

---

### ✏️ Bài 7.1: Cổng lọc ngoại lai Chi-Square `gated_update` (10 điểm)
Loại trừ các phép đo bất thường (ghost/outlier) bằng khoảng cách Mahalanobis bình phương:

```python
def gated_update(kf, z, H, R, p=0.99):
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    # d^2 = y^T * S^{-1} * y
    d2 = float(y @ np.linalg.solve(S, y))
    
    # Ngưỡng Chi-Square theo bậc tự do df = len(z)
    threshold = chi2.ppf(p, df=len(z))
    if d2 > threshold:
        return False   # Từ chối outlier, không cập nhật bộ lọc
    
    kf.update(z, H, R)
    return True
```

---

### 🎯 Phần 9: Nhiệm vụ chẩn đoán xe tự hành Lynx-07 (50 điểm)

#### Bước 1: Đổi `STUDENT_ID`
Trong cell khởi tạo Phần 9, đổi:
```python
STUDENT_ID = "2A202602394"   # Điền MSSV hoặc tên của bạn!
```
*(Nếu giữ nguyên câu mặc định thì tự động nhận 0 điểm toàn bộ Phần 9).*

#### Bước 2: Chạy cell `diagnose(mission)` và đọc chỉ số
Quan sát các dòng in ra cho **GPS** và **UWB**:
- `mean(NIS)`
- `median(NIS)`
- `mean(residual)`

#### Bước 3: Nhận diện lỗi và chọn cách sửa theo bảng quy tắc

| Loại lỗi (`MY_DIAGNOSIS_TYPE`) | Dấu hiệu quan sát | Cách sửa (`FIX_METHOD`) |
|:---|:---|:---|
| **`bias`** (Lệch hằng số) | `mean(residual)` bị lệch xa khỏi 0 (ví dụ $> 2-4$ m). | `bias` (Trừ residual trung bình) |
| **`underrated_noise`** (Nhiễu lớn hơn khai báo) | Cả `mean(NIS)` và `median(NIS)` đều vọt lên rất cao so với mức kỳ vọng 2 (ví dụ $> 10-40$). | `inflate_R` (Tăng ma trận hiệp phương sai $R$) |
| **`outlier_burst`** (Xung nhiễu đột biến) | `mean(NIS)` rất cao, nhưng `median(NIS)` vẫn bình thường ($\approx 2$). | `gate` (Dùng cổng Chi-Square loại bỏ) |

#### Bước 4: Khai báo 4 biến chẩn đoán (30 điểm)
Ví dụ (nếu phát hiện GPS bị nhiễu đánh giá thấp):
```python
MY_DIAGNOSIS_SENSOR = "GPS"              # "GPS" hoặc "UWB"
MY_DIAGNOSIS_TYPE   = "underrated_noise" # "bias", "underrated_noise", hoặc "outlier_burst"
FIX_SENSOR          = "GPS"              # "GPS" hoặc "UWB"
FIX_METHOD          = "inflate_R"        # "bias", "inflate_R", hoặc "gate"
```
Chạy tiếp cell `check_diagnosis_vars()` và cell `check_mission_filter()` $\rightarrow$ Đạt yêu cầu khi hiện `✅ Bộ lọc chạy hợp lệ (pooled mean NIS < 8)`.

#### Bước 5: Điền Báo cáo Lynx-07 vào ô Markdown (20 điểm chấm tay)
Tìm cell Markdown có biểu tượng `✍️ Báo cáo Lynx-07` và điền:
1. **Bằng chứng:** Điền chính xác các giá trị `mean(NIS)`, `median(NIS)`, và `residual` mà notebook của bạn vừa in ra.
2. **Cách sửa:** Ghi rõ bạn chọn cách nào trên cảm biến nào, và giải thích vì sao 2 cách còn lại không phù hợp (ví dụ: residual gần 0 nên không phải bias; median NIS cũng cao nên không phải outlier đơn thuần).
3. **Độ tin cậy cuối:** Lấy giá trị $1\sigma = \sqrt{P_{xx} + P_{yy}}$ ở bước cuối cùng từ ma trận $P$ (in ra từ cell plot), và bán kính 95% ($\approx 2\sigma$).
4. **Hạn chế:** Nêu một tình huống thực tế mà giải pháp này sẽ thất bại (ví dụ: nhiễu thay đổi theo thời gian, hoặc cả 2 cảm biến cùng gặp lỗi đồng thời).

---

### 🌟 Phần 8: Bonus EKF (Tối đa +5 điểm thưởng)
Nếu còn thời gian, hoàn thành bài **8.1**:
```python
def h_rb(x, s):
    dx, dy = x[0] - s[0], x[1] - s[1]
    return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

def H_rb(x, s):
    dx, dy = x[0] - s[0], x[1] - s[1]
    r2 = dx ** 2 + dy ** 2
    r = np.sqrt(r2)
    H = np.zeros((2, 4))
    H[0, 0], H[0, 1] = dx / r, dy / r
    H[1, 0], H[1, 1] = -dy / r2, dx / r2
    return H
```

---

## 4. Cách lưu và nộp bài sau khi làm xong trên Colab

Khi đã hoàn thành và tất cả các cell kiểm tra đều báo `✅ Exercise ... passed`:

### Cách A: Tải file `.ipynb` về máy rồi commit (Khuyên dùng)
1. Trên menu Colab, chọn **File** $\rightarrow$ **Download** $\rightarrow$ **Download .ipynb**.
2. Di chuyển file vừa tải về đè lên file:
   `d:\AI20K\Lab\K4-Track4-Day5-NguyenQuangDao-2A202602394-Kalman-Filter-Pilot\Lab\kalman_fusion_lab_STUDENT.ipynb`
3. Mở terminal tại thư mục dự án và đẩy lên GitHub:
   ```bash
   git add Lab/kalman_fusion_lab_STUDENT.ipynb
   git commit -m "feat: complete kalman filter lab"
   git push origin main
   ```

### Cách B: Lưu trực tiếp lên GitHub từ Colab
1. Trên menu Colab, chọn **File** $\rightarrow$ **Save a copy in GitHub** (Lưu một bản sao vào GitHub).
2. Chọn repository: `nguyenquangdao2004-glitch/K4-Track4-Day5-NguyenQuangDao-2A202602394-Kalman-Filter-Pilot`.
3. File path: `Lab/kalman_fusion_lab_STUDENT.ipynb`.
4. Bấm **OK** để lưu trực tiếp vào nhánh `main`.
