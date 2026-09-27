"""Rescue + thoái lui nearest-centroid cho GB-TSVM ở độ thuần thấp.

Bối cảnh: khi độ thuần thấp và nhiễu bất đối xứng, gộp bóng có thể khiến MỘT lớp
mất sạch bóng -> twin-SVM song sinh suy biến (NaN). Hơn nữa, đo thực nghiệm cho thấy
gần như 100% ca suy biến gom về đúng 1 bóng mỗi lớp; twin-SVM trên 1 với 1 tâm mất
ý nghĩa lề và thua nearest-centroid. Vì vậy:

  1) rescue-only (floor): nếu một lớp còn < floor bóng, bơm thêm tâm của cụm điểm
     lớp thiểu số (trích từ bóng lẫn) để lớp đó có lại đại diện;
  2) nếu sau rescue một lớp VẪN còn <= nc_floor bóng, THOÁI LUI về nearest-centroid
     (1-NN trên tâm bóng) thay vì cố khớp twin-SVM.

Điều kiện kích hoạt chỉ dựa trên số bóng mỗi lớp (đo được), không cần biết loại nhiễu.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

LABELS = (1.0, -1.0)


def _radius(X, c):
    return float(np.sqrt(((X - c) ** 2).sum(1)).mean()) if len(X) else 0.0


def ball_arrays(balls):
    """(centers, radii, labels) từ danh sách Ball."""
    return (np.array([b.center for b in balls]),
            np.array([b.radius for b in balls]),
            np.array([b.label for b in balls]))


def class_counts(labels):
    """Đếm số phần tử mỗi lớp trong {+1, -1}."""
    c = Counter(float(v) for v in labels)
    return {k: c.get(k, 0) for k in LABELS}


def rescue_arrays(balls, floor=1):
    """Rescue-only: nếu một lớp có < floor bóng, thêm tâm cụm điểm lớp thiểu số
    (mang nhãn lớp bị đe dọa) trích từ các bóng lẫn. Trả (C, r, y)."""
    C = [b.center for b in balls]
    r = [b.radius for b in balls]
    y = [b.label for b in balls]
    cnt = Counter(b.label for b in balls)
    threatened = {c for c in LABELS if cnt.get(c, 0) < floor}
    if threatened:
        for b in balls:
            labs = b.data[:, -2]
            for c in threatened:
                if c == b.label:
                    continue
                Xc = b.X[labs == c]
                if len(Xc):
                    cc = Xc.mean(0)
                    C.append(cc)
                    r.append(_radius(Xc, cc))
                    y.append(c)
    return np.array(C), np.array(r), np.array(y)


def nearest_centroid_predict(C, y, X):
    """Nearest-centroid = 1-NN trên tâm bóng: gán mỗi mẫu X theo nhãn tâm gần nhất.
    Tương đương phân lớp nearest-prototype trên các tâm do k-means (2-means) sinh."""
    C = np.asarray(C, float)
    X = np.asarray(X, float)
    y = np.asarray(y)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    return y[d.argmin(1)]


def predict_with_fallback(balls, X, wave_factory, floor=1, nc_floor=1):
    """Đường phân lớp triển-khai-được ở độ thuần thấp.

    Bước:
      1) rescue-only(floor) để không lớp nào mất sạch bóng;
      2) nếu một lớp vẫn <= nc_floor bóng -> nearest-centroid (thoái lui êm);
         ngược lại fit twin-SVM qua wave_factory().

    wave_factory: hàm không đối số trả về một mô hình chưa fit, có .fit(C, r, y)
                  trả về self và .predict(X). Ví dụ:
                      lambda: WaveGBTSVM(c1=10, c2=10, lam0=1, kappa=0,
                                         adaptive_lambda=False, steps=250)
    Trả về mảng nhãn dự đoán cho X.
    """
    C, r, y = rescue_arrays(balls, floor=floor)
    X = np.asarray(X, float)
    if len(C) == 0 or len(np.unique(y)) < 2:
        # rescue cũng không đủ hai lớp (rất hiếm) -> trả nhãn đa số
        maj = Counter(y).most_common(1)[0][0] if len(y) else 1.0
        return np.full(len(X), maj)
    cc = class_counts(y)
    if min(cc.values()) <= nc_floor:
        return nearest_centroid_predict(C, y, X)     # 1 với vài tâm -> nearest-prototype
    return wave_factory().fit(C, r, y).predict(X)     # đủ bóng -> twin-SVM (còn ý nghĩa lề)
