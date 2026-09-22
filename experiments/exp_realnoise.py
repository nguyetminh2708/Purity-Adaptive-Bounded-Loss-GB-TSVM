"""Kiểm nghiệm trên NHIỄU NHÃN THẬT — CIFAR-10N (Wei et al., ICLR 2022).
CIFAR-10N cung cấp nhãn DO NGƯỜI gán (nhiễu thật) kèm nhãn SẠCH cho CIFAR-10.

Ở đây dựng một bài NHỊ PHÂN TABULAR:
  - lấy 2 lớp dễ nhầm (mặc định cat vs dog),
  - đặc trưng: ảnh phẳng 3072 chiều -> chuẩn hóa -> PCA(--pca) -> MinMax(-1,1),
  - nhãn HUẤN LUYỆN = nhãn-người (nhiễu thật, chiếu về nhị phân),
  - nhãn KIỂM TRA  = nhãn sạch (tập test CIFAR-10),
  - chạy: hard (GBTSVM) / wave (pur=1) / wave+rescue (pur=0.7 rescue-only).

CÁCH CHẠY (trên máy có internet):
  1) pip install torch torchvision scikit-learn numpy   (nếu chưa có)
  2) Tải nhãn CIFAR-10N: vào https://github.com/UCSC-REAL/cifar-10-100n
     lấy file data/CIFAR-10_human.pt  đặt cạnh script (hoặc truyền --labels đường dẫn).
  3) python experiments/exp_realnoise.py            (CIFAR-10 ảnh sẽ tự tải qua torchvision)
     tùy chọn: --classes cat dog  --noise worse_label  --pca 50  --purity 0.7

In ra: tỉ lệ nhiễu THẬT của bài nhị phân + accuracy hard/wave/wave_rescue trên test sạch.
"""
from __future__ import annotations
import sys, pathlib, argparse
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from gb.balls import gen_balls   # chỉ cần hàm này (tương thích cả balls.py cũ)
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.registry import fit_binary, decide, Degenerate


def arrays(balls):
    """(centers, radii, labels) — tự định nghĩa để khỏi phụ thuộc bản balls.py mới."""
    return (np.array([b.center for b in balls]),
            np.array([b.radius for b in balls]),
            np.array([b.label for b in balls]))


def viable(balls):
    if not balls:
        return False
    c = Counter(b.label for b in balls)
    return len(c) >= 2 and min(c.values()) >= 1

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)
GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
CIFAR = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
ALL = (1.0, -1.0)


def _radius(X, c):
    return float(np.sqrt(((X - c) ** 2).sum(1)).mean()) if len(X) else 0.0


def rescue_arrays(balls, floor=1):
    C = [b.center for b in balls]; r = [b.radius for b in balls]; y = [b.label for b in balls]
    cnt = Counter(b.label for b in balls)
    threatened = {c for c in ALL if cnt.get(c, 0) < floor}
    if threatened:
        for b in balls:
            labs = b.data[:, -2]
            for c in threatened:
                if c == b.label: continue
                Xc = b.X[labs == c]
                if len(Xc):
                    cc = Xc.mean(0); C.append(cc); r.append(_radius(Xc, cc)); y.append(c)
    return np.array(C), np.array(r), np.array(y)


def _wave(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2: return np.nan
    try: m = WaveGBTSVM(steps=250, **WAVE).fit(C, r, y)
    except WDeg: return np.nan
    return accuracy_score(yte, m.predict(F))


def _hard(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2: return np.nan
    try: return accuracy_score(yte, decide(fit_binary("GBTSVM", GB, np.column_stack([C, r, y])), F))
    except Degenerate: return np.nan


DATA_RAW = HERE.parent / "data" / "raw"


def find_labels(user_path):
    """Tìm CIFAR-10_human.pt: ưu tiên đường dẫn người dùng, rồi data/raw, gốc dự án, cwd."""
    cands = ([pathlib.Path(user_path)] if user_path else []) + [
        DATA_RAW / "CIFAR-10_human.pt",
        HERE.parent / "CIFAR-10_human.pt",
        pathlib.Path("CIFAR-10_human.pt"),
    ]
    for p in cands:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "Không thấy CIFAR-10_human.pt. Tải từ github UCSC-REAL/cifar-10-100n (thư mục data/) "
        f"và đặt vào: {DATA_RAW}  (hoặc truyền --labels <đường_dẫn>).")


def load_cifar(labels_path, cls_pos, cls_neg, noise_key, pca_dim, seed=0):
    import torch, torchvision
    root = str(DATA_RAW / "cifar10")     # ảnh CIFAR-10 tải về data/raw/cifar10
    tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    Xtr = tr.data.reshape(len(tr.data), -1).astype(np.float32)      # (50000, 3072)
    Xte = te.data.reshape(len(te.data), -1).astype(np.float32)
    ytr_clean = np.array(tr.targets); yte_clean = np.array(te.targets)
    try:
        human = torch.load(labels_path, weights_only=False)         # dict các nhãn người
    except TypeError:
        human = torch.load(labels_path)                             # torch cũ không có tham số này
    ynoisy = np.array(human[noise_key]).reshape(-1)                 # nhãn nhiễu thật (train)

    ip, ino = CIFAR.index(cls_pos), CIFAR.index(cls_neg)
    # tập train: mẫu có nhãn SẠCH thuộc {pos,neg}
    m = np.isin(ytr_clean, [ip, ino])
    Xtr, ytr_clean_b, ynoisy_b = Xtr[m], ytr_clean[m], ynoisy[m]
    # nhãn nhị phân: +1 nếu = pos, -1 nếu = neg; nhãn người khác lớp -> coi là LẬT (nhiễu thật)
    yc = np.where(ytr_clean_b == ip, 1.0, -1.0)
    yn = np.where(ynoisy_b == ip, 1.0, np.where(ynoisy_b == ino, -1.0, -yc))  # lớp thứ ba = lật
    # test: nhãn sạch
    mt = np.isin(yte_clean, [ip, ino])
    Xte, yte_b = Xte[mt], np.where(yte_clean[mt] == ip, 1.0, -1.0)

    # đặc trưng: chuẩn hóa -> PCA -> MinMax(-1,1)
    ss = StandardScaler().fit(Xtr)
    pca = PCA(n_components=pca_dim, random_state=seed).fit(ss.transform(Xtr))
    Ftr = pca.transform(ss.transform(Xtr)); Fte = pca.transform(ss.transform(Xte))
    mm = MinMaxScaler((-1, 1)).fit(Ftr); Ftr, Fte = mm.transform(Ftr), mm.transform(Fte)
    noise_rate = float((yn != yc).mean())
    return Ftr, yn, Fte, yte_b, noise_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="đường dẫn CIFAR-10_human.pt (mặc định tự tìm trong data/raw)")
    ap.add_argument("--classes", nargs=2, default=["cat", "dog"])
    ap.add_argument("--noise", default="worse_label",
                    help="worse_label(~40%) | aggre_label(~9%) | random_label1(~17%)")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--purity", type=float, default=0.7)
    args = ap.parse_args()

    labels_path = find_labels(args.labels)
    print(f"nhãn nhiễu thật: {labels_path}")
    Ftr, yn, Fte, yte, nr = load_cifar(labels_path, args.classes[0], args.classes[1],
                                       args.noise, args.pca)
    print(f"Bài nhị phân: {args.classes[0]} vs {args.classes[1]} | đặc trưng PCA={args.pca}")
    print(f"Train {len(Ftr)} mẫu, tỉ lệ NHIỄU THẬT (nhị phân) = {nr:.1%} [{args.noise}]")
    print(f"Test  {len(Fte)} mẫu (nhãn sạch)\n")

    # bóng pur=1 cho hard & wave; pur=P cho hard/wave/rescue (so cùng điều kiện)
    b1 = gen_balls(np.column_stack([Ftr, yn]), pur=1.0, delbals=1, seed=0)
    bp = gen_balls(np.column_stack([Ftr, yn]), pur=args.purity, delbals=1, seed=0)
    C1, r1, y1 = arrays(b1)
    Cp, rp, yp = arrays(bp)          # pur=P thô (chưa rescue)
    Cr, rr, yr = rescue_arrays(bp)   # pur=P + rescue-only
    P = args.purity
    print(f"{'model':26s}{'acc test sạch':>14}")
    print(f"{'GB hard (pur=1)':26s}{_hard(C1, r1, y1, Fte, yte):>14.3f}")
    print(f"{'GB wave (pur=1)':26s}{_wave(C1, r1, y1, Fte, yte):>14.3f}")
    print(f"{'GB hard (pur=%.1f)'%P:26s}{_hard(Cp, rp, yp, Fte, yte):>14.3f}")
    print(f"{'GB wave (pur=%.1f)'%P:26s}{_wave(Cp, rp, yp, Fte, yte):>14.3f}")
    print(f"{'GB wave+rescue (pur=%.1f)'%P:26s}{_wave(Cr, rr, yr, Fte, yte):>14.3f}")


if __name__ == "__main__":
    main()
