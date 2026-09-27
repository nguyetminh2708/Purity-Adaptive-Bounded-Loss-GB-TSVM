"""Kiểm nghiệm trên NHIỄU NHÃN THẬT — CIFAR-10N (Wei et al., ICLR 2022).
CIFAR-10N cung cấp nhãn DO NGƯỜI gán (nhiễu thật) kèm nhãn SẠCH cho CIFAR-10.

Ở đây dựng một bài NHỊ PHÂN TABULAR:
  - lấy 2 lớp (mặc định cat vs dog),
  - đặc trưng: ảnh phẳng 3072 chiều -> chuẩn hóa -> PCA(--pca) -> MinMax(-1,1),
  - nhãn HUẤN LUYỆN = nhãn-người (nhiễu thật, chiếu về nhị phân),
  - nhãn KIỂM TRA  = nhãn sạch (tập test CIFAR-10),
  - chạy: hard(pur=1) / wave(pur=1) / hard(P) / wave(P) / wave+rescue(P).

NHIỀU SEED: nhãn nhiễu & tách train/test là CỐ ĐỊNH (dữ liệu thật). Mỗi seed đổi
PCA(random_state) và khởi tạo sinh bóng (2-means) -> đo ĐỘ ỔN ĐỊNH của pipeline.

CÁCH CHẠY (trên máy có internet lần đầu để tải ảnh CIFAR-10):
  python experiments/exp_realnoise.py --classes automobile truck --noise aggre_label --pca 100 --seeds 5
  tùy chọn: --purity 0.7   --labels <đường_dẫn CIFAR-10_human.pt>
"""
from __future__ import annotations
import sys, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from gb.balls import gen_balls, arrays
from gb.rescue import rescue_arrays
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.registry import fit_binary, decide, Degenerate

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)
GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
CIFAR = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


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


def load_raw(labels_path, cls_pos, cls_neg, noise_key):
    """Tải + chuẩn hóa ảnh (KHÔNG PCA — để PCA đổi theo seed ở ngoài).
    Trả (train chuẩn hóa, nhãn nhiễu train, test chuẩn hóa, nhãn sạch test, tỉ lệ nhiễu)."""
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
    m = np.isin(ytr_clean, [ip, ino])                              # train: nhãn SẠCH thuộc {pos,neg}
    Xtr, ytr_clean_b, ynoisy_b = Xtr[m], ytr_clean[m], ynoisy[m]
    yc = np.where(ytr_clean_b == ip, 1.0, -1.0)                    # nhãn sạch nhị phân
    yn = np.where(ynoisy_b == ip, 1.0, np.where(ynoisy_b == ino, -1.0, -yc))  # lớp thứ ba = lật
    mt = np.isin(yte_clean, [ip, ino])
    Xte, yte_b = Xte[mt], np.where(yte_clean[mt] == ip, 1.0, -1.0)

    ss = StandardScaler().fit(Xtr)
    Str_, Ste_ = ss.transform(Xtr), ss.transform(Xte)
    noise_rate = float((yn != yc).mean())
    return Str_, yn, Ste_, yte_b, noise_rate


def feats(Str_, Ste_, pca_dim, seed):
    """PCA(random_state=seed) -> MinMax(-1,1). Đổi theo seed để đo độ ổn định."""
    pca = PCA(n_components=pca_dim, random_state=seed).fit(Str_)
    Ftr = pca.transform(Str_); Fte = pca.transform(Ste_)
    mm = MinMaxScaler((-1, 1)).fit(Ftr)
    return mm.transform(Ftr), mm.transform(Fte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="đường dẫn CIFAR-10_human.pt (mặc định tự tìm trong data/raw)")
    ap.add_argument("--classes", nargs=2, default=["cat", "dog"])
    ap.add_argument("--noise", default="worse_label",
                    help="worse_label(~40%) | aggre_label(~9%) | random_label1(~17%)")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--purity", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=1, help="số seed (mỗi seed đổi PCA + sinh bóng)")
    args = ap.parse_args()

    labels_path = find_labels(args.labels)
    print(f"nhãn nhiễu thật: {labels_path}")
    Str_, yn, Ste_, yte, nr = load_raw(labels_path, args.classes[0], args.classes[1], args.noise)
    print(f"Bài nhị phân: {args.classes[0]} vs {args.classes[1]} | đặc trưng PCA={args.pca}")
    print(f"Train {len(Str_)} mẫu, tỉ lệ NHIỄU THẬT (nhị phân) = {nr:.1%} [{args.noise}]")
    print(f"Test  {len(Ste_)} mẫu (nhãn sạch) | seeds={args.seeds}\n")

    P = args.purity
    cols = ["hard(p1)", "wave(p1)", f"hard({P})", f"wave({P})", f"wave+R({P})"]
    acc = {c: [] for c in cols}
    hdr = f"{'seed':>4}  " + "".join(f"{c:>12}" for c in cols)
    print(hdr)
    for s in range(args.seeds):
        Ftr, Fte = feats(Str_, Ste_, args.pca, s)
        sub = np.column_stack([Ftr, yn])
        b1 = gen_balls(sub, pur=1.0, delbals=1, seed=s)
        bp = gen_balls(sub, pur=P,   delbals=1, seed=s)
        C1, r1, y1, _, _ = arrays(b1)
        Cp, rp, yp, _, _ = arrays(bp)    # pur=P thô (chưa rescue)
        Cr, rr, yr = rescue_arrays(bp)   # pur=P + rescue-only
        row = [_hard(C1, r1, y1, Fte, yte), _wave(C1, r1, y1, Fte, yte),
               _hard(Cp, rp, yp, Fte, yte), _wave(Cp, rp, yp, Fte, yte),
               _wave(Cr, rr, yr, Fte, yte)]
        for c, v in zip(cols, row):
            acc[c].append(v)
        print(f"{s:>4}  " + "".join(f"{('nan' if v != v else f'{v:.3f}'):>12}" for v in row))

    print("-" * len(hdr))

    def ms(c):
        a = np.array(acc[c], float); a = a[~np.isnan(a)]
        return (np.nan, np.nan, 0) if len(a) == 0 else (a.mean(), a.std(), len(a))

    print(f"{'TB':>4}  " + "".join(f"{('nan' if ms(c)[2] == 0 else f'{ms(c)[0]:.3f}'):>12}" for c in cols))
    print(f"{'±sd':>4}  " + "".join(f"{('' if ms(c)[2] == 0 else f'{ms(c)[1]:.3f}'):>12}" for c in cols))
    print(f"{'n':>4}  " + "".join(f"{ms(c)[2]:>12}" for c in cols))


if __name__ == "__main__":
    main()
