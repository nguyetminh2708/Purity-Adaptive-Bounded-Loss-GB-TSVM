"""Rescue có còn là SVM hay đã thành kNN?  (trả lời câu hỏi của Min)

Ở chế độ suy biến (một lớp mất hết bóng), sau rescue thường chỉ còn RẤT ÍT bóng.
Twin-SVM trên 1 với vài tâm gần như mất ý nghĩa lề -> có thể chỉ tương đương
nearest-centroid (1-NN trên tâm bóng). Script này đo:

  - số bóng mỗi lớp TRƯỚC và SAU rescue (xác nhận 'còn 1 với vài bóng'),
  - acc của wave+rescue so với acc nearest-centroid (1-NN trên chính tâm bóng đó),
  - tách riêng các fold BỊ SUY BIẾN (raw mất 1 lớp) và fold KHÔNG suy biến.

Nếu ở fold suy biến wave ~ kNN -> rescue chỉ là lưới an toàn thoái lui về kNN.
Nếu ở fold không suy biến wave > kNN -> lề/SVM vẫn làm việc thật.

    python experiments/exp_rescue_knn.py [--purity 0.6] [--seeds 5]
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
from collections import Counter
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.datasets import load
from gb.balls import gen_balls
from gb.rescue import rescue_arrays, nearest_centroid_predict
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from noise import inject

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)


def counts(balls):
    c = Counter(b.label for b in balls)
    return c.get(1.0, 0), c.get(-1.0, 0)


def wave_acc(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    try:
        m = WaveGBTSVM(steps=250, **WAVE).fit(C, r, y)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(F))


def nc_acc(C, y, F, yte):
    """Nearest-centroid = 1-NN trên tâm bóng."""
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return accuracy_score(yte, nearest_centroid_predict(C, y, F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["haberman", "heart", "breast_cancer", "german", "australian", "ionosphere"])
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kinds", nargs="*", default=["symmetric", "asymmetric"])
    ap.add_argument("--rates", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    args = ap.parse_args()
    P = args.purity
    rows = []; t0 = time.time()
    print(f"pur={P} seeds={args.seeds} kinds={args.kinds} rates={args.rates}", flush=True)
    for name in args.datasets:
        ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
        cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
        for kind in args.kinds:
            for rate in args.rates:
                for seed in range(args.seeds):
                    for fi, (tr, te) in enumerate(
                            StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                        yb = pm(y[tr]).astype(int)
                        yn = inject(X[tr], yb, rate, kind, seed)[0].astype(float)
                        yte = pm(y[te])
                        sc = MinMaxScaler((-1, 1)).fit(X[tr])
                        Ftr, Fte = sc.transform(X[tr]), sc.transform(X[te])
                        b = gen_balls(np.column_stack([Ftr, yn]), pur=P, delbals=1, seed=seed)
                        rp, rn = counts(b)                     # bóng thô mỗi lớp
                        degen = (rp == 0 or rn == 0)           # raw suy biến?
                        C, r, yl = rescue_arrays(b)            # sau rescue
                        sp, sn = int((yl == 1.0).sum()), int((yl == -1.0).sum())
                        aw = wave_acc(C, r, yl, Fte, yte)
                        ak = nc_acc(C, yl, Fte, yte)
                        rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, fold=fi,
                                         raw_pos=rp, raw_neg=rn, degen=degen,
                                         resc_pos=sp, resc_neg=sn, nballs=sp + sn,
                                         acc_wave=aw, acc_nc=ak))
        print(f"  done {name} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = str(HERE.parent / "results" / "rescue_knn.csv")
    df.to_csv(out, index=False)

    def block(sub, tag):
        if len(sub) == 0:
            print(f"\n[{tag}] (không có fold)", flush=True); return
        print(f"\n[{tag}]  #fold={len(sub)}  bóng TB: +{sub.resc_pos.mean():.1f}/-{sub.resc_neg.mean():.1f}"
              f"  (tổng {sub.nballs.mean():.1f})", flush=True)
        print(f"    acc wave+rescue = {sub.acc_wave.mean():.3f}   |   acc nearest-centroid = {sub.acc_nc.mean():.3f}"
              f"   |   Δ(wave-kNN) = {sub.acc_wave.mean()-sub.acc_nc.mean():+.3f}", flush=True)

    print("\n================ TỔNG HỢP ================", flush=True)
    block(df[df.degen], "FOLD SUY BIẾN (raw mất 1 lớp) -> rescue kích hoạt")
    block(df[~df.degen], "FOLD KHÔNG suy biến (đối chứng)")
    block(df[df.degen & (df.kind == "asymmetric")], "SUY BIẾN + asymmetric")
    # phân bố số bóng ở fold suy biến
    dsub = df[df.degen]
    if len(dsub):
        print(f"\nỞ fold suy biến: tỉ lệ có <=2 bóng tổng = {(dsub.nballs<=2).mean():.0%}; "
              f"<=3 bóng = {(dsub.nballs<=3).mean():.0%}", flush=True)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
