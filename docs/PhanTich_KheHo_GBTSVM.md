# Phân tích khe hở nghiên cứu — nhánh Granular-Ball TSVM
**Nguyễn Trần Nguyệt Minh · cập nhật 07·09·2026**
*(Bản Markdown của artifact "Khe hở nghiên cứu của GBTSVM" — đã chốt hướng Purity-Adaptive Bounded Loss GB-TSVM; các đề xuất không dùng nữa đã lược bỏ.)*

---

## §1. Ý tưởng đã chốt, trong sáu câu

1. **Ý tưởng cốt lõi:** thay hinge trong GBTSVM bằng **Wave loss** (bị chặn, trơn) và cho **tham số chặn λₖ thích nghi theo độ tin cậy của từng granular ball** — tính từ purity, kích thước, bán kính. Bóng đáng tin → trần phạt cao; bóng nghi nhiễu → trần phạt thấp.
2. **Vì sao không đi các đường từng bàn:** adaptive purity ở khâu sinh bóng đã bị Xia (2022) và cả dòng sau đó chiếm; trọng số nhân cho từng bóng là lãnh địa GBFTSVM (IEEE TFS 2025) — cơ chế đó còn đi ngược chiều ở nhiễu cao; pinball trong GB đã có hai bài độc lập, và pinball không bị chặn nên không giải quyết được nhiễu nhãn (van Rooyen, NIPS 2015).
3. **Ô còn trống đã xác minh hai lần (30–31·08):** chưa ai ghép loss bị chặn với GB-twin; cả nhánh GB-TSVM chưa bài nào thử quá 20% nhiễu.
4. **Bằng chứng động cơ nằm sẵn trong tiểu luận:** bóng suy biến 1,3–1,7 điểm/bóng → cơ chế nhãn-đa-số biến mất; GBTSVM thua kernel SVM 8,9 điểm ở nhiễu 40% (§2).
5. **Hai tầng novelty phòng thủ độc lập:** tầng 1 — Wave × granular ball (ô trống, làm nhanh); tầng 2 — λₖ theo từng bóng (góc riêng, sống sót kể cả bị vượt tầng 1).
6. **Mở màn bằng chẩn đoán cơ chế** (số bóng, purity theo nhiễu 0–50%, ba mô hình GB cùng giao thức) — tự đứng được như benchmark, phương án lùi nằm sẵn trong giai đoạn 1.

## §2. Bằng chứng đã nằm sẵn trong số liệu của tiểu luận

**Đường sụp đổ theo nhiễu (Balance Scale, RBF, 5-fold, từ `results/k_noise.csv`):**

| Nhiễu | SVM | TWSVM | LS-TWSVM | GBTSVM | GBTSVM (p=0,8) |
|---|---|---|---|---|---|
| 0% | 98,1 | 98,6 | 99,8 | 99,1 | 98,4 |
| 10% | 97,7 | 96,4 | 96,9 | 97,0 | 95,5 |
| 20% | 95,5 | 96,5 | 94,1 | 94,8 | 92,2 |
| 30% | 93,1 | 85,8 | 87,0 | 85,1 | 84,2 |
| 40% | **77,3** | 73,6 | 77,8 | **68,4** | **63,2** |

Ở 40%, GBTSVM thua SVM 8,9 điểm; hạ purity xuống 0,8 còn tệ hơn ở *mọi* mức nhiễu → ngưỡng purity cố định, cao hay thấp, đều không phải lời giải.

**Bóng không hề nén dữ liệu — cơ chế thất bại:**

| Bộ dữ liệu | Mẫu train | Số bóng | Điểm/bóng |
|---|---|---|---|
| Balance Scale | 500 | 293,4 | 1,70 |
| Haberman | 245 | 185,4 | 1,32 |
| Heart (Cleveland) | 242 | 177,8 | 1,36 |
| Synthetic 4000 | 4 000 | 2 776 | 1,44 |

Bóng trung bình 1,3–1,7 điểm thì **không có "nhãn đa số" để nuốt nhiễu**. Purity cố định + nhiễu cao → chẻ đệ quy tới suy biến → GBTSVM thoái hoá về TWSVM trên tâm bóng lệch. **Việc cần làm ngay:** vẽ số bóng & kích thước bóng theo mức nhiễu — hình minh hoạ cơ chế chưa bài nào công bố.

## §3. Bản đồ những gì đã có người làm

**Nhóm A — trùng trực tiếp:**
- **GBFTSVM** (Lang, Zhao, Miao, Ding · IEEE TFS 33(7), 2025 · arXiv 2408.00699) — adaptive purity của Xia + membership/non-membership Pythagorean fuzzy từng bóng, non-membership từ purity: ν²ᵢ = (1−µ²ᵢ)(1−pᵢ). **Khai thác:** hàm chấm điểm của họ cho bóng KHÔNG thuần điểm *cao hơn* bóng thuần (sᵢ = µᵢ nếu pᵢ=1, ngược lại θᵢ > µᵢ) — hợp lý ở 5–10% nhiễu, khuếch đại nhiễu ở 30–40%.
- **GB-Pin-TSVM** (Quadir, Tanveer · IEEE TCSS 12:3891–3900) — robust loss + GB đã ghép, nhưng pinball lồi & không bị chặn.
- **GB-TBSVM generalized pinball** (Grewal, Gupta, Kumar · Inf. Sci. 751:123557, 2026) — nhóm thứ hai độc lập cùng hướng pinball.

**Nhóm B — adaptive purity (đã chiếm, ở khâu sinh bóng):** Xia 2201.04343; VPGB (Inf. Sci. 2022); GBG++ (TETCI 2024); shadowed-GB (Inf. Fusion 2025); local-density GB (Inf. Sci. 2025); MDL-GBC (arXiv 2605.11406, 2026). Điểm chung: tiêu chí thuần **hình học**, quyết định ở tiền xử lý; không bài nào lấy ước lượng nhiễu làm biến điều khiển hay ghép với hàm mục tiêu bộ phân lớp.

**Nhóm C — nhánh GB-TSVM còn lại:**

| Mô hình | Nguồn | Trục | Nhiễu đã thử |
|---|---|---|---|
| GBTSVM / LS-GBTSVM | IEEE TNNLS 36:12444, 2025 | bài gốc | 5–20% |
| EF-GBTSVM | IEEE TETCI 2026 | đặc trưng RVFL | 5–20% |
| GBLSTSVM | Pattern Recognition 170:112021, 2026 | least squares | 0–20% |
| GB-TWKSVC | Pattern Recognition 166:111636, 2025 | đa lớp | không thử |
| GBU-TSVM | Neural Networks 193:107974, 2026 | Universum | không thử |
| GBTSVM đa lớp (VN) | J. Comp. Sci. & Cybernetics 2025 | phân rã đa lớp | không thử |
| MLGBTSVM | PReMI 2025 | đa nhãn | không thử |

**Nhóm D — loss bị chặn, chín trong TSVM điểm, CHƯA vào GB:** Wave-TSVM (Pattern Recognition 2024), RoBoTS (Pattern Recognition 2026), GL-TSVM, HawkEye; ramp/capped/rescaled hinge/Welsch/truncated pinball (CTSVM 2019, RHTSVM 2019, WCTBSVM 2023, CPin-TBSVM 2024).

**Nhóm E — GB đối mặt nhiễu nhãn, ngoài SVM:** GB Sampling (TNNLS 2021), VPGB, G-GBC (KBS 2026), GEAF (Inf. Sci. 2026), INGB — chính sự tồn tại của dòng này là bằng chứng gián tiếp rằng purity cố định không đủ khi có nhiễu.

## §4. Chấm điểm ba thành phần đề xuất ban đầu

| Thành phần | Tình trạng | Còn gì cho mình |
|---|---|---|
| Adaptive purity | **Đã có** | Chỉ còn nếu điều khiển bằng ước lượng nhiễu, hoặc ghép vào objective |
| Reliability weight từng bóng trong TSVM | **Đã có** (GBFTSVM) | Trọng số *giảm* theo độ bẩn + cơ sở importance reweighting (Liu & Tao 2016) là mô hình khác hẳn |
| Robust loss trong GBTSVM | **Một nửa** (pinball) | Toàn bộ họ loss **bị chặn** còn trống |
| Vùng nhiễu 30–50% cho GB-TSVM | **Trống** | Riêng đường suy giảm đã đăng được |
| Mô tả cơ chế bóng vỡ vụn theo nhiễu | **Trống** | Phần "tại sao" của bài |
| Ghép cả ba | **Trống** | Nhưng phải có ablation từng khối |

## §5. Phương pháp: Purity-Adaptive Bounded Loss GB-TSVM

**Một câu:** thay hinge trong GBTSVM bằng Wave loss, và để độ tin cậy của mỗi bóng quyết định trần phạt λₖ của chính bóng đó.

### Wave loss là gì, làm gì

L(u) = (1/λ)·(1 − 1/(1 + λu²e^(au))) = u²e^(au)/(1 + λu²e^(au)), với u là mức vi phạm lề, λ quyết định trần bão hoà 1/λ, a chỉnh bất đối xứng.

Ba hành vi: **(1) bị chặn** — điểm sai nhãn xa cỡ nào cũng chỉ đóng tối đa 1/λ, và gradient → 0 khi u lớn (redescending) nên nó mất tiếng nói kéo siêu phẳng, trong khi hinge phạt tuyến tính không trần và squared phạt bình phương (lý do LS-TWSVM nhạy nhiễu nhất); **(2) trơn** — khả vi mọi nơi, hai bài twin thành tối ưu không ràng buộc giải bằng Adam/NAG, bỏ QP lẫn nghịch đảo (AᵀA)⁻¹; **(3) phạt nhẹ cả phía đúng** như pinball → ổn định resampling, nhưng tắt dần về 0. Ví dụ tại u = 5: hinge phạt 5, squared 25, Wave ≤ 1/λ (2,0 nếu λ=0,5; 0,5 nếu λ=2). Khi u nhỏ, L ≈ u²e^(au) **không phụ thuộc λ** — λ chỉ đổi trần, không đổi hành vi vùng sạch.

### Cơ chế λₖ — nơi chứa toàn bộ novelty tầng 2

λₖ = λ₀·exp(−κ·sₖ), với sₖ là điểm tin cậy từ purity pₖ, log nₖ, bán kính chuẩn hoá r̄ₖ. Bóng lớn-thuần → λₖ nhỏ → trần cao; bóng nhỏ-bẩn → λₖ lớn → trần thấp.

min_(w₁,b₁) ½‖A_c w₁ + e b₁‖² + C₁ Σₖ L^(λₖ)(1 + rₖ + (cₖᵀw₁ + b₁)) + ½C₂(‖w₁‖² + b₁²) — bán kính rₖ nằm trong argument của loss, kế thừa đúng ràng buộc gốc GBTSVM.

**Vì sao không bị quy về GBFTSVM:** trọng số nhân wₖ·L scale loss ở mọi nơi kể cả vùng sạch; λₖ giữ nguyên hành vi vùng sạch, chỉ đổi *quyền phản đối tối đa* — chỉnh hình dạng hàm phạt, không chỉnh âm lượng. Chứng minh bằng dòng ablation bắt buộc: wₖ·L(λ cố định) vs L(λₖ thích nghi).

### Hai tầng novelty
- **Tầng 1 — Wave × granular ball:** ô trống đã xác minh, nhưng là bước hiển nhiên của nhóm Tanveer → làm nhanh, preprint sớm.
- **Tầng 2 — λₖ theo bóng:** góc riêng; điểm tin cậy chỉ tồn tại tự nhiên trên bóng (điểm lẻ không có purity/kích thước) nên SVM điểm không chiếm được ô này.

### Phần toán bắt buộc
1. Bài toán không ràng buộc đầy đủ cho hai siêu phẳng, với λₖ và bán kính trong loss.
2. Gradient theo (w, b) tường minh; numerical gradient check.
3. Tính chất loss theo λₖ: boundedness, smoothness, gradient triệt tiêu (phát biểu lại cho bản per-ball).
4. Hội tụ về điểm dừng của solver bậc nhất trên objective phi lồi; độ phức tạp mỗi epoch so với GBTSVM (QP) và TWSVM.
5. Kiểm chứng: λₖ bằng nhau ⇒ Wave-GBTSVM thuần; λ rất nhỏ ⇒ nghiệm sát GBTSVM-QP.

## §6. Thiết kế thực nghiệm

Chuẩn nhánh: 21–42 bộ UCI/KEEL, nhiễu tối đa 20% (GB-TSVM) hoặc 30–40% (Wave/HawkEye), Friedman + Nemenyi/Wilcoxon. Đề nghị:
- **Nhiễu:** 0/10/20/30/40/50%, symmetric + asymmetric (instance-dependent nếu kịp); trích Frénay & Verleysen 2014 (NCAR/NAR/NNAR). Nhiễu chỉ tiêm vào train, test giữ sạch — nêu tường minh.
- **Baseline bắt buộc:** SVM, TWSVM, LS-TWSVM, GBTSVM, LS-GBTSVM, GBLSTSVM + **GBFTSVM**, **GB-Pin-TSVM**, một robust TSVM trên điểm (Wave-TSVM/IF-TSVM) — thiếu ba cái sau thì bảng không thuyết phục.
- **Ablation (7 cấu hình, cô lập 3 biến — GB, loss, λ):** TWSVM (mốc không GB) · GBTSVM gốc (GB một mình) · Wave-TSVM điểm (loss một mình — GB có thực sự cần?) · Wave-GBTSVM λ cố định (**tầng 1**) · wₖ·Wave(λ cố định) (**dòng bắt buộc:** chỉnh âm lượng vs chỉnh trần) · trọng số đảo chiều kiểu GBFTSVM (chiều trọng số quan trọng ở nhiễu cao) · PA-BL-GBTSVM đầy đủ (**tầng 2**). Chạy đủ trên 8–10 bộ đại diện ở 0/20/40% là đạt.

**Chẩn đoán (chưa ai có):** số bóng & kích thước bóng theo nhiễu; phân bố purity 0% vs 40%; tương quan wᵢ với tỉ lệ nhãn lật thật; sensitivity heatmap; scalability NDC kèm tỉ lệ nén thật.

## §7. Rủi ro & lộ trình triển khai PA-BL-GBTSVM (14 tuần)

**Rủi ro:** (1) bị nhóm IIT Indore lấp ô "bounded loss × GB" trước → làm nhanh, preprint sớm; (2) bị coi là chắp vá → ablation + lập luận "loss bị chặn = trọng số ẩn"; (3) reviewer nêu Xia 2201.04343 → nói thẳng trong related work, khác biệt là biến điều khiển.

**Phân công:** Minh giữ trục cấu trúc (sinh bóng, pipeline thực nghiệm, tiêm nhiễu, thống kê); phía bạn giữ trục loss (Wave, gradient, solver, chứng minh); ablation + viết bài làm chung.

| Giai đoạn | Thời gian | Việc | Deliverable |
|---|---|---|---|
| **GĐ 0** | 2–3 ngày | Repo chung (`gb/ · loss/ · experiments/ · results/ · paper/`); môi trường + 10 seed cố định; tải code gốc github.com/mtanveer1/GBTSVM; chốt ~12 bộ UCI/KEEL (WDBC, Balance Scale, Haberman, Cleveland, Pima, Ionosphere, Sonar, Australian, German, Monk-2…) | Repo chạy được |
| **GĐ 1** | Tuần 1–3 | Module tiêm nhiễu 4 kiểu (symmetric/asymmetric/boundary/far-outlier), 0–50%, chỉ tiêm train. Chạy GBTSVM/GBFTSVM/GB-Pin-TSVM cùng giao thức + SVM/TWSVM/LS-TWSVM. Đo 4 thứ chưa ai công bố: accuracy theo nhiễu; số bóng & kích thước bóng theo nhiễu; phân bố purity 0% vs 40%; tỉ lệ bóng lật nhãn đa số | **Báo cáo giữa kỳ tự đứng được** = phương án lùi benchmark |
| **GĐ 2** | Tuần 4–6 | Wave-GBTSVM λ cố định: objective không ràng buộc (rⱼ trong argument loss), tự dẫn gradient + numerical gradient check, solver Adam/NAG. Kiểm chứng suy biến (λ nhỏ ≈ GBTSVM-QP). So baseline 12 bộ ở 0/20/40% | Số liệu sơ bộ cho đề cương |
| **GĐ 3** | Tuần 7–9 | λₖ = λ₀·exp(−κ·sₖ), sₖ từ purity + log-size − bán kính chuẩn hoá; thí nghiệm tương quan λₖ vs tỉ lệ nhãn lật thật. Toán: boundedness/smoothness, hội tụ điểm dừng, complexity, trường hợp giới hạn. Ablation 2×2×2 + 2 dòng bắt buộc: wₖ·L(λ cố định) vs L(λₖ); trọng số đảo chiều kiểu GBFTSVM | Bản đầy đủ của phương pháp |
| **GĐ 4** | Tuần 10–12 | 12 bộ × 6 mức × ≥2 kiểu nhiễu × ~9 mô hình × 10 seed; Friedman + Nemenyi (CD) + Wilcoxon; sensitivity λ₀, κ, a, C, γ; scalability NDC + tỉ lệ nén thật; hình trung tâm loss + influence curve | Toàn bộ bảng/hình của bài |
| **GĐ 5** | Tuần 13–14 | Viết (motivation = GĐ 1; related work = §3–§4; phương pháp = GĐ 2–3; thực nghiệm = GĐ 4); arXiv trước, nộp Pattern Recognition / Inf. Sci. / IEEE TETCI / ASOC | Preprint + bản nộp |

**Ba điểm dừng go/no-go:**
1. Cuối GĐ 1: nếu GBTSVM *không* sập ở 30–40% trên đa số bộ → motivation lung lay, dừng xét lại trước khi code phương pháp.
2. Cuối GĐ 2: nếu Wave-GB λ cố định không hơn GB-Pin ở nhiễu cao → kéo tầng 2 (λₖ) lên làm sớm.
3. Bị scoop tầng 1 bất kỳ lúc nào → giữ lộ trình, dồn claim sang tầng 2 + chẩn đoán; GĐ 1 nguyên giá trị.

**Tiêu đề khuyến nghị:** *Purity-Adaptive Bounded Loss Granular-Ball Twin Support Vector Machine for Learning with Heavy Label Noise*.

## §8. Đọc theo thứ tự

1. GBFTSVM — IEEE TFS 33(7), 2025 *(có trên máy — đối thủ số một)*
2. GBTSVM — IEEE TNNLS 36:12444, 2025 · arXiv 2410.04774 *(có trên máy)*
3. GB-Pin-TSVM — IEEE TCSS 12:3891, 2024 *(có trên máy)*
4. Adaptive GB generation — Xia et al., arXiv 2201.04343
5. van Rooyen, Menon, Williamson — "The Importance of Being Unhinged", NIPS 2015
6. Liu & Tao — Importance Reweighting, IEEE TPAMI 38(3), 2016
7. Frénay & Verleysen — Label-noise survey, IEEE TNNLS 25(5), 2014
8. Wave loss / RoBoTS — Akhtar, Tanveer, Arshad, Pattern Recognition 2024 & 2026
9. GB-TBSVM generalized pinball — Inf. Sci. 751:123557, 2026 *(có trên máy)*
10. GBLSTSVM — Pattern Recognition 170:112021, 2026 · arXiv 2410.17338

*Cần xác minh trước khi trích: năm chính thức của GB-Pin-TSVM (2024 vs 2025) và tình trạng xuất bản EF-GBTSVM.*

---
