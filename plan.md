# Kế hoạch triển khai: Đánh giá hiệu quả của Sensor Fusion (RGB-IR) trong bài toán phát hiện đối tượng từ UAV

**Phiên bản:** 1.1
**Ngày:** 2026-08-19
**Trạng thái:** Dữ liệu đã sẵn sàng — compute budget đã chốt (phương án rút gọn), detector đã chốt (Ultralytics)

---

## 0. Trạng thái thực thi

### 0.1 Cấu hình đã chốt (2026-08-19)

Phần cứng thực tế: **RTX 3060 Ti 8GB** (dùng chung với màn hình), i5-13600K 20 luồng, 32GB RAM, 426GB trống.
Đây **không phải** RTX 4090 như giả định ban đầu ở mục 9 → áp dụng phương án rút gọn **ngay từ đầu**, không chờ đến khi thiếu.

| Tham số | Giá trị chốt | Ghi chú |
|---|---|---|
| Epoch | **60** + cosine schedule | thay vì 100 |
| Seed (DroneVehicle) | **2** | không giảm xuống 1 |
| Fold (VEDAI) | **3** (01, 03, 05) | thay vì 4 |
| Batch size | **8 cho MỌI cấu hình** | xem 6.4 — bắt buộc để so sánh công bằng |
| `nbs` (nominal batch) | **32** | gradient accumulate, giữ effective batch bằng nhau |
| Model | YOLO11s-obb | giữ, để còn đối chiếu literature |
| imgsz | 640 | |
| F2b (attention) | hoãn | chỉ làm nếu còn thời gian sau F2a |

**Ngân sách sau rút gọn: ~110–140 GPU-hours** (xem mục 9 đã cập nhật), vừa 3 tuần ở 8h/ngày.
**Không cắt:** C1, C2 (đối chứng capacity) và ≥2 seed.

### 0.2 Dữ liệu — đã hoàn tất

| | DroneVehicle | VEDAI |
|---|---|---|
| Nguồn | HF `McCheng/DroneVehicle` (sha256 đã xác minh khớp) | `downloads.greyc.fr/vedai` (subset 512) |
| Cặp dùng được | **28.400** (train 17.951 / val 1.469 / test 8.980) | **1.246** (10 fold, 1089/121) |
| Box (GT dùng để train) | **494.839** (annotation IR) | **3.746** (GT chung 2 modality) |
| Lớp | 5 | 8 |
| Trạng thái | ✅ cắt viền, chuẩn hoá nhãn, YOLO-OBB, smoke test PASS | ✅ như trên |

Chi tiết đầy đủ và các vấn đề dữ liệu phát hiện được: xem [DATA_REPORT.md](DATA_REPORT.md).

### 0.3 Code — đã triển khai đầy đủ

Toàn bộ 8 cấu hình của ma trận chạy được end-to-end. `pytest`: **44/44 xanh**.

| Thành phần | Trạng thái |
|---|---|
| Dataset ghép cặp nhiều kênh (3/4/6 kênh) | ✅ `src/data/paired_dataset.py` |
| Kiến trúc two-stream sinh tự động | ✅ `src/models/two_stream.py` |
| Nạp pretrained công bằng cho mọi cấu hình | ✅ 98,9% (single/early), 97,0% (two-stream) |
| Trainer dùng chung cho cả ma trận | ✅ `src/train.py` |
| mAP OBB + IoU đa giác chính xác | ✅ `src/eval/metrics.py` |
| Phân tầng + bootstrap CI ghép cặp | ✅ `src/eval/{stratify,bootstrap}.py` |
| Late fusion WBF cho box xoay | ✅ `src/fusion/wbf_obb.py` |
| Ablation RQ4 (dịch IR khi test) | ✅ `scripts/run_rq4_ablation.py` |
| Điều phối ma trận + sinh bảng | ✅ `scripts/{run_matrix,make_tables}.py` |
| **Ngắt/chạy lại ở mức epoch** | ✅ khôi phục optimizer + EMA + lịch LR; test bằng ngắt tín hiệu thật |

**Cách tiếp cận kỹ thuật đã chốt (thay cho phương án MMRotate ở R1):** xếp hai
modality thành một mảng HWC nhiều kênh ngay tại `load_image`, model tách lại bằng
lớp `Index` ở đầu đồ thị. Toàn bộ pipeline Ultralytics dùng lại nguyên vẹn.
Chỉ cần **ba bản vá nhỏ**, không fork — xem README.md mục 2.

**Đối chứng capacity chặt hơn plan yêu cầu:** C1, C2 và F2a dùng **chung một
`arch_spec`**, nên số tham số bằng nhau *tuyệt đối* chứ không phải "sai số < 1%"
như mục 7.3 dự phòng. `tests/test_models.py::test_capacity_control_exact_match`
canh bất biến này.

### 0.4 Ba phát hiện khi triển khai — đều là lỗi âm thầm

| # | Phát hiện | Ảnh hưởng | Xử lý |
|---|---|---|---|
| 1 | **`RandomPerspective` lấp viền khác nhau giữa các nhóm kênh.** `borderValue` của OpenCV tối đa 4 thành phần → ảnh 6 kênh được lấp 114 cho 4 kênh đầu và **0 cho kênh 5-6** | **Nghiêm trọng.** Model học artefact "viền IR bằng 0"; đối chứng C1 không còn đối xứng thật sự → hỏng đúng phần làm nên giá trị đề tài | Đã sửa: warp từng nhóm ≤3 kênh cùng ma trận, cùng giá trị viền. Bắt bởi `test_augment_sync` |
| 2 | **mAP OBB của Ultralytics lạc quan có hệ thống.** Validator khớp box bằng ProbIoU (xấp xỉ dùng cho loss), cao hơn IoU thật **trung bình +0,13, tối đa +0,26** | mAP50-95 bị thổi phồng gần gấp đôi (0,103 vs 0,058 đo được) | Harness dùng IoU đa giác chính xác. **Số của `model.val()` không so trực tiếp được với bảng kết quả** — phải ghi rõ trong báo cáo |
| 3 | `compute_ap` của Ultralytics dùng `trapz` → dự đoán hoàn hảo cho AP = 0,995 | Nhỏ nhưng làm sanity check mục 7.3 không bao giờ đạt 1,0 | Dùng công thức 101 điểm của pycocotools |

Phát hiện #2 có hệ quả trực tiếp lên sanity check ở mục 7.3: **đối chiếu S2 với
~83% của literature chỉ hợp lệ khi dùng harness của đề tài**, không dùng
`model.val()` của Ultralytics.

---

## 1. Tóm tắt điều hành

Đề tài kiểm chứng một cách **có kiểm soát** giả thuyết rằng việc kết hợp (fusion) hai luồng cảm biến RGB và hồng ngoại (IR) gắn trên UAV cải thiện độ chính xác phát hiện đối tượng so với dùng từng cảm biến riêng lẻ.

Điểm khác biệt so với phần lớn công trình fusion hiện có: đề tài bổ sung **nhóm đối chứng capacity** (duplicate-input control) để tách bạch phần đóng góp của *thông tin bổ sung từ cảm biến thứ hai* khỏi phần đóng góp của *mô hình có nhiều tham số hơn* — một confounder thường bị bỏ qua trong literature.

Kết quả được **replicate trên hai dataset độc lập** và **phân tầng theo điều kiện vận hành** (ngày/đêm, kích thước đối tượng) thay vì chỉ báo cáo một con số mAP tổng hợp.

---

## 2. Bài toán và câu hỏi nghiên cứu

### 2.1 Hình thức hóa

Cho một tập dữ liệu ghép cặp (paired):

```
D = { (x_rgb^i, x_ir^i, y^i) }, i = 1..N
```

trong đó `x_rgb^i` và `x_ir^i` là ảnh của **cùng một cảnh, cùng thời điểm, đã đồng đăng ký không gian**, và `y^i` là tập oriented bounding box + nhãn lớp dùng chung cho cả hai modality.

Ta huấn luyện một họ mô hình `f_θ` với các cấu hình đầu vào khác nhau và so sánh hiệu năng trên cùng một tập test.

> **Lưu ý quan trọng:** Fusion là phép hợp **kênh** (mỗi mẫu có thêm chiều), không phải phép hợp **tập mẫu** (thêm mẫu). Vì vậy bắt buộc dùng dataset ghép cặp, không thể ghép hai dataset rời.

### 2.2 Câu hỏi nghiên cứu

**RQ1 (chính).** Với cùng số lượng mẫu huấn luyện và cùng số tham số mô hình, fusion RGB-IR có cải thiện độ chính xác phát hiện so với từng modality đơn lẻ không, và mức cải thiện là bao nhiêu?

**RQ2.** Chiến lược fusion nào (early / mid / late) cho tỉ lệ lợi ích trên chi phí tính toán tốt nhất?

**RQ3.** Lợi ích của fusion phân bố như thế nào theo điều kiện vận hành (chiếu sáng ngày/đêm, kích thước đối tượng, lớp đối tượng)?

**RQ4 (ablation).** Mỗi chiến lược fusion nhạy cảm ra sao với sai lệch đồng đăng ký (registration misalignment) giữa hai modality?

### 2.3 Giả thuyết

| ID | Giả thuyết | Cơ sở |
|---|---|---|
| H1 | Fusion > modality đơn lẻ tốt nhất, ngay cả sau khi kiểm soát capacity | Thông tin bổ sung thật sự tồn tại giữa hai phổ |
| H2 | Chênh lệch tập trung ở phân khúc ban đêm; ban ngày chênh lệch nhỏ hoặc âm | RGB đủ tốt ban ngày; IR bù đắp khi thiếu sáng |
| H3 | Mid fusion > early fusion > late fusion về mAP; late fusion bền nhất với misalignment | Mid có khả năng học trọng số thích ứng; late độc lập không gian |

**H2 quan trọng hơn H1.** Nếu chỉ chứng minh H1 thì đóng góp yếu; phân tích phân tầng theo H2 mới là phần có giá trị.

---

## 3. Lựa chọn dataset

### 3.1 Dataset chính: DroneVehicle

| Thuộc tính | Giá trị |
|---|---|
| Số cặp ảnh | 28.439 cặp RGB-IR — **thực dùng 28.400** (39 cặp hỏng từ upstream) |
| Annotation | 953.087 OBB (đã đếm lại: 953.161, lệch <0,01%) — **GT dùng để train: 494.839 box IR** |
| Lớp | 5 — car, truck, bus, van, freight car |
| Kích thước gốc | 840 × 712 (có viền trắng 100px mỗi cạnh) |
| Sau tiền xử lý | 640 × 512 |
| Độ cao bay | 80–120 m |
| Góc chụp | Thẳng đứng, nghiêng 15° / 30° / 45° |
| Chiếu sáng | Ban ngày / ban đêm / đêm tối — ⚠️ **KHÔNG có nhãn trong dataset**, xem 6.2 và DATA_REPORT.md |
| Split chuẩn | 17.990 train / 1.469 val / 8.980 test |
| Nguồn | HF `McCheng/DroneVehicle` (đã tải, sha256 xác minh khớp) |

**Lý do chọn:**
- Đã ghép cặp sẵn → cô lập được biến cảm biến (⚠️ đồng đăng ký **không hoàn hảo**: lệch trung vị 1,8 px, 20,9% cặp >5 px — xem R2)
- Có biến thiên chiếu sáng rõ rệt → phục vụ RQ3 (⚠️ phải tự suy diễn nhãn, xem 6.2)
- Quy mô đủ lớn để kết quả không bị nhiễu bởi variance
- Có nhiều baseline công bố để đối chiếu tính đúng đắn của pipeline

**Cảnh báo kỹ thuật:**
- ⚠️ Viền trắng 100px **phải** được cắt bỏ trước khi train. Nếu quên, model sẽ học được artifact và kết quả vô nghĩa.
- ⚠️ Dataset có annotation **riêng cho từng modality** (số lượng box RGB và IR khác nhau — ví dụ lớp car có 389.779 box trên RGB nhưng 428.086 box trên IR, do IR nhìn thấy xe trong bóng tối mà RGB không thấy). Đây là điểm cực kỳ quan trọng.
  - **Quyết định:** dùng **annotation IR làm ground truth thống nhất** cho tất cả các nhánh thí nghiệm (đây cũng là quy ước phổ biến trong literature). Nếu mỗi nhánh dùng GT khác nhau thì so sánh mất ý nghĩa.
  - Hệ quả: baseline RGB-only sẽ bị phạt ở các box mà RGB không thể nhìn thấy. Điều này **phải được ghi rõ trong báo cáo** như một đặc điểm của giao thức, không phải lỗi.
  - ⚠️ **Hệ quả sâu hơn, bổ sung ở v1.1.** Chênh lệch box IR−RGB là 494.839 vs 447.833 (+10,5%), tập trung ở cảnh tối. Nghĩa là ở phân khúc thiếu sáng, RGB-only bị phạt trên **chính những box mà nó về nguyên tắc không thể phát hiện**. Do đó `F2 − S1` ở phân khúc tối **bị thổi phồng một phần do định nghĩa GT**, không thuần do fusion.
    - So sánh lõi `F2a − C2` **không bị ảnh hưởng** (cả hai đều thấy IR, cùng GT IR) → đây mới là con số để claim.
    - Trong báo cáo: `Δ so với S1` = giới hạn trên có confounder; `Δ so với C2` = con số trung thực. Thiếu câu này là bị bắt lỗi ngay.
  - Annotation RGB vẫn được xuất ra `labels_rgb/` để phân tích phụ nếu cần.

### 3.2 Dataset thứ hai (replication): VEDAI

| Thuộc tính | Giá trị |
|---|---|
| Số ảnh | **1.246 cặp** (đã tải và xử lý, subset 512) |
| Modality | RGB (3 kênh) + NIR (near-infrared, 1 kênh), đồng đăng ký |
| Độ phân giải | 1024 × 1024 và 512 × 512 (hai subset) |
| Lớp | **8** — car, pickup, camping, truck, other, tractor, boat, van (gộp 3 mã cực hiếm vào `other`) |
| Số instance | **3.746** |
| GSD | 12,5 cm/pixel |
| Giao thức | 10-fold cross-validation có sẵn |
| Nguồn | Utah AGRC / Đại học Caen |

**Vai trò:** không phải để gộp với DroneVehicle, mà để **replicate độc lập** toàn bộ ma trận thí nghiệm, tăng external validity.

**Lý do chọn:**
- Nhỏ, chạy nhanh → dùng làm môi trường pilot/debug pipeline trước khi tốn compute cho DroneVehicle
- Khác biệt có chủ đích với DroneVehicle: NIR (phản xạ) thay vì TIR (bức xạ nhiệt), nadir thay vì oblique, nông thôn/sa mạc thay vì đô thị, đối tượng cực nhỏ (~0,7% diện tích ảnh)
- Nếu kết luận giữ nguyên trên cả hai → rất vững. Nếu khác nhau → phần thảo luận trở nên thú vị hơn nhiều.

**Cảnh báo:**
- ⚠️ VEDAI dùng tiêu chí đánh giá phi chuẩn trong paper gốc (tâm dự đoán nằm trong ellipse của GT). **Không dùng tiêu chí này.** Chuyển sang mAP theo chuẩn COCO/DOTA để so sánh nhất quán với DroneVehicle, và ghi rõ trong báo cáo rằng số liệu không so trực tiếp được với paper gốc VEDAI.
- ⚠️ Chỉ ~1.100 ảnh train → variance giữa các seed sẽ lớn. Bắt buộc chạy nhiều fold/seed.
- Để tiết kiệm compute: dùng **3 fold** (01, 03, 05) thay vì đủ 10, coi mỗi fold như một seed (đã rút gọn ở v1.1).
- ✅ **Ưu điểm bổ sung phát hiện khi xử lý dữ liệu:** VEDAI chỉ có **một bộ GT dùng chung** cho cả hai modality → không tồn tại vấn đề "chọn GT nào", nên VEDAI là phép kiểm chứng **sạch hơn** cho H1. Đổi lại, VEDAI **không kiểm chứng được H2** (toàn ảnh ban ngày, NIR phản xạ chứ không phải bức xạ nhiệt). Ngoài ra RGB-NIR đồng đăng ký **tuyệt đối** (cùng ảnh chính trực) → là baseline δ=0 lý tưởng cho RQ4.


## 4. Lựa chọn mô hình

### 4.1 Detector nền: YOLO11-OBB (Ultralytics)

**Lý do:**

| Tiêu chí | Đánh giá |
|---|---|
| Hỗ trợ OBB native | ✅ Bắt buộc — cả hai dataset đều dùng oriented box |
| Chi phí tính toán | ✅ Train được trên 1 GPU consumer (RTX 3090/4090, Colab A100) |
| Dễ nhân đôi backbone | ✅ Kiến trúc định nghĩa bằng YAML, đăng ký module tùy chỉnh được |
| Tài liệu / cộng đồng | ✅ Rất tốt, giảm thời gian debug |
| Tính linh hoạt kiến trúc | ⚠️ Kém hơn mmrotate; sửa sâu hơi vướng |

**Biến thể:** `YOLO11s-obb` cho thí nghiệm chính. Dùng `YOLO11n-obb` cho pha pilot để chạy nhanh.

**Phương án dự phòng:** MMRotate + MMDetection. Linh hoạt hơn nhiều cho fusion mức feature, nhưng learning curve dốc và version hell là rủi ro thật. **Chỉ chuyển sang nếu việc sửa kiến trúc Ultralytics bị chặn cứng sau 1 tuần thử.**

### 4.2 Các cấu hình fusion

#### F1 — Early fusion (input-level)
Ghép kênh thành tensor 4 kênh `[R, G, B, IR]`, sửa conv đầu tiên từ `in_channels=3` thành `4`.
- Chi phí: ~bằng single-stream
- Cài đặt: dễ nhất, khoảng 20 dòng code
- Rủi ro: buộc hai modality chia sẻ toàn bộ filter ngay từ tầng đầu, dễ bị modality mạnh hơn lấn át

#### F2 — Mid fusion (feature-level)
Hai backbone song song (không chia sẻ trọng số), hợp nhất tại các mức feature P3/P4/P5 trước khi vào neck.

Toán tử hợp nhất — thử hai biến thể:
- **F2a:** `concat` → `Conv1x1` giảm chiều (baseline đơn giản, ổn định)
- **F2b:** cổng chú ý theo kênh kiểu CBAM/SE, học trọng số động cho từng modality (kỳ vọng tốt hơn khi điều kiện thay đổi)

- Chi phí: ~1,7–1,9× single-stream
- Đây là cấu hình kỳ vọng cho kết quả tốt nhất

#### F3 — Late fusion (decision-level)
Chạy hai detector đã train riêng (S1 và S2), hợp nhất box đầu ra.
- Toán tử: Weighted Box Fusion (WBF) với rotated IoU. Nếu rotated WBF quá phức tạp, fallback: chuyển sang axis-aligned IoU để tính độ trùng, giữ góc từ box có confidence cao hơn.
- Chi phí train: **0** (tái sử dụng weight của S1, S2). Chi phí inference: 2×.
- Là cấu hình bền nhất với misalignment → điểm chính của RQ4

### 4.3 Nhóm đối chứng capacity (bắt buộc)

**C1 / C2:** dùng **đúng kiến trúc hai luồng của F2**, nhưng nạp **cùng một modality vào cả hai luồng**.

- Cùng số tham số, cùng FLOPs, cùng độ sâu như F2
- Nhưng không có bất kỳ thông tin bổ sung nào từ cảm biến thứ hai
- **So sánh trung thực là `F2 vs C1`, không phải `F2 vs S1`**

Đây là đóng góp phương pháp luận chính của đề tài. Đừng bỏ qua.

---

## 5. Ma trận thí nghiệm

Chạy toàn bộ trên **cả hai** dataset.

| ID | Đầu vào | Kiến trúc | Params | Vai trò |
|---|---|---|---|---|
| S1 | RGB | 1 luồng | 1× | Baseline đơn lẻ |
| S2 | IR | 1 luồng | 1× | Baseline đơn lẻ |
| **C1** | RGB → cả 2 luồng | 2 luồng | 2× | **Đối chứng capacity** |
| **C2** | IR → cả 2 luồng | 2 luồng | 2× | **Đối chứng capacity** |
| F1 | RGB + IR | Early (4 kênh) | ~1× | Fusion |
| F2a | RGB + IR | Mid, concat+1×1 | 2× | Fusion |
| F2b | RGB + IR | Mid, attention gate | ~2× | Fusion |
| F3 | RGB + IR | Late (WBF từ S1+S2) | 2× (inference) | Fusion |

**8 cấu hình × 3 seed × 2 dataset**, trong đó F3 không tốn chi phí train.

### 5.1 So sánh chính

| Phép so sánh | Trả lời |
|---|---|
| `max(F1,F2a,F2b,F3)` vs `max(S1,S2)` | RQ1 — phiên bản "ngây thơ", so với literature |
| `F2a` vs `C2` | **RQ1 — phiên bản trung thực**, đã kiểm soát capacity |
| `C2` vs `S2` | Lợi ích thuần từ tăng capacity (giá trị chẩn đoán) |
| `F1` vs `F2a` vs `F2b` vs `F3` | RQ2 |
| Delta phân tầng theo ngày/đêm | RQ3 |
| F1/F2/F3 dưới các mức dịch chuyển IR | RQ4 |

### 5.2 Ablation RQ4 — độ nhạy với sai lệch đồng đăng ký

Cố tình dịch ảnh IR theo trục x/y một lượng `δ ∈ {0, 2, 5, 10, 20}` pixel **tại thời điểm test** (model train trên δ=0), đo mức suy giảm mAP của từng chiến lược fusion.

Giả thuyết: `late > mid > early` về độ bền. Kết quả này tự nó đã đáng báo cáo và có giá trị thực tiễn (hệ thống UAV thật hiếm khi đồng đăng ký hoàn hảo).

---

## 6. Giao thức đánh giá

### 6.1 Chỉ số

**Chính:** `mAP50` (OBB) — ⚠️ đổi so với v1.0.
**Phụ:** `mAP50-95` (OBB), AP theo từng lớp, Recall ở FPPI thấp, params, GFLOPs, FPS

> **Lý do đổi:** gần như toàn bộ literature trên DroneVehicle báo cáo AP50 kiểu VOC (SOTA ~83% là **mAP50**, không phải mAP50-95). Nếu lấy mAP50-95 làm chỉ số chính thì mất mốc neo để biết pipeline có đúng không ở sanity check mục 7.3. Vẫn báo cáo mAP50-95 kèm theo.

### 6.2 Phân tầng bắt buộc

Đây là phần quan trọng nhất của giao thức. **Không được chỉ báo cáo một con số mAP tổng.**

| Chiều phân tầng | Nhóm |
|---|---|
| **Chiếu sáng** | theo `illum_proxy` = p10 độ xám ảnh RGB: `lowlight` (<10) / `midlight` (10–55) / `bright` (≥55) — ⚠️ đổi so với v1.0 |
| Kích thước đối tượng | small (<32²) / medium (32²–96²) / large (>96²) theo chuẩn COCO |
| Lớp | 5 lớp riêng biệt (DroneVehicle mất cân bằng cực đoan: car chiếm 86%) |
| Góc chụp | ❌ bỏ — metadata không tin cậy (xem DATA_REPORT.md) |

> **⚠️ Đính chính quan trọng so với v1.0.** DroneVehicle **KHÔNG có sẵn metadata chiếu sáng**. Đã quét toàn bộ 57k file XML: chỉ có `folder/filename/size/object`, không có trường nào về ngày/đêm.
>
> Đã thử suy diễn nhãn day/night từ ảnh và **không đạt độ chính xác chấp nhận được**: cảnh đêm có đèn đường mạnh nhìn từ trên cao thực sự giống ngày âm u (mean luminance lên tới 105–130; luật 2 đặc trưng tốt hơn nhưng vẫn sai ~25% ở biên). Bằng chứng: `results/illumination/`.
>
> **Quyết định:** không tuyên bố nhãn "day/night". Phân tầng theo **độ chiếu sáng đo được** — một đại lượng liên tục, khách quan, không cần ground truth. Báo cáo chính nên là **đường cong Δ mAP theo decile của `illum_proxy`**; bảng tóm tắt dùng 3 bin ở trên.
>
> Cách này **mạnh hơn** nhãn day/night thủ công về mặt phương pháp: nó trả lời "fusion giúp bao nhiêu ở mỗi mức chiếu sáng" thay vì chỉ so hai nhóm thô. Cột `day_heuristic` vẫn được lưu nhưng chỉ để tham khảo, không dùng kết luận.

Phân bố thực tế trên tập test: `lowlight` 3.016 ảnh (33,6% ảnh — 43,7% số box) / `midlight` 3.180 (35,4% — 30,2%) / `bright` 2.784 (31,0% — 26,1%). Ba nhóm cân đối, đủ mẫu cho bootstrap ở từng tầng.

Kỳ vọng câu chuyện thật sẽ có dạng: *"fusion tăng 1,5 mAP tổng thể nhưng tăng 7–9 mAP ở decile chiếu sáng thấp nhất và gần như không đổi ở decile cao nhất."* Đó mới là kết luận đáng viết.

### 6.3 Xử lý thống kê

- **2 seed** cho mỗi cấu hình trên DroneVehicle; **3 fold** (01, 03, 05) trên VEDAI — đã rút gọn, xem 0.1
- Báo cáo `mean ± std`, **không bao giờ báo cáo số của một lần chạy đơn lẻ**
- Kiểm định: paired bootstrap trên tập test (10.000 lần lấy mẫu lại) để tính khoảng tin cậy 95% cho hiệu số giữa hai cấu hình
- **Ngưỡng ý nghĩa thực tiễn:** nếu khoảng tin cậy của `F2a − C2` chứa 0, kết luận là "không có bằng chứng về lợi ích của fusion sau khi kiểm soát capacity" — và đó vẫn là một kết quả hợp lệ, đáng báo cáo.

### 6.4 Kiểm soát công bằng

Tất cả cấu hình dùng chung:
- Cùng split, cùng seed cho thứ tự dữ liệu
- Cùng số epoch (60), cùng optimizer, cùng lịch learning rate (cosine)
- **`batch=8` và `nbs=32` cho MỌI cấu hình** — ⚠️ bổ sung ở v1.1, bắt buộc
- Cùng pipeline augmentation (⚠️ augmentation hình học phải áp dụng **đồng nhất** cho cả hai modality của một cặp — đây là bug rất dễ mắc)
- **`hsv_h=hsv_s=hsv_v=0` cho mọi cấu hình** — ⚠️ bổ sung ở v1.1, xem lý do bên dưới
- **Normalization riêng cho từng modality** (mean/std tính riêng trên RGB và trên IR — dùng chung là hỏng ngay)
- Budget tuning hyperparameter bằng nhau cho mọi cấu hình (nếu tune cho F2 mà không tune cho C2 thì so sánh mất giá trị)

> **Vì sao batch phải bằng nhau — một cái bẫy đã đo được.** Trên RTX 3060 Ti 8GB: 1 luồng bs16@640 chiếm 3,9GB (chạy được), 2 luồng bs16@640 cần ~7,7GB (không vừa). Nếu để single-stream chạy bs16 còn two-stream bs8 thì **đã vi phạm chính mục này**: batch size khác nhau đổi cả effective learning rate lẫn thống kê BatchNorm, và chênh lệch đó sẽ bị quy nhầm cho fusion. Giải pháp: bs=8 cho tất cả, `nbs=32` để gradient accumulate về cùng nominal batch. Chi phí không đáng kể (1 luồng bs8 đạt 109 img/s vs bs16 116 img/s).

> **Vì sao phải tắt HSV augmentation.** `RandomHSV` của Ultralytics gọi `cv2.cvtColor` giả định ảnh 3 kênh → **vỡ với tensor 4 kênh (F1) và 6 kênh (F2/C1/C2)**. Phải tắt cho *mọi* cấu hình, kể cả S1/S2, nếu không thì các nhánh không còn dùng chung pipeline augmentation.

---

## 7. Cấu trúc code

### 7.1 Layout repository

> Đã triển khai. Khác dự kiến ở vài chỗ, đều có lý do:
> `src/patches.py` (điểm mở rộng Ultralytics) và `src/train.py` (trainer dùng chung)
> là hai file không có trong bản phác thảo nhưng cần thiết; `stems.py` không cần
> vì việc tách kênh do lớp `Index` trong đồ thị model đảm nhiệm.

```
uav_fusionsensor/
├── configs/
│   ├── base.yaml                 # siêu tham số dùng chung
│   ├── datasets/                 # 4 view: {dronevehicle,vedai} × {rgb,ir}
│   └── experiments/              # 8 cấu hình, sinh bởi gen_experiment_configs.py
├── src/
│   ├── patches.py                # 3 bản vá Ultralytics (có 1 bản vá SỬA LỖI, xem 0.4)
│   ├── train.py                  # trainer + runner cho một ô ma trận
│   ├── data/paired_dataset.py    # xếp chồng modality, dịch IR cho RQ4
│   ├── models/
│   │   ├── two_stream.py         # sinh kiến trúc two-stream từ YAML gốc
│   │   ├── fusion_ops.py         # GatedFusion (F2b)
│   │   └── build.py              # dựng model + nạp pretrained công bằng
│   ├── fusion/wbf_obb.py         # late fusion cho box xoay
│   ├── eval/
│   │   ├── metrics.py            # mAP OBB, IoU đa giác chính xác
│   │   ├── stratify.py           # phân tầng chiếu sáng / kích thước / lớp
│   │   ├── bootstrap.py          # CI ghép cặp theo ảnh, công thức có trọng số
│   │   └── harness.py            # suy luận + kết xuất + đánh giá phân tầng
│   └── utils/seed.py             # seed + provenance
├── tests/                        # 44 test
├── scripts/
│   ├── prepare_dronevehicle.py   scan_labels.py       compute_illum_features.py
│   ├── prepare_vedai.py          label_illumination.py  make_yolo_views.py
│   ├── check_alignment.py        draw_gt.py           sheet_by_feature.py
│   ├── gen_experiment_configs.py smoke_test_data.py   smoke_pipeline.py
│   ├── run_experiment.py         run_matrix.py        evaluate_run.py
│   ├── run_late_fusion.py        run_rq4_ablation.py  make_tables.py
│   └── diag_metrics.py           # chẩn đoán chênh lệch chỉ số (xem 0.4)
├── data/                         # dronevehicle/, vedai/ (đã xử lý)
├── results/                      # alignment/, illumination/, gt_check/
├── plan.md  DATA_REPORT.md  README.md
```

### 7.2 Nguyên tắc

- **Config-driven hoàn toàn.** Mọi cấu hình thí nghiệm là một file YAML. Không sửa code để đổi thí nghiệm.
- **Seed cố định và ghi lại.** `torch`, `numpy`, `random`, `cudnn.deterministic`.
- **Ghi lại git commit hash + config hash trong mỗi run.** Không có cái này thì không reproduce được sau 2 tháng.
- **Logging tập trung.** Weights & Biases hoặc TensorBoard, một project cho toàn bộ ma trận.
- Không bao giờ đọc kết quả từ stdout — mọi số liệu phải ghi ra JSON để `make_tables.py` tổng hợp.

### 7.3 Evaluation harness và test — làm TRƯỚC khi train

Đây là phần dễ bị bỏ qua nhất và cũng là nguyên nhân số một khiến kết quả thí nghiệm phải làm lại từ đầu. **Viết trước khi chạy bất kỳ training run nào.**

Bộ test tối thiểu, dùng fixture gồm 20 cặp ảnh đã kiểm tra thủ công:

| Test | Kiểm tra gì | Vì sao quan trọng |
|---|---|---|
| `test_pairing` | Với mọi index i, `x_rgb[i]` và `x_ir[i]` đúng là cùng một cảnh | Lệch cặp → toàn bộ thí nghiệm vô nghĩa mà không có dấu hiệu báo lỗi |
| `test_augment_sync` | Sau random flip/rotate, box vẫn khớp với **cả hai** ảnh | Bug phổ biến nhất trong pipeline paired |
| `test_border_removal` | Ảnh DroneVehicle sau tiền xử lý đúng 640×512, không còn pixel viền trắng | Model học artifact viền nếu quên |
| `test_normalization` | RGB và IR dùng mean/std khác nhau, đúng giá trị đã tính | Dùng chung stats làm IR bị lệch phân phối nặng |
| `test_metrics_known` | mAP tính trên GT-vs-GT phải bằng 1.0; trên dự đoán rỗng phải bằng 0.0 | Sanity check hàm metric |
| `test_wbf_identity` | WBF hai bản sao giống hệt nhau phải trả về chính nó | Bắt lỗi logic hợp nhất box |
| `test_capacity_control_exact_match` | Số param C1 == C2 == F2a **bằng nhau tuyệt đối** | Đạt chặt hơn dự kiến: cả ba dùng chung `arch_spec` nên bằng nhau theo cấu trúc, không cần dung sai |
| `test_augment_sync_border_fill` | Vùng viền sau biến đổi hình học cùng giá trị ở mọi kênh | **Đã bắt được lỗi thật** — xem mục 0.4 #1 |
| `test_bootstrap_weighted_matches_naive` | Công thức bootstrap có trọng số ≡ nhân bản vật lý | Tối ưu hai bậc độ lớn, phải chứng minh không đổi kết quả |
| `test_pretrained_symmetric_between_streams` | Hai nhánh nhận cùng trọng số pretrained | Nếu lệch, C1/C2/F2a không xuất phát từ cùng một điểm |

**Sanity check bắt buộc trước khi chạy ma trận đầy đủ:**
1. Chồng 30 cặp RGB-IR ngẫu nhiên, kiểm tra bằng mắt mức đồng đăng ký. Ghi lại độ lệch quan sát được.
2. Vẽ box GT lên cả hai modality cho 30 mẫu, kiểm tra bằng mắt.
3. Train S2 (IR-only) đến hội tụ, đối chiếu mAP với con số công bố trong literature (SOTA gần đây trên DroneVehicle ~83% mAP; nếu bạn chỉ ra 40% thì pipeline có bug, không phải model kém).
4. Overfit có chủ đích trên 50 mẫu — nếu không đạt gần 100% train mAP thì có bug ở loss/label.

---

## 8. Kế hoạch theo tuần (12 tuần)

### Pha 0 — Nền tảng (Tuần 1–2)

| Tuần | Việc | Deliverable |
|---|---|---|
| 1 | Tải và tiền xử lý DroneVehicle + VEDAI. Viết parser, cắt viền trắng, chuyển annotation sang định dạng YOLO-OBB. Kiểm tra trực quan đồng đăng ký. | Dataset sạch trên đĩa + báo cáo kiểm tra 30 mẫu |
| 2 | Dựng khung repo, config system, logging. **Viết toàn bộ evaluation harness và test suite.** Xác minh metric bằng fixture. | `pytest` xanh toàn bộ; harness chạy được trên model random |

**Cổng kiểm soát G0:** không sang pha 1 nếu test suite chưa xanh.

### Pha 1 — Pilot trên VEDAI (Tuần 3–4)

| Tuần | Việc | Deliverable |
|---|---|---|
| 3 | Cài đặt S1, S2, F1 (single-stream + early). Chạy trên VEDAI fold 01. Debug end-to-end. | 3 cấu hình chạy thông |
| 4 | Cài đặt two-stream (C1, C2, F2a). Đây là phần kỹ thuật khó nhất — sửa YAML kiến trúc Ultralytics, đăng ký module tùy chỉnh. Chạy pilot. | 6 cấu hình chạy thông; xác nhận param count của C1 == F2a |

**Cổng kiểm soát G1:** nếu hết tuần 4 mà two-stream chưa chạy được trên Ultralytics → kích hoạt phương án dự phòng MMRotate (xem mục 10).

### Pha 2 — Thí nghiệm chính (Tuần 5–8)

| Tuần | Việc | Deliverable |
|---|---|---|
| 5 | Cài F2b (attention) và F3 (WBF-OBB). Hoàn thiện ma trận trên VEDAI, 4 fold. | Bảng kết quả VEDAI đầy đủ |
| 6–7 | Chạy ma trận đầy đủ trên DroneVehicle, 3 seed. Đây là phần tốn compute nhất. | Log của 21 training run |
| 8 | Ablation RQ4 (độ nhạy misalignment). Chạy inference-only, rẻ. | Bảng độ bền theo δ |

### Pha 3 — Phân tích và viết (Tuần 9–12)

| Tuần | Việc | Deliverable |
|---|---|---|
| 9 | Phân tầng kết quả theo ngày/đêm, size, class. Bootstrap CI. Sinh toàn bộ bảng và biểu đồ. | `results/tables/`, `results/figures/` |
| 10 | Phân tích định tính: chọn ~20 trường hợp fusion thắng và ~20 trường hợp fusion thua, tìm quy luật. | Phần qualitative analysis |
| 11 | Viết báo cáo. | Draft đầy đủ |
| 12 | Dọn repo, viết README tái lập, hoàn thiện báo cáo, chuẩn bị slide. | Bản nộp cuối |

**Buffer:** kế hoạch không có buffer rõ ràng. Thực tế nên coi tuần 12 là buffer và hoàn thành ở tuần 11.

---

## 9. Ngân sách tính toán

### 9.1 Thông lượng đã ĐO trên phần cứng thật

Benchmark forward+backward YOLO11s-obb, AMP, trên RTX 3060 Ti 8GB (không phải ước tính):

| Cấu hình | ms/iter | img/s | Peak VRAM |
|---|---|---|---|
| 1 luồng, bs16, 640 | 138 | 116 | 3,91 GB |
| 1 luồng, bs8, 640 | 74 | 109 | 2,00 GB |
| **2 luồng, bs8, 640** | 147 | **55** | 3,87 GB |
| 2 luồng, bs16, 640 | — | — | ~7,7 GB ⚠️ không vừa |
| 2 luồng, bs8, 512 | 99 | 81 | 2,49 GB |

Chưa gồm OBB loss, dataloader và val mỗi epoch → cộng thêm ~30–40%. CPU (i5-13600K, 20 luồng) và RAM không phải nút thắt; **GPU là nút thắt duy nhất**.

### 9.2 Ngân sách theo cấu hình đã chốt (60 epoch, 2 seed, bs8)

**DroneVehicle (17.951 ảnh train)**

| Cấu hình | Loại | Giờ/run | Số run | Tổng giờ |
|---|---|---|---|---|
| S1, S2, F1 | 1 luồng | ~5 | 6 | 30 |
| C1, C2, F2a | 2 luồng | ~9,5 | 6 | 57 |
| F2b | 2 luồng | ~9,5 | (2) | (19) — hoãn |
| F3 | inference-only | ~1 | 2 | 2 |
| | | | **Tổng** | **~89 h** (108 h nếu làm F2b) |

**VEDAI (1.089 ảnh train × 3 fold)** — nhỏ nhưng cần nhiều epoch hơn: **~30 h**

**Tổng cộng: ~120 GPU-hours** (~140 h nếu làm F2b).
Ở 8h/ngày là **~15–18 ngày lịch** → vừa pha 2 (4 tuần) và còn dư slack cho một lần train lại.

*So sánh: phương án đầy đủ v1.0 (100 epoch, 3 seed) trên chính GPU này sẽ tốn ~330 h ≈ 30–40 ngày lịch — không khả thi.*

### 9.3 Nếu vẫn thiếu compute

Áp dụng theo thứ tự, dừng khi đủ:
1. Giảm imgsz 640 → 512 (đo được: 2 luồng nhanh hơn 1,5×, tiết kiệm ~33%)
2. Giảm YOLO11s → YOLO11n (~40%) — **đánh đổi: mất khả năng đối chiếu literature**
3. Subsample DroneVehicle train xuống 9.000 ảnh (~50%) — ghi rõ trong báo cáo

**Không được cắt:** C1, C2 (đối chứng capacity) và việc chạy ≥2 seed. Cắt hai thứ này là cắt vào đúng phần làm nên giá trị của đề tài.

---

## 10. Sổ đăng ký rủi ro

| ID | Rủi ro | Khả năng | Tác động | Giảm thiểu / Phương án dự phòng |
|---|---|---|---|---|
| R1 | Không sửa được kiến trúc two-stream trên Ultralytics | Trung bình | Cao | Timebox 1 tuần (G1). ⚠️ **MMRotate KHÔNG còn là fallback khả dụng** — môi trường là Python 3.14 + torch 2.11, còn mmcv pin torch <2.1 và không có wheel cho 3.14; build từ nguồn trên Windows không khả thi. Fallback thực tế: (a) đường 6 kênh ở mục 4.2 — ít ma sát nhất; (b) detector two-stream tối giản trên PyTorch thuần (timm backbone + head OBB) |
| R2 | Cặp RGB-IR đồng đăng ký kém → mid fusion thua single-modality | **Đã đo — xác nhận có thật** | Trung bình | Phase correlation trên 300 cặp val: lệch trung vị **1,8 px**, p90 **7,2 px**, **20,9% số cặp lệch >5 px**, 5,1% >10 px (ước lượng tin cậy, resp>0,3). So với xe dài ~30–50 px thì đây là mức đáng kể. → dải δ ∈ {0,2,5,10,20} của RQ4 là phù hợp và có căn cứ đo đạc. Báo cáo con số này như một đặc điểm của dataset |
| R3 | Nhầm cặp trong data loader (silent bug) | Thấp | **Rất cao** | ✅ Đã kiểm chứng ở mức dữ liệu: NMI(cặp khớp)=0,033 vs NMI(cặp ngẫu nhiên)=0,011, 90,7% số cặp khớp ăn đứt cặp ngẫu nhiên → ghép cặp theo chỉ số là ĐÚNG. ⚠️ Không dùng trường `filename` trong XML làm khóa ghép cặp: dataset có ≥5 quy ước đặt tên khác nhau và nhiều file RGB mang tên `..._R`. `test_pairing` phải dựa trên ảnh, không dựa trên tên |
| R4 | Augmentation không đồng bộ giữa hai modality | Trung bình | Cao | `test_augment_sync`. Viết transform nhận cặp làm đơn vị nguyên tử, không transform từng ảnh riêng |
| R5 | Vượt ngân sách compute | Trung bình | Trung bình | Phương án rút gọn ở mục 9. Theo dõi giờ tích lũy hàng tuần |
| R6 | Fusion không thắng đối chứng capacity | Trung bình | Thấp | **Đây là kết quả hợp lệ, không phải thất bại.** Đóng góp trở thành "chứng minh rằng lợi ích fusion báo cáo trong literature có phần đáng kể đến từ tăng capacity". Cần chuẩn bị tinh thần và cách viết cho tình huống này ngay từ đầu |
| R7 | WBF cho oriented box phức tạp hơn dự kiến | Trung bình | Thấp | Fallback: dùng axis-aligned IoU để tính độ trùng, giữ góc từ box confidence cao hơn. Ghi rõ giới hạn |
| R8 | Mất cân bằng lớp cực đoan (car **85,8%**, đã đếm chính xác) làm AP các lớp hiếm nhiễu loạn | Cao | Trung bình | Luôn báo cáo per-class AP kèm số instance. Không dùng macro-mAP làm chỉ số duy nhất |
| R9 | Scope creep — muốn thêm modality/dataset/kiến trúc | Cao | Trung bình | Khóa ma trận thí nghiệm sau tuần 4. Mọi ý tưởng mới ghi vào "Future work", không thực hiện |
| **R10** | **Không có metadata chiếu sáng → RQ3 mất nền** | **Đã xảy ra** | Cao | ✅ Đã xử lý: dataset thật sự không có trường này (quét 57k XML). Thay bằng phân tầng theo `illum_proxy` đo được — xem 6.2. Rủi ro đã đóng, nhưng phải ghi rõ trong báo cáo rằng phân tầng là đại lượng suy diễn từ ảnh |
| **R11** | **Archive gốc trên upstream có file hỏng** | **Đã xảy ra** | Thấp | ✅ Đã xử lý: 39/71.960 file trong `train.zip` lỗi CRC. sha256 của bản tải **khớp** blob HF → hỏng từ nguồn, không phải do tải. Loại 39 cặp (0,22% train); val/test nguyên vẹn 100% nên **đánh giá không bị ảnh hưởng**. Danh sách: `data/corrupt_files.json` |

---

## 11. Tiêu chí thành công và tiêu chí dừng

### Thành công tối thiểu (đủ để nộp)
- Ma trận 8 cấu hình chạy đủ trên ≥1 dataset với ≥2 seed
- Có bảng so sánh `F2a vs C2` kèm khoảng tin cậy
- Có phân tầng theo ít nhất chiều chiếu sáng
- Repo tái lập được từ README

### Thành công tốt
- Đủ trên cả 2 dataset, 3 seed
- Hoàn thành ablation RQ4
- Kết luận nhất quán hoặc có giải thích thuyết phục cho sự khác biệt giữa hai dataset

### Thành công xuất sắc (đủ chất lượng workshop paper)
- Tất cả điều trên, cộng thêm phân tích định lượng về *khi nào* fusion giúp ích, có thể dự đoán được từ đặc trưng của mẫu
- Phân tích định tính có hệ thống về các chế độ thất bại

### Tiêu chí dừng (kill criteria)
- Hết tuần 4 chưa có 6/8 cấu hình chạy được → cắt bỏ F2b và F3, chỉ giữ S1/S2/C1/C2/F1/F2a
- Hết tuần 7 chưa xong DroneVehicle → chuyển VEDAI thành dataset chính, DroneVehicle thành phần phụ

---

## 12. Việc cần làm ngay (tuần này)

**Đã xong (2026-08-19):**

- [x] DroneVehicle: tải, xác minh sha256, giải nén, cắt viền, chuẩn hoá nhãn, xuất YOLO-OBB — 28.400 cặp
- [x] VEDAI: tải từ `downloads.greyc.fr/vedai`, giải nén, chuyển annotation, tách 10 fold — 1.246 cặp
- [x] Xác nhận GPU thực tế (RTX 3060 Ti 8GB) → **chốt phương án rút gọn**, xem 0.1 và mục 9
- [x] Khởi tạo repo, dựng skeleton theo mục 7.1
- [x] Đo đồng đăng ký RGB-IR (300 cặp, phase correlation + 30 ảnh overlay) → R2 đã đóng
- [x] Kiểm chứng ghép cặp bằng NMI → R3 đã đóng ở mức dữ liệu
- [x] Vẽ box GT lên cả hai modality, kiểm tra bằng mắt → `results/gt_check/`
- [x] Smoke test: Ultralytics nạp được cả 4 view, 0 file lỗi, toạ độ trong [0,1]
- [x] Chốt detector: **Ultralytics** (MMRotate không dùng được trên Python 3.14 — xem R1)

**Tiếp theo:**

- [ ] Viết evaluation harness + test suite (mục 7.3) — **cổng G0, làm trước mọi training run**
- [ ] Train S2 (IR-only) trên DroneVehicle đến hội tụ, đối chiếu mAP50 với literature (~83%) — sanity check pipeline
- [ ] Overfit có chủ đích 50 mẫu để bắt lỗi loss/label
- [ ] Dựng prototype 6 kênh trên VEDAI (nhanh) trước khi đụng DroneVehicle

---

## 13. Ghi chú về đóng góp khoa học

Cần thẳng thắn: **fusion RGB-IR cho phát hiện đối tượng từ UAV không phải chủ đề mới.** DroneVehicle và VEDAI đều là benchmark đã được khai thác nhiều, với hàng chục phương pháp fusion công bố.

Đóng góp của đề tài này **không nằm ở việc đề xuất kiến trúc fusion mới**, mà ở:

1. **Kiểm soát capacity** — phần lớn paper fusion so sánh model 2 backbone với model 1 backbone và quy toàn bộ chênh lệch cho fusion. Đề tài này tách bạch hai nguồn đóng góp.
2. **Phân tầng theo điều kiện** — trả lời "khi nào fusion đáng dùng" thay vì "fusion có tốt không".
3. **Độ bền với misalignment** — có giá trị thực tiễn cho hệ thống UAV triển khai thật.
4. **Replication trên hai dataset có đặc tính khác biệt có chủ đích** (TIR vs NIR, đô thị vs nông thôn, oblique vs nadir).

Nếu viết báo cáo, hãy định vị đề tài là **empirical study / reproducibility study**, không phải method paper. Định vị đúng sẽ tránh được câu hỏi "đâu là điểm mới của kiến trúc" mà đề tài không có ý định trả lời.
