# Báo cáo chuẩn bị dữ liệu

**Ngày:** 2026-08-19 · **Trạng thái:** hoàn tất, smoke test PASS · Đi kèm [plan.md](plan.md) v1.1

---

## 1. Tóm tắt

| | DroneVehicle | VEDAI (512) |
|---|---|---|
| Cặp RGB-IR dùng được | **28.400** | **1.246** |
| — train / val / test | 17.951 / 1.469 / 8.980 | 10 fold × (1.089 / 121) |
| Box GT dùng để train | **494.839** (annotation IR) | **3.746** (GT chung) |
| Lớp | 5 | 8 |
| Kích thước ảnh | 640 × 512 (đã cắt viền) | 512 × 512 |
| Cặp bị loại | 39 (0,22%, chỉ ở train) | 0 |
| Dung lượng sau xử lý | 14,5 GB | 1,7 GB |

Cả hai dataset đã ở định dạng YOLO-OBB, Ultralytics nạp được, **0 file lỗi, toạ độ 100% trong [0,1]**.

---

## 2. DroneVehicle

### 2.1 Nguồn và tính toàn vẹn

Tải từ Hugging Face `McCheng/DroneVehicle` (3 zip, 14 GB).

**Đã xác minh sha256 của cả 3 blob — khớp chính xác** với tên blob HF (HF đặt tên blob theo sha256), tức bản tải về không hỏng.

Tuy vậy quét CRC toàn bộ 113.756 entry phát hiện:

| Archive | Entry | Lỗi CRC |
|---|---|---|
| `val.zip` | 5.876 | **0** |
| `test.zip` | 35.920 | **0** |
| `train.zip` | 71.960 | **39** |

→ **Hỏng từ nguồn upstream, không phải do tải.** 39 file ảnh lỗi thuộc 39 cặp khác nhau, toàn bộ nằm trong tập train (0,22%). **Tập val và test nguyên vẹn 100% nên đánh giá không bị ảnh hưởng.**

Danh sách đầy đủ: `data/corrupt_files.json`. Các cặp bị loại: `02220-02223, 04895-04898, 04991-04994, 05001-05003, 05598-05599, 06463-06467, 09012-09014, 12900-12902, 14035-14038, 14048-14050, 16009-16012`.

### 2.2 Kiểm kê nhãn — đối chiếu với README gốc

Quét toàn bộ 57.878 file XML (`scripts/scan_labels.py`):

| Lớp | Tổng | RGB | IR | README gốc (RGB / IR) |
|---|---|---|---|---|
| car | 817.926 | 389.808 | 428.118 | 389.779 / 428.086 |
| truck | 48.085 | 22.124 | 25.961 | 22.123 / 25.960 |
| bus | 31.924 | 15.334 | 16.590 | 15.333 / 16.590 |
| van | 24.643 | 11.935 | 12.708 | 11.935 / 12.708 |
| freight car | 30.583 | 13.405 | 17.178 | 13.400 / 17.173 |

Lệch < 0,01% so với README chính thức → **dữ liệu toàn vẹn, đúng bản gốc**.

Mất cân bằng lớp: **car chiếm 85,8%** (plan v1.0 ghi ~80%).

### 2.3 Các vấn đề dữ liệu phải xử lý

| # | Vấn đề | Quy mô | Cách xử lý |
|---|---|---|---|
| 1 | Viền trắng 100px mỗi cạnh | 100% ảnh | Cắt `[100:-100, 100:-100]` → 640×512. Đã xác minh: 4 cạnh đều đúng giá trị 255 |
| 2 | **Tên lớp không nhất quán** | 30.583 box | `feright_car` / `feright car` / `feright` → `freight_car` (typo trong dataset gốc: "feright" thay vì "freight") |
| 3 | **Typo lớp khác** | 1 box | `truvk` → `truck` |
| 4 | **Nhãn rác** | 2 box | lớp `*` → loại bỏ |
| 5 | **Hình học lẫn lộn** | **65.184 box (6,8%)** dùng `<bndbox>` axis-aligned thay vì `<polygon>` 4 góc | Chuyển bndbox thành 4 góc hình chữ nhật |
| 6 | Object không có hình học | 75 box | Loại bỏ |
| 7 | Trường `<size><depth>` sai | 141/1469 file val | Bỏ qua XML, đọc kích thước thật từ ảnh |
| 8 | IR lưu 3 kênh | hầu hết ảnh | Đã kiểm chứng 3 kênh **giống hệt nhau** (max\|B−R\| = 0) → lấy 1 kênh |
| 9 | Box vượt ra vùng viền | 12,8% box | Box có **tâm** ngoài khung → loại (1,2%); box chỉ **chạm** viền → clip toạ độ |

### 2.4 ⚠️ Không có metadata chiếu sáng

**DroneVehicle không cung cấp nhãn ngày/đêm.** XML chỉ chứa `folder / filename / path / size / segmented / object`. Trường `folder` không dùng được: giá trị lộn xộn (`B`, `image`, `modalA`, `5-15 (2)`, `39_120m45_2`), không phải mã phiên chụp nhất quán.

Đây là rủi ro nghiêm trọng với RQ3 vì plan xếp H2 (phân tầng ngày/đêm) quan trọng hơn H1. Đã thử hai cách suy diễn:

**Cách 1 — độ sáng trung bình.** Thất bại. Histogram lưỡng đỉnh rõ, nhưng ở ngưỡng L=105 các ảnh **vẫn là cảnh đêm có đèn đường mạnh** (thấy rõ vệt đèn pha, ám vàng đèn cao áp). Đèn nhân tạo kéo mean lên 105–130. Bằng chứng: `results/illumination/sheet_boundary_day.jpg`.

**Cách 2 — luật 2 đặc trưng** (`p10 ≥ 55` và `cast ≤ 0.15`). Tốt hơn nhưng **vẫn sai ~25% ở biên**: cảnh đêm được chiếu sáng mạnh nhìn từ trên cao thực sự giống ngày âm u. Bằng chứng: `results/illumination/rule_day_boundary.jpg`.

**Quyết định: không tuyên bố nhãn day/night.** Phân tầng theo **độ chiếu sáng đo được**:

```
illum_proxy = p10 (phân vị 10%) của độ xám ảnh RGB đã cắt viền
```

p10 tốt hơn mean vì ban ngày toàn cảnh được chiếu sáng nên phân vị thấp vẫn cao, còn ban đêm chỉ vùng quanh nguồn sáng mới sáng. p10 miễn nhiễm với việc bị vài đèn pha kéo lệch.

| Bin | Ngưỡng | test: ảnh | test: box IR |
|---|---|---|---|
| `lowlight` | p10 < 10 | 3.016 (33,6%) | 69.128 (43,7%) |
| `midlight` | 10 ≤ p10 < 55 | 3.180 (35,4%) | 47.694 (30,2%) |
| `bright` | p10 ≥ 55 | 2.784 (31,0%) | 41.305 (26,1%) |

Ba nhóm cân đối, đủ mẫu để bootstrap ở từng tầng.

**Cách này mạnh hơn nhãn day/night thủ công về phương pháp luận:** nó cho phép báo cáo *đường cong* Δ mAP theo decile chiếu sáng — trả lời "fusion giúp bao nhiêu ở mỗi mức sáng" thay vì chỉ so hai nhóm thô. Cột `day_heuristic` vẫn được lưu nhưng chỉ tham khảo.

Nếu vẫn muốn nhãn day/night thật: gán tay ~500 ảnh test phân tầng, huấn luyện classifier nhỏ trên các đặc trưng đã tính sẵn (`meta/*_feat.csv`). Chi phí ~2 giờ.

### 2.5 Kiểm chứng ghép cặp (R3)

⚠️ **Trường `filename` trong XML KHÔNG dùng làm khoá ghép cặp được.** Dataset có ít nhất 5 quy ước đặt tên (`DJI_#` ↔ `DJI_#_R`, `DJI_#_#`, `/DJI_#_R.jpg`, và một số file **RGB** lại mang tên `..._R`). Quy luật phổ biến nhất là số hiệu IR = số hiệu RGB − 1 (cảm biến kép ghi số liên tiếp) nhưng chỉ đúng cho 902/1469 file val.

Vì vậy kiểm chứng bằng **ảnh**, dùng normalized mutual information:

| | NMI trung bình |
|---|---|
| Cặp khớp theo chỉ số `(RGB_i, IR_i)` | **0,0332** |
| Cặp ngẫu nhiên `(RGB_i, IR_j)` | 0,0112 |
| Tỉ lệ cặp khớp > cặp ngẫu nhiên | **90,7%** (272/300) |

→ **Ghép cặp theo chỉ số là đúng.** (9% còn lại là ảnh RGB quá tối nên NMI mất ý nghĩa, không phải lệch cặp.)

`test_pairing` trong test suite phải dựa trên ảnh, **không** dựa trên tên file.

### 2.6 Đo đồng đăng ký (R2) — có sai lệch thật

Phase correlation trên ảnh gradient (bất biến với chênh lệch cường độ giữa hai phổ), 300 cặp val:

| Tập con | n | Lệch trung vị | p90 | >5px | >10px |
|---|---|---|---|---|---|
| Tất cả | 300 | 2,36 px | 11,10 | 27,7% | 11,7% |
| **Ước lượng tin cậy (resp>0,3)** | 158 | **1,78 px** | **7,23** | **20,9%** | **5,1%** |
| Ảnh sáng (lum>60, resp>0,3) | 121 | 1,79 px | 6,40 | 19,8% | 3,3% |
| Ảnh tối (lum<20) | 42 | 3,10 px | 48,93 | 33,3% | 21,4% |

Tương quan `lum` vs `resp` = 0,56 → trên ảnh RGB tối, phase correlation không có gradient để bám nên ước lượng không đáng tin. **Con số nên trích dẫn là dòng "ước lượng tin cậy": trung vị ~1,8 px, p90 ~7,2 px, 20,9% số cặp lệch quá 5 px.**

So với xe dài ~30–50 px trong ảnh, mức này là đáng kể. Kết luận:

- DroneVehicle đồng đăng ký **tốt nhưng không hoàn hảo** — trái với giả định "đã đồng đăng ký sẵn" ở plan mục 3.1
- Dải `δ ∈ {0, 2, 5, 10, 20}` px của ablation RQ4 là **phù hợp và có căn cứ đo đạc**, không phải chọn tuỳ tiện
- Đây là số liệu đáng đưa vào báo cáo như một đặc điểm của dataset

30 ảnh overlay (biên RGB màu đỏ chồng trên nền IR) để soi bằng mắt: `results/alignment/overlay_*.jpg`. Số liệu thô: `results/alignment/report.json`.

---

## 3. VEDAI

### 3.1 Nguồn và quy mô

Tải từ `downloads.greyc.fr/vedai` (chính thức, Đại học Caen). Đã lấy cả hai độ phân giải; **dùng subset 512** làm chính (khớp vai trò pilot, và là subset các công trình multimodal gần đây dùng).

- **1.246 cặp khớp hoàn hảo** (`_co.png` / `_ir.png`), 0 ảnh lẻ
- 3.757 dòng annotation → **3.746 box** sau khi loại 11 box có tâm ngoài khung; 47 box bị clip
- 10 fold có sẵn: mỗi fold 1.089 train / 121 test
- IR lưu 3 kênh nhưng **1.246/1.246 ảnh có 3 kênh giống hệt nhau** → lấy 1 kênh
- 0 ảnh rỗng nhãn

### 3.2 Ánh xạ lớp

Annotation gốc có 11 mã lớp, trong đó 3 mã cực hiếm. Gộp theo thông lệ của các công trình multimodal trên VEDAI → **8 lớp**:

| Lớp | Số box | Mã gốc |
|---|---|---|
| car | 1.377 | 1 |
| pickup | 955 | 11 |
| camping | 397 | 5 |
| truck | 307 | 2 |
| other | 259 | 10 + 7 (4 box) + 8 (3 box) + 31 (48 box, plane) |
| tractor | 190 | 4 |
| boat | 171 | 23 |
| van | 101 | 9 |

Plan v1.0 ghi 9 lớp; thực tế gộp còn 8 vì hai mã chỉ có 3–4 mẫu, không đủ để tính AP có ý nghĩa.

### 3.3 Khác biệt quan trọng so với DroneVehicle

**VEDAI chỉ có MỘT bộ ground truth dùng chung cho cả hai modality.** Không tồn tại vấn đề "chọn GT nào".

Hệ quả cho thiết kế thí nghiệm:

- VEDAI là phép kiểm chứng **sạch hơn** cho H1 — không có confounder do chọn GT
- Nhưng VEDAI **không kiểm chứng được H2**: toàn bộ ảnh chụp ban ngày, NIR là phổ phản xạ chứ không phải bức xạ nhiệt
- RGB và NIR **đồng đăng ký tuyệt đối** (cùng ảnh chính trực) → VEDAI cũng là baseline δ=0 lý tưởng cho RQ4

---

## 4. Cấu trúc đầu ra

```
data/
├── raw/                          # DroneVehicle giải nén nguyên bản
├── dronevehicle/
│   ├── images/{train,val,test}/  # RGB 640×512
│   ├── imagesr/{train,val,test}/ # IR  640×512 grayscale
│   ├── labels/{train,val,test}/  # YOLO-OBB từ annotation IR  ← GT dùng để train
│   ├── labels_rgb/{...}/         # YOLO-OBB từ annotation RGB ← chỉ phân tích phụ
│   ├── rgb/{images,labels}/      # view hardlink cho Ultralytics
│   ├── ir/{images,labels}/       # view hardlink cho Ultralytics
│   └── meta/
│       ├── {split}.csv           # thống kê làm sạch từng ảnh
│       ├── {split}_feat.csv      # 12 đặc trưng chiếu sáng
│       ├── {split}_illum.csv     # illum_proxy + bin phân tầng
│       └── summary.json
├── vedai/
│   ├── images/, imagesr/, labels/
│   ├── rgb/, ir/                 # view hardlink + danh sách theo fold
│   ├── folds/fold{NN}_{train,test}.txt
│   └── meta/summary.json
└── corrupt_files.json
```

View `rgb/` và `ir/` dùng **hard link** (không tốn thêm dung lượng) vì Ultralytics suy ra đường dẫn nhãn bằng cách thay `/images/` → `/labels/`, mà thư mục `imagesr/` không khớp quy ước đó.

Cả hai view dùng **chung một bộ nhãn** (annotation IR với DroneVehicle).

---

## 5. Script

| Script | Việc |
|---|---|
| `scripts/scan_labels.py` | Quét toàn bộ XML, liệt kê tên lớp và kiểu hình học |
| `scripts/prepare_dronevehicle.py` | Cắt viền, chuẩn hoá nhãn, xuất YOLO-OBB, sinh meta |
| `scripts/prepare_vedai.py` | Chuyển annotation VEDAI, tách fold |
| `scripts/compute_illum_features.py` | 12 đặc trưng chiếu sáng cho mỗi ảnh |
| `scripts/label_illumination.py` | Phân tầng `illum_proxy` + contact sheet |
| `scripts/check_alignment.py` | Kiểm chứng ghép cặp (NMI) + đo lệch đồng đăng ký |
| `scripts/draw_gt.py` | Vẽ box GT lên cả hai modality để soi bằng mắt |
| `scripts/make_yolo_views.py` | Dựng view hardlink theo quy ước Ultralytics |
| `scripts/smoke_test_data.py` | Xác nhận Ultralytics nạp được cả 4 view |
| `scripts/sheet_by_feature.py` | Contact sheet theo giá trị đặc trưng (dò ngưỡng) |

Chạy lại từ đầu:

```bash
python scripts/_extract.py                    # giải nén, bỏ qua file hỏng
python scripts/prepare_dronevehicle.py
python scripts/prepare_vedai.py
python scripts/compute_illum_features.py
python scripts/label_illumination.py
python scripts/make_yolo_views.py
python scripts/smoke_test_data.py             # phải PASS
```

---

## 6. Kiểm chứng đã thực hiện

| Kiểm chứng | Kết quả |
|---|---|
| sha256 của archive nguồn | ✅ khớp cả 3 |
| CRC toàn bộ entry | ⚠️ 39/113.756 hỏng (upstream), đã loại |
| Kiểm kê lớp vs README gốc | ✅ lệch < 0,01% |
| Viền trắng 100px | ✅ cả 4 cạnh đúng giá trị 255, core đúng 640×512 |
| IR 3 kênh giống hệt nhau | ✅ max\|B−R\| = 0 |
| Ghép cặp (NMI) | ✅ 90,7% cặp khớp ăn đứt cặp ngẫu nhiên |
| Đồng đăng ký | ⚠️ trung vị 1,8 px, 20,9% cặp lệch >5 px |
| Vẽ GT lên cả 2 modality | ✅ box khớp sát phương tiện, hướng đúng — `results/gt_check/` |
| Ultralytics nạp dữ liệu | ✅ PASS cả 4 view, 0 lỗi, toạ độ trong [0,1] |

### Còn phải làm trước khi train (plan mục 7.3)

- [ ] `test_augment_sync` — augmentation hình học đồng bộ hai modality
- [ ] `test_metrics_known` — mAP(GT, GT) = 1,0
- [ ] `test_config_capacity` — params C1 == F2a (sai số < 1%)
- [ ] `test_wbf_identity`
- [ ] Train S2 (IR-only) đến hội tụ, đối chiếu mAP50 với literature (~83%)
- [ ] Overfit có chủ đích 50 mẫu
