# Sensor Fusion RGB-IR cho phát hiện đối tượng từ UAV

Kiểm chứng **có kiểm soát** giả thuyết: kết hợp RGB + hồng ngoại có thật sự tốt hơn từng cảm biến riêng lẻ không — sau khi đã tách bạch phần đóng góp của *thông tin bổ sung* khỏi phần đóng góp của *mô hình nhiều tham số hơn*.

Kế hoạch đầy đủ: [plan.md](plan.md) · Báo cáo dữ liệu: [DATA_REPORT.md](DATA_REPORT.md)

---

## 1. Ý tưởng cốt lõi

Phần lớn công trình fusion so sánh model hai backbone với model một backbone rồi quy toàn bộ chênh lệch cho fusion. Đó là một confounder: model hai backbone đơn giản là **lớn hơn**.

Đề tài này thêm **nhóm đối chứng capacity**: đúng kiến trúc hai luồng đó, nhưng nạp **cùng một modality vào cả hai luồng**.

| ID | Đầu vào | Kiến trúc | Vai trò |
|---|---|---|---|
| S1 | RGB | 1 luồng | baseline |
| S2 | IR | 1 luồng | baseline |
| **C1** | RGB → cả 2 luồng | 2 luồng | **đối chứng capacity** |
| **C2** | IR → cả 2 luồng | 2 luồng | **đối chứng capacity** |
| F1 | RGB + IR | early, 4 kênh | fusion |
| F2a | RGB + IR | mid, concat+1×1 | fusion |
| F2b | RGB + IR | mid, attention gate | fusion |
| F3 | RGB + IR | late, WBF từ S1+S2 | fusion |

**So sánh trung thực là `F2a vs C2`, không phải `F2a vs S1`.**

C1, C2 và F2a dùng **chung một file kiến trúc** — chỉ khác trường `modalities`. Số tham số bằng nhau tuyệt đối theo cấu trúc, không phải "bằng nhau trong sai số". `tests/test_models.py::test_capacity_control_exact_match` canh bất biến này.

---

## 2. Cách hoạt động

### Xếp chồng kênh thay vì dataloader hai nhánh

Hai modality được xếp thành **một mảng HWC nhiều kênh** ngay tại `load_image`. Toàn bộ pipeline Ultralytics dùng lại nguyên vẹn: Mosaic, LetterBox, RandomPerspective, OBB loss, validator.

Lợi ích quyết định: **augmentation hình học tự động đồng nhất trên mọi kênh**. Rủi ro R4 của plan ("bug phổ biến nhất trong pipeline paired") biến mất về mặt cấu trúc chứ không nhờ ta cẩn thận.

Model tách kênh trở lại thành hai nhánh bằng lớp `Index` ở đầu đồ thị.

```
ảnh 6 kênh ──┬── Index[0:3] ── backbone A ──┐
             │                              ├── Concat → Conv1×1 → neck → OBB head
             └── Index[3:6] ── backbone B ──┘
```

### Ba bản vá vào Ultralytics

Không fork. Chỉ ba điểm, đều idempotent và có test phủ (`src/patches.py`):

1. **`Index` hỗ trợ cắt kênh** — tận dụng nhánh `c2 = args[0]` có sẵn của `parse_model`; giữ nguyên hành vi cũ.
2. **`build_yolo_dataset`** trả về `PairedYOLODataset` khi `data["modalities"]` tồn tại — đi qua `data` nên train và val không thể lệch cấu hình.
3. **Sửa lỗi lấp viền của `RandomPerspective`** — xem mục 5.

---

## 3. Bắt đầu

```bash
# --- dữ liệu (một lần) ---------------------------------------------------
python scripts/prepare_dronevehicle.py      # 28.400 cặp, YOLO-OBB
python scripts/prepare_vedai.py             # 1.246 cặp, 10 fold
python scripts/compute_illum_features.py
python scripts/label_illumination.py        # phân tầng RQ3
python scripts/make_yolo_views.py           # view hardlink cho Ultralytics
python scripts/smoke_test_data.py           # phải PASS

# --- cổng G0: test phải xanh trước mọi training run ----------------------
python -m pytest -q

# --- kiểm thử tích hợp rẻ trước khi tốn GPU ------------------------------
python scripts/smoke_pipeline.py --epochs 3

# --- ma trận thí nghiệm --------------------------------------------------
python scripts/run_matrix.py --dataset vedai --folds 01 03 05
python scripts/run_matrix.py --dataset dronevehicle --seeds 0 1

# --- đánh giá ------------------------------------------------------------
python scripts/evaluate_run.py --run runs/dronevehicle/F2a_seed0 --split test \
    --illum data/dronevehicle/meta/test_illum.csv

python scripts/run_late_fusion.py --s1 runs/dronevehicle/S1_seed0 \
    --s2 runs/dronevehicle/S2_seed0 --out runs/dronevehicle/F3_seed0

python scripts/run_rq4_ablation.py --runs runs/dronevehicle/F1_seed0 \
    runs/dronevehicle/F2a_seed0 --deltas 0 2 5 10 20

python scripts/make_tables.py --project runs/dronevehicle --split test \
    --illum data/dronevehicle/meta/test_illum.csv
```

`run_matrix.py` bỏ qua run đã có `result.json`, nên bị ngắt giữa chừng chỉ cần chạy lại đúng lệnh cũ.

---

## 4. Cấu trúc

```
configs/
├── base.yaml                  # siêu tham số dùng chung — mọi khác biệt phải nằm ở experiments/
├── datasets/                  # 4 view: {dronevehicle,vedai} × {rgb,ir}
└── experiments/               # 8 cấu hình, sinh bởi scripts/gen_experiment_configs.py
src/
├── patches.py                 # 3 bản vá Ultralytics
├── train.py                   # trainer + runner cho một ô ma trận
├── data/paired_dataset.py     # xếp chồng modality, dịch IR cho RQ4
├── models/
│   ├── two_stream.py          # sinh kiến trúc two-stream từ YAML gốc
│   ├── fusion_ops.py          # GatedFusion (F2b)
│   └── build.py               # dựng model + nạp pretrained công bằng
├── fusion/wbf_obb.py          # late fusion cho box xoay
├── eval/
│   ├── metrics.py             # mAP OBB, IoU đa giác chính xác
│   ├── stratify.py            # phân tầng chiếu sáng / kích thước / lớp
│   ├── bootstrap.py           # CI ghép cặp theo ảnh, công thức có trọng số
│   └── harness.py             # suy luận + kết xuất + đánh giá phân tầng
└── utils/seed.py              # seed + provenance
tests/                         # 44 test
scripts/                       # chuẩn bị dữ liệu, chạy, đánh giá, tổng hợp
```

Mỗi run ghi `provenance.json` (config + hash + phiên bản torch/ultralytics/GPU), `data.yaml`, `result.json`, `preds_*.npz`, `metrics_*.json`. Không có số liệu nào phải đọc từ stdout.

---

## 5. Ba lỗi/khác biệt đã phát hiện khi triển khai

Đây là phần đáng đọc nhất — tất cả đều là loại sai âm thầm, không báo lỗi.

### 5.1 `RandomPerspective` lấp viền khác nhau giữa các nhóm kênh — **lỗi thật, đã sửa**

`cv2.warpAffine(img, M, borderValue=(114,114,114))` — `borderValue` của OpenCV là `Scalar`, tối đa **4 thành phần**. Với ảnh 6 kênh, bốn kênh đầu được lấp 114 còn **kênh 5-6 bị lấp 0**.

Phép biến đổi hình học vẫn đồng nhất (đúng), nhưng vùng viền lộ ra sau khi xoay/co giãn mang giá trị khác nhau giữa hai modality. Hậu quả: model học artefact "viền IR bằng 0", và **đối chứng C1 không còn đối xứng thật sự** — hỏng đúng phần làm nên giá trị của đề tài.

Cách sửa: warp từng nhóm ≤3 kênh bằng cùng ma trận và cùng giá trị viền (`src/patches.py::apply_warp_patch`). Ảnh 3 kênh đi đúng đường code cũ.

Bắt được bởi `tests/test_data.py::test_augment_sync`: nạp cùng một modality vào cả hai luồng, hai nửa phải bằng nhau từng pixel sau augment.

### 5.2 mAP của Ultralytics lạc quan có hệ thống trên OBB

`OBBValidator` khớp box bằng **ProbIoU** — một xấp xỉ khả vi dùng cho *hàm mất mát*. Đo trên cùng bộ dự đoán (`scripts/diag_metrics.py`):

| | mAP50 | mAP50-95 |
|---|---|---|
| IoU đa giác chính xác + khớp theo confidence (COCO) | 0,1585 | 0,0576 |
| IoU chính xác + khớp kiểu Ultralytics | 0,1552 | 0,0571 |
| ProbIoU + khớp theo confidence | 0,1761 | 0,1034 |
| ProbIoU + khớp kiểu Ultralytics | 0,1698 | 0,1014 |
| *Ultralytics báo cáo* | *0,176* | *0,103* |

Thứ tự khớp gần như không ảnh hưởng. **ProbIoU cao hơn IoU thật trung bình +0,13, tối đa +0,26**, và lệch mạnh nhất ở ngưỡng IoU cao — nên mAP50-95 bị thổi phồng gần gấp đôi.

Đề tài **báo cáo theo IoU đa giác chính xác** để so được với baseline công bố (mmrotate/DOTA devkit). Hệ quả phải ghi rõ: số của `model.val()` không so trực tiếp được với bảng kết quả, và sanity check "đối chiếu S2 với ~83% của literature" chỉ hợp lệ khi dùng harness này.

### 5.3 `compute_ap` của Ultralytics dùng `trapz`, không phải chuẩn COCO

Với `trapz`, một dự đoán hoàn hảo duy nhất cho AP = 0,995 chứ không phải 1,0. pycocotools lấy **trung bình 101 giá trị precision đã nội suy** → AP(GT, GT) = 1,0 chính xác. Harness dùng công thức pycocotools; `tests/test_metrics.py::test_map_perfect_prediction` canh điều này.

---

## 6. Vài quyết định thiết kế đáng chú ý

**Pretrained cho mọi cấu hình.** Nếu S1/S2 nạp được `yolo11s-obb.pt` mà two-stream train from scratch thì chênh lệch sẽ bị quy nhầm cho fusion. `src/models/build.py` ánh xạ chỉ số lớp để **cả hai nhánh nhận cùng bộ trọng số**, và nong stem 3→4 kênh cho F1. Độ phủ: 98,9% (single/early), 97,0% (two-stream).

**Bootstrap có trọng số.** Cách ngây thơ tốn ~25 phút cho một phép so sánh trên 8.980 ảnh. Nhận xét rằng ảnh được chọn `c_i` lần chỉ làm mọi phát hiện của nó xuất hiện `c_i` lần với cùng confidence, nên đường cong P-R chỉ cần một `cumsum` có trọng số sau khi sắp xếp toàn cục **một lần**. Kết quả giống hệt, nhanh hơn hai bậc — `tests/test_metrics.py::test_bootstrap_weighted_matches_naive` kiểm chứng sự tương đương.

**`batch=8` cho mọi cấu hình.** Two-stream bs16@640 cần ~7,7GB, không vừa RTX 3060 Ti 8GB. Nếu để single-stream chạy bs16 còn two-stream bs8 thì batch khác nhau đổi cả effective LR lẫn thống kê BatchNorm — và chênh lệch đó bị quy nhầm cho fusion. Dùng `nbs=32` để gradient accumulate về cùng nominal batch.

**Tắt HSV cho mọi cấu hình.** `RandomHSV` và `Albumentations` của Ultralytics tự bỏ qua ảnh ≠3 kênh. Chúng không lỗi, nhưng nghĩa là nhánh 3 kênh được augment màu còn nhánh 4/6 kênh thì không → mất công bằng.

**Không chuẩn hoá mean/std (lệch có chủ đích so với plan 6.4).** Mối lo của plan là *dùng chung* thống kê giữa hai modality; với phép chia 255 thì không có thống kê nào được chia sẻ, và BatchNorm ở stem tự học riêng cho từng kênh. Thêm nữa, pretrained YOLO kỳ vọng đầu vào trong [0,1]. Tuỳ chọn `normalize: per_modality` vẫn có sẵn để kiểm chứng lại bằng thực nghiệm.

---

## 7. Trạng thái

`pytest`: **44/44 xanh**. Pipeline đã chạy thông end-to-end trên VEDAI qua `scripts/smoke_pipeline.py`: train ma trận → đánh giá phân tầng → late fusion WBF → ablation RQ4 → bảng kết quả có khoảng tin cậy.

Ví dụ đầu ra (3 epoch, **chỉ để kiểm chứng lắp ghép — không phải kết quả nghiên cứu**):

```
| so sanh   | delta mAP50 | CI95              | p     | y nghia |
| F2a vs C2 | +5.09       | [+0.56, +10.94]   | 0.027 | co      |
| C2  vs S2 | -7.33       | [-11.58, -3.36]   | 0.007 | co      |
| F2a vs S2 | -2.23       | [-6.97,  +3.23]   | 0.493 | khong   |

RQ4: F2a giữ 76,0% mAP50 khi dịch IR 5px
```

Còn lại là chạy ma trận thật: `run_matrix.py --dataset vedai` rồi `--dataset dronevehicle`.

---

## 8. Môi trường

RTX 3060 Ti 8GB · i5-13600K · 32GB RAM · Python 3.14 · torch 2.11+cu128 · ultralytics 8.4.61

⚠️ MMRotate **không dùng được** trên môi trường này (mmcv pin torch <2.1, không có wheel cho Python 3.14). Phương án dự phòng của plan cho R1 là đường 6 kênh ở đây, không phải MMRotate.
