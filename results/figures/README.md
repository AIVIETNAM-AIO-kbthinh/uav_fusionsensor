# Biểu đồ kết quả

Sinh toàn bộ bằng:

```bash
.venv/Scripts/python.exe scripts/make_figures.py                    # tất cả
.venv/Scripts/python.exe scripts/make_figures.py --only 1 2 3       # một vài hình
.venv/Scripts/python.exe scripts/make_figures.py --reuse-effects    # bỏ qua bootstrap (~15 phút)
```

Mọi con số lấy từ `runs/*/metrics_{split}.json`, `runs/*/results.csv`,
`runs/*/result.json` và `runs/*/preds_{split}.npz` — không có giá trị nào gõ tay.
Bảng số liệu tương ứng: [`results/tables/figure_data.md`](../tables/figure_data.md).

| Hình | Tệp | Trả lời câu hỏi gì | Nguồn số |
|---|---|---|---|
| 1 | `fig01_map_comparison.png` | Cấu hình nào tốt hơn, tổng thể? | `metrics_*.json` |
| 2 | `fig02_per_class_f1.png` | Lợi ích tập trung ở lớp nào? | `preds_*.npz` |
| 3 | `fig03_per_class_ap50.png` | Như hình 2 nhưng không phụ thuộc ngưỡng | `metrics_*.json` |
| 4 | `fig04_training_curves_*.png` | Có hội tụ / overfit không? Cấu hình nào học nhanh hơn? | `results.csv` |
| 5 | `fig05_illumination_strata.png` | RQ3 — lợi ích có lớn hơn khi thiếu sáng không? | `metrics_*.json` |
| 6 | `fig06_effect_sizes.png` | **RQ1** — chênh lệch có vượt nhiễu không? | `preds_*.npz` + bootstrap |
| 7 | `fig07_f1_confidence_curve.png` | Ngưỡng vận hành nào tối ưu? | `preds_*.npz` |
| 8 | `fig08_capacity_vs_map.png` | Bao nhiêu phần là do cảm biến, bao nhiêu do số tham số? | `metrics_*.json` + `result.json` |

## Hai lưu ý bắt buộc khi đưa vào báo cáo

1. **Hình 4 không so trực tiếp được với các hình còn lại.** `results.csv` là số của
   validator Ultralytics (khớp box bằng ProbIoU — lạc quan có hệ thống, xem
   plan.md mục 0.4). Các hình khác dùng harness của đề tài (IoU đa giác chính
   xác, AP 101 điểm). Hình 4 chỉ để đọc *động lực học khi train*.

2. **Con số để kết luận RQ1 là `F2a − C2`** (hình 6, hình 8), không phải
   `F2a − S1/S2`. C2 có cùng số tham số tuyệt đối với F2a nên hiệu số giữa chúng
   đã loại bỏ confounder dung lượng mô hình.

## Bảng màu

Theo bảng màu định danh đã kiểm CVD (`#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`,
`#e87ba4`, `#008300`), gán cố định theo cấu hình và không xoay vòng. Hai cấu hình
đối chứng (C1, C2) thêm **nét đứt** làm mã hoá phụ vì trên DroneVehicle đường của
C1 gần như trùng khít S1 — nếu chỉ phân biệt bằng màu thì một đường bị che hoàn
toàn.

## Tệp phụ trợ

- `_prf_cache.json` — cache P/R/F1 theo lớp (khớp IoU đa giác lại từ `preds_*.npz`,
  ~3 phút). Tự làm mới khi `preds_*.npz` đổi; ép làm mới bằng `--refresh-cache`.
- `_effect_sizes.json` — kết quả bootstrap của hình 6, dùng lại được qua
  `--reuse-effects`.
