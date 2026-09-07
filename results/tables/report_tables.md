# Bang bieu ket qua

Sinh tu dong bang `python scripts/make_report_tables.py`. Moi con so doc tu artifact trong `runs/` va `results/figures/` — khong gia tri nao go tay.

Don vi mac dinh: **%**. Chi tieu chinh: **mAP50 (OBB)**, do bang harness cua de tai (IoU da giac chinh xac, AP 101 diem chuan COCO).

## Bang 1 — Ma tran thi nghiem va trang thai

| ID | Dau vao | Kien truc | Vai tro | Tham so | n run DV | n run VEDAI | Trang thai |
|---|---|---|---|---|---|---|---|
| S1 | RGB | 1 luong | baseline | 9.715.906 | 2 | 3 | da chay |
| S2 | IR | 1 luong | baseline | 9.715.906 | 2 | 3 | da chay |
| C1 | RGB vao ca 2 luong | 2 luong | **doi chung capacity** | 15.946.370 | 2 | 3 | da chay |
| C2 | IR vao ca 2 luong | 2 luong | **doi chung capacity** | 15.946.370 | 2 | 3 | da chay |
| C1b | RGB vao ca 2 luong | 2 luong + attn | **doi chung capacity (attn)** | — | — | — | **chua chay** |
| C2b | IR vao ca 2 luong | 2 luong + attn | **doi chung capacity (attn)** | 16.145.154 | 1 | 3 | da chay |
| F1 | RGB + IR | 1 luong, 4 kenh | fusion som | 9.716.194 | 2 | 3 | da chay |
| F2a | RGB + IR | 2 luong | fusion giua (concat+1x1) | 15.946.370 | 2 | 3 | da chay |
| F2b | RGB + IR | 2 luong + attn | fusion giua (attention) | 16.145.154 | 1 | 3 | da chay |
| F3 | RGB + IR | 2 x 1 luong | fusion muon (WBF) | — | 2 | 3 | da chay |

> Hai nhom co **so tham so bang nhau tuyet doi** trong noi bo nhom — cung file kien truc, chi khac truong `modalities`: **{C1, C2, F2a}** va **{C1b, C2b, F2b}**. Day la dieu kien de `F2a - C2` va `F2b - C2b` doc duoc la dong gop cua *thong tin bo sung*, khong phai cua *dung luong mo hinh*.

> Cot *Tham so* lay tu run dau tien tim duoc nen phu thuoc so lop `nc` cua dataset do (DroneVehicle 5 lop, VEDAI 8 lop) — bat bien duoc canh la *bang nhau trong cung mot nhom tren cung mot dataset*, xem `scripts/gen_experiment_configs.py`.

> Cong attn cua F2b them **198.784 tham so** so voi F2a (khong phu thuoc `nc`), nen **`F2b - C2` khong phai so sanh co kiem soat** — doi chung dung cua F2b la C2b.

## Bang 2 — DroneVehicle (test): ket qua tong the

mean ± std qua 2 seed.

| cau hinh | vai tro | n seed | mAP50 | mAP50-95 |
|---|---|---|---|---|
| S1 | baseline | 2 | 69.68 ± 0.05 | 38.18 ± 0.02 |
| S2 | baseline | 2 | 81.06 ± 0.17 | 54.19 ± 0.11 |
| C1 | **doi chung capacity** | 2 | 69.29 ± 0.50 | 38.07 ± 0.33 |
| C2 | **doi chung capacity** | 2 | 81.57 ± 0.08 | 54.80 ± 0.04 |
| C2b | **doi chung capacity (attn)** | 1 | 81.53 ± 0.00 | 54.76 ± 0.00 |
| F1 | fusion som | 2 | 82.83 ± 0.03 | 56.26 ± 0.04 |
| F2a | fusion giua (concat+1x1) | 2 | 82.67 ± 0.07 | 56.34 ± 0.11 |
| F2b | fusion giua (attention) | 1 | 83.14 ± 0.00 | 56.63 ± 0.00 |
| F3 | fusion muon (WBF) | 2 | 79.32 ± 0.21 | 49.94 ± 0.25 |

## Bang 3 — VEDAI (val): ket qua tong the

mean ± std qua 3 fold.

| cau hinh | vai tro | n fold | mAP50 | mAP50-95 |
|---|---|---|---|---|
| S1 | baseline | 3 | 77.20 ± 2.08 | 42.27 ± 1.88 |
| S2 | baseline | 3 | 72.10 ± 2.97 | 39.25 ± 2.90 |
| C1 | **doi chung capacity** | 3 | 74.92 ± 1.89 | 41.20 ± 1.73 |
| C2 | **doi chung capacity** | 3 | 71.08 ± 4.33 | 39.30 ± 3.03 |
| C2b | **doi chung capacity (attn)** | 3 | 71.53 ± 3.96 | 38.84 ± 3.17 |
| F1 | fusion som | 3 | 76.92 ± 3.70 | 42.28 ± 2.33 |
| F2a | fusion giua (concat+1x1) | 3 | 77.54 ± 2.36 | 41.66 ± 1.40 |
| F2b | fusion giua (attention) | 3 | 77.69 ± 2.33 | 41.63 ± 2.75 |
| F3 | fusion muon (WBF) | 3 | 77.52 ± 4.68 | 41.85 ± 3.62 |

## Bang 4 — Kiem dinh hieu so (bootstrap ghep cap theo anh, 2.000 lan lap, CI 95%)

Chi tieu: mAP50. Ghep cap theo anh nen nhieu do do kho de anh gay ra bi triet tieu.

| dataset | so sanh | y nghia | n anh | delta mAP50 | CI 95% | p | co y nghia |
|---|---|---|---|---|---|---|---|
| DroneVehicle | F2a vs C2 | RQ1 trung thực — fusion vs đối chứng capacity | 8.980 | +1.11 | [+0.75, +1.52] | 0.000 | **co** |
| DroneVehicle | F2a vs S2 | RQ1 ngây thơ — fusion vs modality đơn | 8.980 | +1.71 | [+1.34, +2.12] | 0.000 | **co** |
| DroneVehicle | F2a vs F1 | RQ2 — fusion giữa vs fusion sớm | 8.980 | -0.20 | [-0.53, +0.17] | 0.282 | khong |
| DroneVehicle | C2 vs S2 | chẩn đoán — lợi ích thuần từ capacity (IR) | 8.980 | +0.60 | [+0.25, +0.94] | 0.001 | **co** |
| DroneVehicle | C1 vs S1 | chẩn đoán — lợi ích thuần từ capacity (RGB) | 8.980 | +0.06 | [-0.32, +0.47] | 0.727 | khong |
| DroneVehicle | F2a vs F3 | RQ2 — fusion giữa vs fusion muộn | 8.980 | +3.50 | [+3.05, +4.02] | 0.000 | **co** |
| DroneVehicle | F2b vs F1 | RQ2 — two-stream + cổng vs early fusion 4 kênh | 8.980 | +0.34 | [-0.02, +0.64] | 0.062 | khong |
| DroneVehicle | F2b vs C2b | RQ1 trung thực (attn) — F2b vs đối chứng capacity của chính nó | 8.980 | +1.61 | [+1.23, +1.99] | 0.000 | **co** |
| DroneVehicle | F2b vs F2a | RQ2 — attention gate vs concat+1×1 | 8.980 | +0.54 | [+0.23, +0.76] | 0.000 | **co** |
| DroneVehicle | C2b vs C2 | chẩn đoán — cổng attn khi KHÔNG có tín hiệu bổ sung | 8.980 | +0.04 | [-0.25, +0.33] | 0.823 | khong |
| VEDAI | F2a vs C2 | RQ1 trung thực — fusion vs đối chứng capacity | 363 | +6.49 | [+3.24, +9.88] | 0.000 | **co** |
| VEDAI | F2a vs S2 | RQ1 ngây thơ — fusion vs modality đơn | 363 | +5.56 | [+2.32, +8.79] | 0.001 | **co** |
| VEDAI | F2a vs F1 | RQ2 — fusion giữa vs fusion sớm | 363 | +0.63 | [-2.42, +3.89] | 0.660 | khong |
| VEDAI | C2 vs S2 | chẩn đoán — lợi ích thuần từ capacity (IR) | 363 | -0.93 | [-3.85, +1.53] | 0.454 | khong |
| VEDAI | C1 vs S1 | chẩn đoán — lợi ích thuần từ capacity (RGB) | 363 | -2.60 | [-5.32, +0.50] | 0.097 | khong |
| VEDAI | F2a vs F3 | RQ2 — fusion giữa vs fusion muộn | 363 | +0.22 | [-2.80, +3.27] | 0.874 | khong |
| VEDAI | F2b vs F1 | RQ2 — two-stream + cổng vs early fusion 4 kênh | 363 | +0.39 | [-2.52, +3.43] | 0.742 | khong |
| VEDAI | F2b vs C2b | RQ1 trung thực (attn) — F2b vs đối chứng capacity của chính nó | 363 | +5.79 | [+2.54, +9.22] | 0.001 | **co** |
| VEDAI | F2b vs F2a | RQ2 — attention gate vs concat+1×1 | 363 | -0.24 | [-2.65, +2.19] | 0.840 | khong |
| VEDAI | C2b vs C2 | chẩn đoán — cổng attn khi KHÔNG có tín hiệu bổ sung | 363 | +0.46 | [-2.45, +3.38] | 0.734 | khong |

> Bootstrap lay mau lai theo **anh**, khong theo seed. DroneVehicle dung seed 0; VEDAI gop ca 3 fold (cac tap val roi nhau) de du co mau — do la ly do n = 363.

## Bang 5 — Phan ra: bao nhieu la do cam bien, bao nhieu la do tham so?

| dataset | hieu so | dien giai | delta mAP50 | ty trong |
|---|---|---|---|---|
| DroneVehicle | F2a − S2 | tong chenh lech (cach bao cao pho bien) | +1.71 | 100% |
| DroneVehicle | C2 − S2 | phan do **tang so tham so** (cung modality, 2 luong) | +0.60 | 35% |
| DroneVehicle | **F2a − C2** | phan do **thong tin bo sung tu RGB** — con so de claim | **+1.11** | **65%** |
| VEDAI | F2a − S2 | tong chenh lech (cach bao cao pho bien) | +5.56 | 100% |
| VEDAI | C2 − S2 | phan do **tang so tham so** (cung modality, 2 luong) | -0.93 | -17% |
| VEDAI | **F2a − C2** | phan do **thong tin bo sung tu RGB** — con so de claim | **+6.49** | **117%** |

## Bang 6 — RQ3 (DroneVehicle): ket qua theo tang chieu sang

mAP50, mean ± std qua 2 seed.

| cau hinh | tat ca | thieu sang | trung binh | sang |
|---|---|---|---|---|
| S1 | 69.68 ± 0.05 | 63.43 ± 0.23 | 71.17 ± 0.16 | 78.21 ± 0.22 |
| S2 | 81.06 ± 0.17 | 82.60 ± 0.19 | 80.30 ± 0.30 | 78.18 ± 0.09 |
| C1 | 69.29 ± 0.50 | 62.83 ± 0.08 | 70.51 ± 0.83 | 78.33 ± 0.67 |
| C2 | 81.57 ± 0.08 | 83.15 ± 0.04 | 80.57 ± 0.30 | 79.01 ± 0.05 |
| C2b | 81.53 ± 0.00 | 83.25 ± 0.00 | 80.23 ± 0.00 | 78.98 ± 0.00 |
| F1 | 82.83 ± 0.03 | 83.57 ± 0.22 | 81.66 ± 0.04 | 82.14 ± 0.34 |
| F2a | 82.67 ± 0.07 | 83.34 ± 0.13 | 80.90 ± 0.02 | 82.58 ± 0.02 |
| F2b | 83.14 ± 0.00 | 83.66 ± 0.00 | 81.11 ± 0.00 | 83.38 ± 0.00 |
| F3 | 79.32 ± 0.21 | 79.25 ± 0.40 | 78.85 ± 0.00 | 79.61 ± 0.21 |

So anh moi tang: thieu sang 3.016 / trung binh 3.180 / sang 2.784 — tong 8.980.

### Bang 6b — Chenh lech theo tang (diem uoc luong, trung binh 2 seed)

| tang | F2a | C2 | S2 | F2a − C2 | F2a − S2 |
|---|---|---|---|---|---|
| tat ca | 82.67 | 81.57 | 81.06 | +1.10 | +1.61 |
| thieu sang | 83.34 | 83.15 | 82.60 | +0.19 | +0.74 |
| trung binh | 80.90 | 80.57 | 80.30 | +0.33 | +0.60 |
| sang | 82.58 | 79.01 | 78.18 | +3.58 | +4.40 |

> Day la **hieu cua trung binh 2 seed**, chua co CI. CI theo tang nam trong `runs/dronevehicle/tables_test.md` (sinh boi `scripts/make_tables.py --illum`).

## Bang 7 — DroneVehicle: AP50 theo tung lop

mean qua cac seed.

| cau hinh | car | truck | bus | van | freight_car |
|---|---|---|---|---|---|
| S1 | 87.3 | 65.3 | 87.8 | 55.4 | 52.7 |
| S2 | 97.5 | 79.5 | 93.4 | 65.7 | 69.3 |
| C1 | 87.4 | 64.5 | 87.9 | 54.5 | 52.1 |
| C2 | 97.5 | 79.8 | 93.6 | 66.4 | 70.5 |
| C2b | 97.5 | 80.4 | 93.5 | 66.0 | 70.2 |
| F1 | 97.7 | 82.9 | 94.6 | 67.1 | 71.9 |
| F2a | 97.7 | 82.7 | 94.5 | 68.7 | 69.7 |
| F2b | 97.8 | 82.9 | 94.9 | 69.3 | 70.9 |
| F3 | 95.1 | 77.7 | 89.8 | 65.4 | 68.5 |
| *so box GT* | *135.916* | *8.565* | *4.371* | *4.274* | *5.001* |

## Bang 8 — DroneVehicle: Precision / Recall / F1 theo lop tai nguong toi uu

| cau hinh | car | truck | bus | van | freight_car | macro-F1 |
|---|---|---|---|---|---|---|
| S1 · P | 86.1 | 71.3 | 84.6 | 62.7 | 57.9 | — |
| S1 · R | 84.9 | 61.9 | 88.0 | 51.6 | 50.2 | — |
| S1 · F1-score | 85.5 | 66.2 | 86.2 | 56.6 | 53.7 | **69.7** |
| S2 · P | 91.7 | 74.2 | 88.4 | 67.2 | 65.6 | — |
| S2 · R | 96.3 | 75.2 | 90.4 | 59.7 | 68.3 | — |
| S2 · F1-score | 94.0 | 74.7 | 89.4 | 63.3 | 66.9 | **77.6** |
| C1 · P | 86.5 | 71.0 | 85.3 | 63.7 | 56.5 | — |
| C1 · R | 85.0 | 62.2 | 88.0 | 53.3 | 52.9 | — |
| C1 · F1-score | 85.7 | 66.3 | 86.6 | 58.0 | 54.6 | **70.3** |
| C2 · P | 92.2 | 76.6 | 89.3 | 69.1 | 68.8 | — |
| C2 · R | 96.0 | 75.3 | 90.9 | 60.7 | 67.3 | — |
| C2 · F1-score | 94.1 | 75.9 | 90.1 | 64.6 | 68.0 | **78.5** |
| F1 · P | 92.4 | 78.2 | 89.5 | 70.6 | 68.7 | — |
| F1 · R | 96.3 | 76.5 | 93.1 | 58.8 | 69.0 | — |
| F1 · F1-score | 94.3 | 77.3 | 91.3 | 64.2 | 68.8 | **79.2** |
| F2a · P | 92.5 | 78.9 | 90.1 | 70.4 | 66.8 | — |
| F2a · R | 96.4 | 77.3 | 93.1 | 62.8 | 70.1 | — |
| F2a · F1-score | 94.4 | 78.1 | 91.6 | 66.4 | 68.4 | **79.8** |

> Nguong confidence toi uu hoa macro-F1: **0.375**. Lay tu run dau tien cua moi cau hinh (seed0).

## Bang 9 — VEDAI: AP50 theo tung lop

mean qua cac fold.

| cau hinh | car | pickup | camping | truck | other | tractor | boat | van |
|---|---|---|---|---|---|---|---|---|
| S1 | 89.1 | 84.2 | 80.8 | 74.7 | 55.5 | 72.8 | 78.6 | 81.8 |
| S2 | 86.2 | 80.6 | 78.6 | 74.0 | 51.0 | 57.1 | 72.5 | 76.9 |
| C1 | 87.3 | 79.8 | 74.2 | 73.3 | 59.9 | 72.3 | 78.5 | 74.0 |
| C2 | 84.9 | 78.6 | 78.4 | 70.5 | 40.3 | 65.1 | 71.7 | 79.1 |
| C2b | 85.4 | 79.3 | 78.5 | 68.6 | 45.4 | 65.3 | 75.0 | 74.8 |
| F1 | 88.5 | 83.7 | 81.8 | 74.9 | 60.5 | 72.1 | 77.4 | 76.4 |
| F2a | 87.3 | 81.7 | 77.0 | 78.5 | 61.8 | 80.3 | 82.9 | 70.9 |
| F2b | 86.9 | 80.7 | 79.2 | 77.4 | 60.9 | 80.0 | 80.5 | 75.8 |
| F3 | 88.9 | 80.6 | 81.9 | 76.1 | 60.0 | 72.6 | 80.3 | 79.8 |
| *so box GT* | *134* | *95* | *39* | *30* | *22* | *19* | *17* | *10* |

> VEDAI: so box GT la cua **mot fold** (fold01); ba fold co tap val roi nhau.

## Bang 10 — Chi phi tinh toan va cau hinh train

Gio train = cot `time` cua epoch cuoi trong `results.csv`, cong don qua cac run (khong dung `hours` trong `matrix_summary.json` — truong do chi ghi lan goi `run_matrix.py` gan nhat, khong phai tong cong). Cot *run hoan tat* dem tu chinh cac thu muc run, khong tu truong `done` vi truong do co cung khiem khuyet. Cot *run loi* thi van la cua lan goi gan nhat.

| dataset | run hoan tat | run loi | epoch | tong gio train | gio/run (TB) |
|---|---|---|---|---|---|
| DroneVehicle | 14 | 0 | 60 | 69.6 | 5.0 |
| VEDAI | 24 | 0 | 60 | 8.1 | 0.3 |

### Bang 10b — Gio train theo cau hinh (trung binh moi run)

| cau hinh | kien truc | DroneVehicle | VEDAI |
|---|---|---|---|
| S1 | 1 luong | 4.01 | 0.26 |
| S2 | 1 luong | 4.01 | 0.26 |
| C1 | 2 luong | 5.90 | 0.38 |
| C2 | 2 luong | 5.99 | 0.38 |
| C2b | 2 luong + attn | 6.12 | 0.39 |
| F1 | 1 luong, 4 kenh | 4.06 | 0.26 |
| F2a | 2 luong | 4.71 | 0.38 |
| F2b | 2 luong + attn | 6.14 | 0.39 |

Sieu tham so dung chung cho **moi** cau hinh (dieu kien so sanh cong bang):

`epochs=60` · `batch=8` · `nbs=32` · `imgsz=640` · `optimizer=SGD` · `lr0=0.01` · `cos_lr=True` · `mosaic=1.0` · `close_mosaic=10` · `hsv_h/s/v=0` · `augmentations=[]` · `amp=True` · `deterministic=True`

## Bang 11 — Canh bao doc so: validator Ultralytics vs harness cua de tai

Cung mot bo trong so, hai cach do cho hai con so khac han.

| cau hinh | mAP50 Ultralytics (val) | mAP50 harness (test) | mAP50-95 Ultralytics (val) | mAP50-95 harness (test) | chenh 50-95 |
|---|---|---|---|---|---|
| S1 | 75.61 | 69.68 | 55.07 | 38.18 | +16.89 |
| S2 | 84.70 | 81.06 | 69.36 | 54.19 | +15.17 |
| C1 | 75.76 | 69.29 | 55.32 | 38.07 | +17.26 |
| C2 | 85.47 | 81.57 | 70.29 | 54.80 | +15.49 |
| C2b | 85.61 | 81.53 | 70.30 | 54.76 | +15.55 |
| F1 | 86.46 | 82.83 | 71.46 | 56.26 | +15.20 |
| F2a | 86.93 | 82.67 | 71.83 | 56.34 | +15.50 |
| F2b | 87.57 | 83.14 | 72.55 | 56.63 | +15.92 |

> Hai cot **khong so truc tiep duoc**: khac ca split (val vs test) lan cach do. `OBBValidator` khop box bang ProbIoU — xap xi kha vi von dung cho *ham mat mat*, lac quan co he thong; harness dung IoU da giac chinh xac + AP 101 diem. Bang nay chi de cho thay do lon cua khoang cach — **moi ket luan trong bao cao phai lay tu cot harness**.
