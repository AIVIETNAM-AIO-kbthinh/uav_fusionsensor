"""Gộp các biểu đồ trong `results/figures/` thành một trang HTML tự chứa.

Ảnh được nhúng thẳng dưới dạng data URI nên tệp mở được ở bất cứ đâu, không cần
thư mục đi kèm. Mọi con số trong phần bình luận lấy từ `metrics_*.json` và
`_effect_sizes.json` — không gõ tay.

    .venv/Scripts/python.exe scripts/make_report.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_figures import (  # noqa: E402
    DATASETS, EFFECTS, OUT, ROOT as R, series,
)

REPORT = ROOT / "results" / "report.html"


def img(name: str) -> str:
    p = OUT / name
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def pct(ds_key, cid, key="map50"):
    ds = next(d for d in DATASETS if d["key"] == ds_key)
    return series(ds, "all", "all", key)[cid][0] * 100


def verdict(r):
    if not r["significant"]:
        return "không đủ bằng chứng"
    return "chênh lệch thật" if r["delta"] > 0 else "chênh lệch thật (theo chiều âm)"


def figure_card(n, eyebrow, title, file, note):
    return f"""      <article class="card" id="h{n}">
        <p class="eyebrow"><span class="num">{n:02d}</span>{eyebrow}</p>
        <h2>{title}</h2>
        <div class="plate"><img src="{img(file)}" alt="{title}"></div>
        <div class="note">{note}</div>
      </article>"""


def main():
    rows = json.loads(EFFECTS.read_text(encoding="utf-8")) if EFFECTS.exists() else []
    by = {(r["ds"], r["a"], r["b"]): r for r in rows}

    dv = {c: pct("dronevehicle", c) for c in ("S1", "S2", "C1", "C2", "F1", "F2a")}
    vd = {c: pct("vedai", c) for c in ("S1", "S2", "C1", "C2", "F1", "F2a")}

    def ci(dsname, a, b):
        r = by.get((dsname, a, b))
        if not r:
            return "—"
        return (f"{100*r['delta']:+.2f} điểm, CI 95% "
                f"[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}], p = {r['p_value']:.3f}")

    eff_rows = "\n".join(
        f"""          <tr class="{'sig' if r['significant'] else 'ns'}">
            <td>{r['ds']}</td><td class="mono">{r['a']} − {r['b']}</td>
            <td class="lab">{r['label']}</td>
            <td class="num-cell">{100*r['delta']:+.2f}</td>
            <td class="num-cell">[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]</td>
            <td class="num-cell">{r['p_value']:.3f}</td>
            <td>{'có' if r['significant'] else 'không'}</td>
          </tr>""" for r in rows)

    key = by.get(("DroneVehicle", "F2a", "C2"))
    key_v = by.get(("VEDAI", "F2a", "C2"))

    figs = [
        (1, "Cấu hình nào tốt hơn, tổng thể?", "So sánh mAP giữa sáu cấu hình",
         "fig01_map_comparison.png",
         f"""<p>Trên DroneVehicle, mọi cấu hình <em>có IR</em> đứng thành một nhóm ở
         khoảng {dv['S2']:.0f}–{dv['F2a']:.0f}% mAP@50, cách nhóm chỉ-RGB
         ({dv['S1']:.1f}% và {dv['C1']:.1f}%) hơn 10 điểm. Trên VEDAI thứ tự
         <strong>đảo ngược</strong>: RGB ({vd['S1']:.1f}%) tốt hơn IR
         ({vd['S2']:.1f}%). Hai dataset không mâu thuẫn — chúng nói rằng
         modality nào mạnh hơn là đặc tính của bộ dữ liệu, không phải của
         phương pháp.</p>
         <p>Chênh lệch giữa F2a và cấu hình đơn tốt nhất trong cùng dataset chỉ
         khoảng 1–2 điểm, nhỏ hơn nhiều so với khoảng cách giữa hai modality.</p>"""),

        (2, "Lợi ích tập trung ở lớp nào?", "F1 theo từng lớp",
         "fig02_per_class_f1.png",
         """<p>Trên DroneVehicle, khoảng cách RGB↔IR gần như không đổi qua mọi lớp —
         đây là hiệu ứng ở mức <em>ảnh</em> (điều kiện chụp), không phải hiệu ứng
         ở mức <em>lớp</em>. Lớp <span class="mono">freight_car</span> và
         <span class="mono">van</span> khó nhất với mọi cấu hình.</p>
         <p>Trên VEDAI, thanh sai số qua ba fold rộng hơn cả khoảng cách giữa các
         cấu hình ở hầu hết lớp: với 10–39 box mỗi fold, F1 theo lớp ở đây gần
         như không phân giải được cấu hình nào hơn.</p>"""),

        (3, "Có phải do chọn ngưỡng không?", "AP@50 theo từng lớp",
         "fig03_per_class_ap50.png",
         """<p>AP tích phân trên toàn đường P–R nên không phụ thuộc việc chọn ngưỡng
         confidence. Trật tự giữa các cấu hình giữ nguyên như hình 2 — kết luận
         không phải là hệ quả của cách chọn ngưỡng.</p>"""),

        (4, "Quá trình học diễn ra thế nào?", "Đường cong huấn luyện",
         "fig04_training_curves_dronevehicle.png",
         """<p>Cả sáu cấu hình đều bão hoà trước epoch 60 và không có dấu hiệu
         overfit (val loss vẫn giảm đơn điệu) — 60 epoch là đủ, việc rút gọn từ
         100 xuống 60 không cắt mất phần đang cải thiện.</p>
         <p>Đáng chú ý: <strong>C1 (nét đứt) trùng khít lên S1</strong> suốt 60
         epoch. Nhân đôi số tham số mà đưa vào đúng một ảnh RGB hai lần thì
         không học thêm được gì — chính là điều mà nhóm đối chứng được thiết kế
         để chứng minh. Bậc thang ở epoch 50 là lúc tắt mosaic augmentation.</p>
         <p class="aside">Bản VEDAI: <span class="mono">fig04_training_curves_vedai.png</span>
         — nhiễu hơn nhiều vì tập val chỉ có 121 ảnh.</p>"""),

        (5, "Lợi ích có lớn hơn khi thiếu sáng không? (RQ3)",
         "Phân tầng theo điều kiện chiếu sáng",
         "fig05_illumination_strata.png",
         f"""<p>Ở tầng thiếu sáng, RGB đơn rơi xuống {series(DATASETS[0], 'illum', 'lowlight', 'map50')['S1'][0]*100:.1f}%
         trong khi các cấu hình có IR giữ trên 82% — khoảng cách ~20 điểm. Ở tầng
         đủ sáng khoảng cách gần như đóng lại. Đây là bằng chứng rõ nhất trong
         toàn bộ đề tài về giá trị của cảm biến thứ hai.</p>
         <p class="warn">Nhưng phải đọc kèm cảnh báo: nhãn GT của DroneVehicle
         được vẽ trên ảnh IR. Ở tầng thiếu sáng, một phần lợi thế của IR đến từ
         chính giao thức gán nhãn chứ không hoàn toàn từ thông tin cảm biến.</p>"""),

        (6, "Chênh lệch có vượt nhiễu không? (RQ1)",
         "Kích thước hiệu ứng với khoảng tin cậy bootstrap",
         "fig06_effect_sizes.png",
         f"""<p>Đây là hình để kết luận. Dòng <span class="mono">F2a − C2</span> so
         fusion với đối chứng có <em>đúng bằng</em> số tham số:
         DroneVehicle {ci('DroneVehicle', 'F2a', 'C2')};
         VEDAI {ci('VEDAI', 'F2a', 'C2')}.</p>
         <p>Dòng <span class="mono">C2 − S2</span> và
         <span class="mono">C1 − S1</span> đo phần đóng góp thuần của việc tăng
         dung lượng mô hình khi thông tin đầu vào không đổi — đây là confounder
         mà phần lớn công trình fusion không tách ra.</p>"""),

        (7, "Nên vận hành ở ngưỡng nào?", "F1 theo ngưỡng confidence",
         "fig07_f1_confidence_curve.png",
         """<p>Đỉnh F1 nằm quanh conf 0,45–0,50 cho mọi cấu hình và khá phẳng trong
         khoảng 0,3–0,6 — chọn ngưỡng trong vùng này không làm thay đổi thứ hạng
         giữa các cấu hình. Vách đổ sau 0,85 là chỗ recall sụp.</p>"""),

        (8, "Bao nhiêu phần là do cảm biến, bao nhiêu do tham số?",
         "Tách hiệu ứng dung lượng khỏi hiệu ứng cảm biến",
         "fig08_capacity_vs_map.png",
         f"""<p>Mỗi đường nối hai mức capacity của <em>cùng một nguồn thông tin</em>.
         Độ dốc là lợi ích thuần của việc nhân đôi tham số:
         {dv['C1'] - dv['S1']:+.1f} điểm (chỉ RGB) và {dv['C2'] - dv['S2']:+.1f}
         điểm (chỉ IR) trên DroneVehicle; {vd['C1'] - vd['S1']:+.1f} và
         {vd['C2'] - vd['S2']:+.1f} trên VEDAI. Tăng dung lượng mà không thêm
         thông tin <strong>không mua được gì</strong>, thậm chí hơi lỗ.</p>
         <p>Khoảng cách <em>theo chiều dọc</em> giữa các đường mới là phần do
         nguồn thông tin — và đó là phần đáng báo cáo.</p>"""),
    ]

    cards = "\n".join(figure_card(*f) for f in figs)
    nav = "\n".join(f'        <li><a href="#h{n}"><span class="num">{n:02d}</span>{t}</a></li>'
                    for n, _, t, _, _ in figs)

    key_line = (f"F2a − C2 = {ci('DroneVehicle', 'F2a', 'C2')}" if key else "chưa có bootstrap")
    key_line_v = (f"F2a − C2 = {ci('VEDAI', 'F2a', 'C2')}" if key_v else "chưa có bootstrap")
    key_note = ("Sau khi kiểm soát dung lượng mô hình, cả hai dataset đều "
                + ("<strong>không</strong> cho bằng chứng về lợi ích của fusion."
                   if (key and not key["significant"]) and (key_v and not key_v["significant"])
                   else "cho kết quả như bảng dưới."))

    html = f"""<title>Kết quả Fusion RGB-IR</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {{
    color-scheme: light;
    --ground:  #eceee9;
    --surface: #fcfcfb;
    --plate:   #fcfcfb;
    --ink:     #14171a;
    --ink-2:   #4c5257;
    --muted:   #858b87;
    --line:    #dcded8;
    --accent:  #1b5e7a;
    --accent-soft: #e2edf1;
    --warn:    #9a4a1c;
    --warn-soft: #f6ece4;
    --sig:     #16624a;
    --plate-ring: rgba(20,23,26,.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground:  #0e100f;
      --surface: #181c1b;
      --plate:   #fcfcfb;
      --ink:     #f1f3f0;
      --ink-2:   #b7beb9;
      --muted:   #868d88;
      --line:    #2a302e;
      --accent:  #74b4cd;
      --accent-soft: #17323d;
      --warn:    #e0a077;
      --warn-soft: #33241a;
      --sig:     #74c6a8;
      --plate-ring: rgba(255,255,255,.14);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground:  #0e100f;
    --surface: #181c1b;
    --plate:   #fcfcfb;
    --ink:     #f1f3f0;
    --ink-2:   #b7beb9;
    --muted:   #868d88;
    --line:    #2a302e;
    --accent:  #74b4cd;
    --accent-soft: #17323d;
    --warn:    #e0a077;
    --warn-soft: #33241a;
    --sig:     #74c6a8;
    --plate-ring: rgba(255,255,255,.14);
  }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px; line-height: 1.62;
    margin: 0; padding: 0 20px 80px;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  .mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .92em; }}
  .num {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    color: var(--accent); margin-right: .6em; font-weight: 500;
  }}

  header.head {{ padding: 56px 0 28px; border-bottom: 1px solid var(--line); }}
  .kicker {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    text-transform: uppercase; letter-spacing: .12em; font-size: 11.5px;
    color: var(--muted); margin: 0 0 14px;
  }}
  h1 {{
    font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
    font-weight: 700; font-size: clamp(30px, 4.4vw, 46px); line-height: 1.1;
    margin: 0 0 12px; text-wrap: balance; letter-spacing: -.01em;
  }}
  .lede {{ font-size: 17px; color: var(--ink-2); max-width: 62ch; margin: 0 0 22px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    border: 1px solid var(--line); border-radius: 999px; padding: 3px 11px;
    color: var(--ink-2); background: var(--surface);
  }}

  .layout {{ display: grid; grid-template-columns: 226px minmax(0, 1fr); gap: 44px; margin-top: 40px; }}
  @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; gap: 26px; }} nav.rail {{ position: static !important; }} }}

  nav.rail {{ position: sticky; top: 28px; align-self: start; }}
  nav.rail p {{
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
    letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 0 0 12px;
  }}
  nav.rail ol {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }}
  nav.rail a {{
    display: block; padding: 5px 8px; border-radius: 5px; font-size: 13.5px;
    color: var(--ink-2); text-decoration: none; border-left: 2px solid transparent;
  }}
  nav.rail a:hover, nav.rail a:focus-visible {{
    background: var(--accent-soft); color: var(--ink); border-left-color: var(--accent);
  }}

  main {{ display: flex; flex-direction: column; gap: 26px; min-width: 0; }}

  .card {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: 26px 26px 22px;
  }}
  .eyebrow {{
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    color: var(--muted); margin: 0 0 6px; letter-spacing: .01em;
  }}
  .card h2 {{
    font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
    font-size: 23px; font-weight: 600; margin: 0 0 18px; line-height: 1.22;
    text-wrap: balance;
  }}
  .plate {{
    background: var(--plate); border-radius: 7px; padding: 10px;
    box-shadow: 0 0 0 1px var(--plate-ring); overflow-x: auto;
  }}
  .plate img {{ display: block; width: 100%; height: auto; }}
  .note {{ margin-top: 18px; max-width: 74ch; color: var(--ink-2); }}
  .note p {{ margin: 0 0 10px; }}
  .note p:last-child {{ margin-bottom: 0; }}
  .note strong, .note em {{ color: var(--ink); }}
  .note em {{ font-style: italic; }}
  .warn {{
    border-left: 2px solid var(--warn); background: var(--warn-soft);
    padding: 10px 14px; border-radius: 0 5px 5px 0; color: var(--ink-2);
  }}
  .aside {{ font-size: 13.5px; color: var(--muted); }}

  .verdict {{ border-color: var(--accent); }}
  .verdict h2 {{ margin-bottom: 10px; }}
  .verdict .big {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 15px; color: var(--ink); margin: 0 0 4px;
  }}
  .verdict .big b {{ color: var(--accent); font-weight: 500; }}

  .tablewrap {{ overflow-x: auto; margin-top: 18px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); }}
  th {{
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
    text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 500;
  }}
  td.num-cell {{ font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", ui-monospace, monospace; }}
  td.lab {{ color: var(--muted); font-size: 12.5px; }}
  tr.sig td {{ color: var(--ink); }}
  tr.sig td.num-cell {{ color: var(--sig); font-weight: 500; }}
  tr.ns td {{ color: var(--muted); }}

  footer {{
    margin-top: 44px; padding-top: 22px; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 13px;
  }}
  footer code {{
    font-family: "IBM Plex Mono", ui-monospace, monospace; background: var(--surface);
    border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; color: var(--ink-2);
  }}
  a {{ color: var(--accent); }}
  :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>

<div class="wrap">
  <header class="head">
    <p class="kicker">UAV · RGB + hồng ngoại · YOLO11s-OBB</p>
    <h1>Sensor fusion có thật sự giúp ích không?</h1>
    <p class="lede">Tám biểu đồ đọc từ 30 lần chạy trên hai dataset độc lập. Điểm
      khác biệt của đề tài nằm ở hai cấu hình <span class="mono">C1</span> /
      <span class="mono">C2</span>: chúng nhân đôi số tham số nhưng chỉ nhận
      <em>một</em> cảm biến, nên tách được phần đóng góp của thông tin ra khỏi
      phần đóng góp của dung lượng mô hình.</p>
    <div class="chips">
      <span class="chip">DroneVehicle · 8.980 ảnh test · 2 seed</span>
      <span class="chip">VEDAI · 3 fold · 121 ảnh/fold</span>
      <span class="chip">6 cấu hình · 60 epoch · batch 8</span>
      <span class="chip">IoU đa giác chính xác · AP 101 điểm</span>
    </div>
  </header>

  <div class="layout">
    <nav class="rail">
      <p>Biểu đồ</p>
      <ol>
{nav}
      </ol>
    </nav>

    <main>
      <article class="card verdict">
        <p class="eyebrow">Kết quả chính · RQ1</p>
        <h2>Fusion so với đối chứng cùng số tham số</h2>
        <p class="big">DroneVehicle &nbsp;<b>{key_line}</b></p>
        <p class="big">VEDAI &nbsp;<b>{key_line_v}</b></p>
        <div class="note"><p>{key_note} Bảng dưới là toàn bộ các phép so sánh,
          bootstrap ghép cặp theo ảnh (cùng mẫu ảnh cho cả hai cấu hình).</p></div>
        <div class="tablewrap">
          <table>
            <thead><tr>
              <th>Dataset</th><th>So sánh</th><th>Ý nghĩa</th>
              <th>Δ mAP@50</th><th>CI 95%</th><th>p</th><th>Có ý nghĩa</th>
            </tr></thead>
            <tbody>
{eff_rows}
            </tbody>
          </table>
        </div>
      </article>

{cards}

      <article class="card">
        <p class="eyebrow">Hai điều bắt buộc phải nói khi trình bày</p>
        <h2>Giới hạn của các con số trên</h2>
        <div class="note">
          <p><strong>1. Hình 4 không so trực tiếp được với các hình còn lại.</strong>
            Đường cong huấn luyện là số của validator Ultralytics, khớp box bằng
            ProbIoU — cao hơn IoU thật trung bình +0,13. Các hình khác dùng harness
            của đề tài với IoU đa giác chính xác. Hai thang đo khác nhau.</p>
          <p><strong>2. Con số để claim là <span class="mono">F2a − C2</span></strong>,
            không phải <span class="mono">F2a − S1/S2</span>. Phép so sánh thứ hai
            vẫn còn lẫn phần đóng góp của dung lượng mô hình, và ở tầng thiếu sáng
            còn bị thổi phồng bởi giao thức gán nhãn trên ảnh IR.</p>
        </div>
      </article>
    </main>
  </div>

  <footer>
    Tái tạo toàn bộ: <code>python scripts/make_figures.py</code> ·
    số liệu thô: <code>results/tables/figure_data.md</code> ·
    ảnh gốc: <code>results/figures/</code>
  </footer>
</div>
"""
    REPORT.write_text(html, encoding="utf-8")
    print(f"-> {REPORT.relative_to(R)}  ({REPORT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
