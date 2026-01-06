# -*- coding: utf-8 -*-
# 生成 80cm x 80cm 标定板的 A4 纵向 多页 SVG，圆直径 50mm，间距 50mm，页边 5mm
# 无第三方依赖，浏览器打开 SVG 打印为 A4，缩放=100%，边距=无

import math
from datetime import datetime
import os

def mm(v):  # for SVG units
    return f"{v}mm"

def main():
    # 参数
    target_w_mm = 800
    target_h_mm = 800
    circle_d_mm = 50
    pitch_mm = 50
    margin_mm = 5
    green = "rgb(0,170,0)"

    # A4 纵向尺寸
    a4_w_mm, a4_h_mm = 210.0, 297.0
    usable_w_mm = a4_w_mm - 2 * margin_mm
    usable_h_mm = a4_h_mm - 2 * margin_mm
    r_mm = circle_d_mm / 2.0

    # 每页可容纳的列行（保证圆完整落在可用区域）
    page_cols = int(math.floor((usable_w_mm - 2 * r_mm) / pitch_mm)) + 1
    page_rows = int(math.floor((usable_h_mm - 2 * r_mm) / pitch_mm)) + 1
    if page_cols <= 0 or page_rows <= 0:
        raise SystemExit("可绘制区域过小，请降低边距或减小圆直径/间距。")

    # 总网格列行（以左下角为 (0,0)，从 margin 起步）
    total_cols = int(math.floor((target_w_mm - circle_d_mm) / pitch_mm)) + 1
    total_rows = int(math.floor((target_h_mm - circle_d_mm) / pitch_mm)) + 1

    pages_x = int(math.ceil(total_cols / float(page_cols)))
    pages_y = int(math.ceil(total_rows / float(page_rows)))
    total_pages = pages_x * pages_y

    outdir = "grid_svg_pages"
    os.makedirs(outdir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    page_index = 0
    for py in range(pages_y):
        for px in range(pages_x):
            col_offset = px * page_cols
            row_offset = py * page_rows

            # 组装 SVG
            svg = []
            svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{mm(a4_w_mm)}" height="{mm(a4_h_mm)}" viewBox="0 0 {a4_w_mm} {a4_h_mm}">')
            # 可打印区域边框
            x0 = margin_mm
            y0 = margin_mm
            w = usable_w_mm
            h = usable_h_mm
            svg.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="black" stroke-width="0.2"/>')

            # 四角对齐十字
            cross = 5.0
            for (cx, cy) in [(x0, y0), (x0+w, y0), (x0, y0+h), (x0+w, y0+h)]:
                svg.append(f'<line x1="{cx-cross}" y1="{cy}" x2="{cx+cross}" y2="{cy}" stroke="black" stroke-width="0.2"/>')
                svg.append(f'<line x1="{cx}" y1="{cy-cross}" x2="{cx}" y2="{cy+cross}" stroke="black" stroke-width="0.2"/>')

            # 绿色圆
            for j in range(page_rows):
                for i in range(page_cols):
                    col = col_offset + i
                    row = row_offset + j
                    if col >= total_cols or row >= total_rows:
                        continue
                    cx = x0 + r_mm + col * pitch_mm
                    cy = y0 + r_mm + row * pitch_mm
                    # 完整性检查（可不需，但以防浮点误差）
                    if (cx - r_mm) >= x0 and (cx + r_mm) <= (x0 + w) and (cy - r_mm) >= y0 and (cy + r_mm) <= (y0 + h):
                        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_mm}" fill="{green}" stroke="none"/>')

            # 页信息
            page_text = f"80x80 cm grid • Ø={circle_d_mm} mm • pitch={pitch_mm} mm • A4 portrait • page {page_index+1}/{total_pages}"
            svg.append(f'<text x="{x0}" y="{y0-2}" font-size="3" fill="black">{page_text}</text>')
            svg.append(f'<text x="{x0+w}" y="{y0-2}" font-size="3" text-anchor="end" fill="black">{timestamp}</text>')

            svg.append("</svg>")

            fname = os.path.join(outdir, f"grid_a4_portrait_p{page_index+1:02d}_of_{total_pages:02d}.svg")
            with open(fname, "w", encoding="utf-8") as f:
                f.write("\n".join(svg))

            page_index += 1

    print(f"已生成 {total_pages} 个 SVG 文件，目录：{outdir}")
    print(f"页数（横×纵）：{pages_x} × {pages_y}")
    print(f"每页网格点：{page_cols} × {page_rows}")
    print(f"总网格点：{total_cols} × {total_rows}")

if __name__ == "__main__":
    main()