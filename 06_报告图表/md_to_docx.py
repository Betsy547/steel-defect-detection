#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把实践报告 Markdown 初稿转成排版好的 Word 文档。

用法（在项目根目录）：
    python 06_报告图表/md_to_docx.py

输出：
    实践报告-初稿.docx

说明：
- 不依赖 pandoc / LibreOffice，只用 python-docx。
- 正文宋体小四、1.5 倍行距、首行缩进 2 字符；章标题三号黑体居中。
- 表格五号字、带表头底纹；图题、表题居中。
- 页脚居中页码。
"""

import re
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "实践报告-初稿.md"
OUT = ROOT / "实践报告-初稿.docx"

TEXT_WIDTH_CM = 14.6
BODY_PT = 12
TABLE_PT = 10.5


def set_east_asia(run_or_style, name):
    rpr = run_or_style._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)


def shade_cell(cell, fill="E7E6E6"):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def shade_paragraph(p, fill="F2F2F2"):
    ppr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def remove_table_borders(table):
    tblpr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tblpr.append(borders)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)

    run = p.add_run()
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"
    set_east_asia(run, "宋体")

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def _add_tokens(p, text, size, bold, code_size):
    for part in re.split(r"(`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            run = p.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(code_size or max(size - 1.5, 8))
            run.bold = bold
        else:
            run = p.add_run(part)
            run.bold = bold
            run.font.size = Pt(size)
        set_east_asia(run, "宋体")


def add_inline(p, text, size=BODY_PT, bold=False, code_size=None):
    """把 **粗体** 和 `代码` 解析成 run（支持粗体里嵌套代码）。"""
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) >= 4:
            _add_tokens(p, part[2:-2], size, True, code_size)
        else:
            _add_tokens(p, part, size, bold, code_size)


def cropped_image(path):
    """裁掉近似纯白的外边距，让 matplotlib 图在 Word 里更清楚。"""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return path
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L").point(lambda v: 255 if v > 8 else 0)
    bbox = diff.getbbox()
    if not bbox:
        return path
    pad = 10
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.width, bbox[2] + pad)
    bottom = min(img.height, bbox[3] + pad)
    if right - left < img.width * 0.98 or bottom - top < img.height * 0.98:
        out = Path(tempfile.gettempdir()) / f"docx_{path.stem}_{img.width}x{img.height}.png"
        img.crop((left, top, right, bottom)).save(out)
        return out
    return path


def disp_len(s):
    return sum(2 if ord(ch) > 127 else 1 for ch in s)


def add_caption(doc, text, kind):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(6 if kind == "figure" else 12)
    p.paragraph_format.space_after = Pt(12 if kind == "figure" else 3)
    if kind == "table":
        p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.0
    add_inline(p, text, size=10.5, bold=True)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
        run.bold = True
        set_east_asia(run, "黑体" if kind == "table" else "宋体")


def add_image(doc, path, alt, width_cm=TEXT_WIDTH_CM, caption=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3 if caption else 6)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run()
    run.add_picture(str(cropped_image(path)), width=Cm(width_cm))
    if caption and alt:
        add_caption(doc, alt, "figure")


def add_code_block(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.right_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    shade_paragraph(p)
    for i, line in enumerate(lines):
        if i:
            p.add_run().add_break()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        set_east_asia(run, "宋体")


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    header, sep, body = rows[0], rows[1], rows[2:]
    aligns = []
    for s in sep:
        s = s.strip()
        if s.startswith(":") and s.endswith(":"):
            aligns.append(WD_ALIGN_PARAGRAPH.CENTER)
        elif s.endswith(":"):
            aligns.append(WD_ALIGN_PARAGRAPH.RIGHT)
        else:
            aligns.append(WD_ALIGN_PARAGRAPH.LEFT)
    return header, body, aligns


def add_table(doc, lines):
    header, body, aligns = parse_table(lines)
    ncol = len(header)
    table = doc.add_table(rows=1 + len(body), cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_cell_margins(table)

    font_pt = TABLE_PT if ncol < 7 else 9
    weights = []
    for j in range(ncol):
        w = max([disp_len(header[j])] + [disp_len(r[j]) if j < len(r) else 0 for r in body])
        weights.append(max(w, 6))
    total = sum(weights)
    min_w = 1.7 if ncol <= 5 else 1.35
    widths = [max(TEXT_WIDTH_CM * w / total, min_w) for w in weights]
    scale = TEXT_WIDTH_CM / sum(widths)
    widths = [w * scale for w in widths]

    rows = [header] + body
    for i, cells in enumerate(rows):
        row = table.rows[i]
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j in range(ncol):
            cell = row.cells[j]
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i < len(rows) - 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else aligns[j]
            text = cells[j] if j < len(cells) else ""
            add_inline(p, text, size=font_pt, bold=(i == 0))
            if i == 0:
                shade_cell(cell)


def add_sample_table(doc, rows):
    """图 2 用的无边框样例表：标签行 + 图片行。"""
    rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in r if c.strip())]
    table = doc.add_table(rows=len(rows), cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    remove_table_borders(table)
    for i, cells in enumerate(rows):
        for j, cell_text in enumerate(cells):
            cell = table.rows[i].cells[j]
            cell.width = Cm(TEXT_WIDTH_CM / 3)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            m = re.match(r"!\[(.*?)\]\((.*?)\)", cell_text)
            if m:
                run = p.add_run()
                run.add_picture(str(cropped_image(ROOT / m.group(2))), width=Cm(4.2))
            else:
                add_inline(p, cell_text, size=10.5, bold=True)
    return table


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)
    add_page_number(section)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(BODY_PT)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    set_east_asia(normal, "宋体")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    title_style = doc.styles.add_style("报告标题", WD_STYLE_TYPE.PARAGRAPH)
    title_style.base_style = doc.styles["Normal"]
    title_style.font.name = "Times New Roman"
    title_style.font.size = Pt(20)
    title_style.font.bold = True
    set_east_asia(title_style, "黑体")
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.space_before = Pt(0)
    title_style.paragraph_format.space_after = Pt(18)
    title_style.paragraph_format.line_spacing = 1.5

    heading_specs = {
        "Heading 1": (16, 18, 8),
        "Heading 2": (14, 12, 6),
        "Heading 3": (12, 10, 4),
    }
    for name, (size, before, after) in heading_specs.items():
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        set_east_asia(st, "黑体")
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.keep_with_next = True
        if name == "Heading 1":
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    i = 0
    skip_fig_caption = False
    needs_gap = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith(">"):
            i += 1
            continue

        pending_gap = needs_gap
        needs_gap = False

        if stripped.startswith("```"):
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j].rstrip())
                j += 1
            add_code_block(doc, code)
            i = j + 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                p = doc.add_paragraph(style="报告标题")
                if "系统设计与实现" in text:
                    head, tail = text.split("设计与实现", 1)
                    run = p.add_run(head)
                    run.add_break()
                    p.add_run("设计与实现" + tail)
                else:
                    p.add_run(text)
                for run in p.runs:
                    run.font.size = Pt(20)
                    run.bold = True
                    set_east_asia(run, "黑体")
            else:
                style = {2: "Heading 1", 3: "Heading 2", 4: "Heading 3"}[level]
                doc.add_paragraph(text, style=style)
            skip_fig_caption = False
            i += 1
            continue

        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", stripped)
        if m:
            alt, src = m.group(1), m.group(2)
            add_image(doc, ROOT / src, alt)
            skip_fig_caption = True
            i += 1
            continue

        if skip_fig_caption:
            skip_fig_caption = False
            if re.match(r"^\*\*图\s*\d", stripped):
                i += 1
                continue

        if stripped.startswith("|"):
            j = i
            table_lines = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            if any("![" in line for line in table_lines):
                rows = [[c.strip() for c in r.strip("|").split("|")] for r in table_lines]
                add_sample_table(doc, rows)
                needs_gap = False
            else:
                add_table(doc, table_lines)
                needs_gap = True
            i = j
            continue

        if re.match(r"^\*\*表\s*\d", stripped):
            add_caption(doc, stripped, "table")
            i += 1
            continue

        if re.match(r"^\*\*图\s*\d", stripped):
            add_caption(doc, stripped, "figure")
            i += 1
            continue

        if re.match(r"^\[\d+\]", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.first_line_indent = Cm(-0.74)
            p.paragraph_format.line_spacing = 1.25
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_inline(p, stripped, size=10.5)
            i += 1
            continue

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.5
        if pending_gap:
            p.paragraph_format.space_before = Pt(6)
        if not stripped.startswith("**关键词**"):
            p.paragraph_format.first_line_indent = Pt(24)
        add_inline(p, stripped, size=BODY_PT)
        i += 1

    props = doc.core_properties
    props.title = "基于 YOLO11s 的热轧钢带表面缺陷检测系统设计与实现"
    props.subject = "计算机视觉课程实践报告"
    props.author = "Betsy547"

    doc.save(OUT)
    print(f"已生成：{OUT}")


if __name__ == "__main__":
    main()
