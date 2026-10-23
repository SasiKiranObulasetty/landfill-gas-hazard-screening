"""
generate_pdf_report.py - Converts Final_Research_Report.md into a publication PDF.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.
"""

import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MD_PATH = REPORTS_DIR / "Final_Research_Report.md"
PDF_PATH = REPORTS_DIR / "Final_Research_Report.pdf"


def markdown_to_pdf(md_file: Path, pdf_file: Path):
    """Parses markdown headers, paragraphs, and tables into ReportLab flowables."""
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1f2937')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    flowables = []

    lines = text.split("\n")
    i = 0
    in_table = False
    table_lines = []

    def flush_table(t_lines):
        if not t_lines:
            return None
        # Parse markdown table
        rows = []
        for tl in t_lines:
            tl = tl.strip()
            if not tl.startswith("|") or "---" in tl:
                continue
            cells = [c.strip() for c in tl.split("|")[1:-1]]
            rows.append(cells)

        if not rows:
            return None

        table_data = []
        for r_idx, r in enumerate(rows):
            row_cells = []
            for c in r:
                # remove bold markdown markers
                clean_c = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", c)
                clean_c = clean_c.replace("$", "").replace("\\ge", ">=").replace("\\le", "<=")
                if r_idx == 0:
                    row_cells.append(Paragraph(clean_c, table_header_style))
                else:
                    row_cells.append(Paragraph(clean_c, table_cell_style))
            table_data.append(row_cells)

        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t

    while i < len(lines):
        line = lines[i].strip()

        # Handle table
        if line.startswith("|"):
            table_lines.append(line)
            i += 1
            continue
        elif table_lines:
            t_flowable = flush_table(table_lines)
            if t_flowable:
                flowables.append(Spacer(1, 6))
                flowables.append(t_flowable)
                flowables.append(Spacer(1, 8))
            table_lines = []

        if not line:
            i += 1
            continue

        # Skip horizontal rules
        if line.startswith("---"):
            flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=8, spaceAfter=8))
            i += 1
            continue

        # Title (# )
        if line.startswith("# "):
            flowables.append(Paragraph(line[2:].strip(), title_style))
            flowables.append(Spacer(1, 8))
        # H1 (## )
        elif line.startswith("## "):
            flowables.append(Paragraph(line[3:].strip(), h1_style))
        # H2 (### )
        elif line.startswith("### "):
            flowables.append(Paragraph(line[4:].strip(), h2_style))
        # Bullet list (- or * or numbered)
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            bullet_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", bullet_text)
            flowables.append(Paragraph(f"• {bullet_text}", bullet_style))
        # Numbered list
        elif re.match(r"^\d+\.\s+", line):
            num_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            flowables.append(Paragraph(num_text, bullet_style))
        # Standard paragraph
        else:
            p_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            p_text = p_text.replace("<br>", "<br/>")
            flowables.append(Paragraph(p_text, body_style))

        i += 1

    if table_lines:
        t_flowable = flush_table(table_lines)
        if t_flowable:
            flowables.append(t_flowable)

    doc.build(flowables)
    print(f"Successfully compiled PDF report to {pdf_file}")


if __name__ == "__main__":
    if MD_PATH.exists():
        markdown_to_pdf(MD_PATH, PDF_PATH)
    else:
        print(f"Error: {MD_PATH} not found.")
