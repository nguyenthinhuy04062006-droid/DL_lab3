"""
Convert REPORT.md into a standalone elegant HTML report embedding all generated figures and tables.
"""
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(ROOT_DIR, "outputs", "REPORT.md")
HTML_PATH = os.path.join(ROOT_DIR, "outputs", "REPORT.html")

with open(MD_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Simple markdown parsing helpers
def parse_markdown_to_html(md):
    lines = md.split("\n")
    html_lines = []
    in_table = False
    in_code = False
    table_header = True

    for line in lines:
        # Code block
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</pre></code>")
                in_code = False
            else:
                lang = line.strip()[3:]
                html_lines.append(f"<pre><code class='language-{lang}'>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            continue
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            continue
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
            continue
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:].strip()}</h4>")
            continue

        # Images
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if img_match:
            alt, src = img_match.groups()
            if src.startswith("outputs/"):
                src = src[len("outputs/"):]
            html_lines.append(f"<div class='figure-box'><img src='{src}' alt='{alt}'><p class='caption'><i>Hình: {alt}</i></p></div>")
            continue

        # Tables
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if all(set(c).issubset({"-", ":"}) for c in cells):
                table_header = False
                continue
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                table_header = True
            
            tag = "th" if table_header else "td"
            row_html = "".join([f"<{tag}>{c}</{tag}>" for c in cells])
            html_lines.append(f"<tr>{row_html}</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

        # Blockquotes
        if line.strip().startswith(">"):
            html_lines.append(f"<blockquote>{line.strip()[1:].strip()}</blockquote>")
            continue

        # Horizontal rule
        if line.strip() in ["---", "***", "___"]:
            html_lines.append("<hr>")
            continue

        # Normal text / Paragraphs
        stripped = line.strip()
        if stripped:
            # Inline bold/code
            formatted = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", stripped)
            formatted = re.sub(r"\*(.*?)\*", r"<em>\1</em>", formatted)
            formatted = re.sub(r"`(.*?)`", r"<code>\1</code>", formatted)
            html_lines.append(f"<p>{formatted}</p>")

    if in_table:
        html_lines.append("</table>")
    if in_code:
        html_lines.append("</pre></code>")

    return "\n".join(html_lines)

body_content = parse_markdown_to_html(text)

# Thêm gallery 5 biểu đồ chuẩn vào cuối trang
gallery_html = """
<hr>
<h2>8. BỘ BIỂU ĐỒ THỰC NGHIỆM CHÍNH THỨC CỦA DỰ ÁN (OUTPUTS/REPORTS/)</h2>
<p>Toàn bộ 5 biểu đồ đồ họa phân tích thực nghiệm được lưu tại thư mục <code>outputs/reports/</code>:</p>

<div class='figure-box'><img src='reports/training_curves.png' alt='Đường cong huấn luyện'><p class='caption'><b>Biểu đồ 1:</b> Đường cong mất mát Loss &amp; Validation Accuracy qua 100 steps (Khớp Cell 27 Notebook)</p></div>
<div class='figure-box'><img src='reports/confusion_matrix.png' alt='Ma trận nhầm lẫn'><p class='caption'><b>Biểu đồ 2:</b> Biểu đồ nhiệt Ma trận nhầm lẫn Confusion Matrix (Khớp Cell 34 Notebook)</p></div>
<div class='figure-box'><img src='reports/test_comparison_and_per_class.png' alt='So sánh hiệu năng'><p class='caption'><b>Biểu đồ 3:</b> So sánh hiệu năng Test Set &amp; chỉ số từng lớp (Khớp Cell 33 Notebook)</p></div>
<div class='figure-box'><img src='reports/eda_distribution.png' alt='Khám phá dữ liệu EDA kèm PCA'><p class='caption'><b>Biểu đồ 4:</b> Khám phá dữ liệu EDA 3 cột: Phân bố nhãn, Phân phối độ dài Token &amp; PCA 2D Không gian dữ liệu thô (Khớp 100% Cell 15 Notebook)</p></div>
<div class='figure-box'><img src='reports/pca_2d_distribution.png' alt='Biểu đồ PCA 2D phóng to'><p class='caption'><b>Biểu đồ 5:</b> Phân tích chuyên sâu không gian vector ngữ nghĩa TF-IDF qua PCA 2D (Độ chồng lấn dữ liệu thô)</p></div>

<hr>
<h2>9. BẢNG BIỂU TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (OUTPUTS/REPORTS/)</h2>
<p>Toàn bộ 5 bảng số liệu tổng hợp được kết xuất trực tiếp dưới dạng ảnh chất lượng cao:</p>

<div class='figure-box'><img src='reports/table_data_split.png' alt='Bảng phân chia dữ liệu'><p class='caption'><b>Bảng 1:</b> Phân chia dữ liệu Train/Validation/Test Set (Chống rò rỉ)</p></div>
<div class='figure-box'><img src='reports/table_training_progress.png' alt='Bảng tiến trình huấn luyện'><p class='caption'><b>Bảng 2:</b> Tiến trình tối ưu hóa qua từng Epoch &amp; Step</p></div>
<div class='figure-box'><img src='reports/table_classification_report.png' alt='Bảng Classification Report'><p class='caption'><b>Bảng 3:</b> Báo cáo phân loại chi tiết từng lớp (Per-class Metrics)</p></div>
<div class='figure-box'><img src='reports/table_benchmark_metrics.png' alt='Bảng đối chiếu hiệu năng'><p class='caption'><b>Bảng 4:</b> Đối chiếu hiệu năng Baseline vs Fine-tuned DistilBERT</p></div>
<div class='figure-box'><img src='reports/table_error_analysis.png' alt='Bảng phân tích cắt cụt token'><p class='caption'><b>Bảng 5:</b> Phân tích tỷ lệ sai số theo mức độ cắt cụt token &amp; Kiểm định Fisher</p></div>
"""

full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Báo Cáo Nghiệm Thu Practice 3 — Hugging Face & DistilBERT</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.65;
        color: #24292e;
        max-width: 960px;
        margin: 40px auto;
        padding: 0 24px;
        background-color: #ffffff;
    }}
    h1 {{ color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 8px; margin-top: 30px; font-size: 26px; }}
    h2 {{ color: #1565c0; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; margin-top: 28px; font-size: 21px; }}
    h3 {{ color: #1976d2; margin-top: 22px; font-size: 17px; }}
    h4 {{ color: #424242; margin-top: 18px; font-size: 15px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    th, td {{ border: 1px solid #e1e4e8; padding: 10px 14px; text-align: left; }}
    th {{ background-color: #f1f8ff; color: #0366d6; font-weight: 600; }}
    tr:nth-child(even) {{ background-color: #fcfcfc; }}
    code {{ background-color: rgba(27,31,35,0.06); padding: 0.2em 0.4em; border-radius: 4px; font-family: Consolas, "Courier New", monospace; font-size: 88%; }}
    pre {{ background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; border: 1px solid #e1e4e8; font-size: 13px; }}
    blockquote {{ padding: 8px 16px; color: #424242; border-left: 4px solid #2196f3; background: #e3f2fd; margin: 18px 0; border-radius: 0 4px 4px 0; }}
    .figure-box {{ text-align: center; margin: 25px 0; padding: 12px; border: 1px solid #eaecef; border-radius: 8px; background: #fafbfc; }}
    .figure-box img {{ max-width: 95%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .caption {{ margin-top: 8px; font-size: 13.5px; color: #586069; }}
    hr {{ border: 0; height: 1px; background: #e1e4e8; margin: 30px 0; }}
    @media print {{
        body {{ max-width: 100%; margin: 0; padding: 10mm; font-size: 12pt; }}
        .figure-box {{ page-break-inside: avoid; }}
    }}
</style>
</head>
<body>
{body_content}
{gallery_html}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"Generated standalone report: {HTML_PATH} ({os.path.getsize(HTML_PATH)/1024:.2f} KB)")
