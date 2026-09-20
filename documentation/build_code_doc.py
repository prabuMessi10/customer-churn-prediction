"""Build the 'Complete Project Source Code' documentation (Markdown + HTML).

Mirrors the reference HCL _All_Code.pdf layout:
  Title: project name, 'Complete Project Source Code',
         'Included code files: N', GitHub repository URL
  Then numbered sections: 'N. <path>/<file>' followed by the full file text.
"""
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "documentation"
GITHUB_URL = "https://github.com/prabuMessi10/customer-churn-prediction"

FILES = [
    ("requirements.txt", "requirements.txt"),
    ("src/preprocess.py", "src/preprocess.py"),
    ("src/train_models.py", "src/train_models.py"),
    ("src/optimise_thresholds.py", "src/optimise_thresholds.py"),
    ("app.py", "app.py"),
    ("streamlit_app.py", "streamlit_app.py"),
]

title = "Customer Churn Prediction"
subtitle = "Complete Project Source Code"


def md_doc() -> str:
    lines = [f"# {title}\n", f"**{subtitle}**\n", f"Included code files: **{len(FILES)}**\n",
             f"GitHub Repository: {GITHUB_URL}\n", "---\n"]
    for i, (label, path) in enumerate(FILES, start=1):
        text = (ROOT / path).read_text(encoding="utf-8")
        lines.append(f"\n## {i}. {label}\n")
        lines.append("```python" if path.endswith(".py") else "```text")
        lines.append(text.rstrip("\n"))
        lines.append("```")
    return "\n".join(lines)


def html_doc() -> str:
    cards = []
    for i, (label, path) in enumerate(FILES, start=1):
        text = (ROOT / path).read_text(encoding="utf-8")
        cards.append(
            f'<div class="file">'
            f'<div class="file-head"><span class="n">{i}.</span> <span class="p">{html.escape(label)}</span></div>'
            f"<pre><code>{html.escape(text.rstrip(chr(10)))}</code></pre></div>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — {subtitle}</title>
<style>
  body {{ margin:0; background:#0f172a; color:#e2e8f0; font-family:"Segoe UI",system-ui,sans-serif; }}
  header {{ background:linear-gradient(135deg,#0f172a,#1e293b 55%,#312e81); padding:34px 26px; text-align:center; border-bottom:3px solid #6366f1; }}
  header h1 {{ margin:0 0 6px; font-size:30px; }}
  header p {{ margin:2px 0; color:#94a3b8; }}
  a {{ color:#a5b4fc; }}
  main {{ max-width:1000px; margin:0 auto; padding:22px; }}
  .file {{ background:#1e293b; border:1px solid #334155; border-radius:12px; margin:18px 0; overflow:hidden; }}
  .file-head {{ background:#334155; padding:10px 14px; font-weight:700; }}
  .n {{ color:#6366f1; }}
  .p {{ color:#e2e8f0; }}
  pre {{ margin:0; padding:14px; overflow-x:auto; background:#0b1220; font:12px/1.55 Consolas,monospace; color:#cbd5e1; white-space:pre; }}
  @media print {{ body {{ background:#fff; color:#000; }} header {{ border:none; }} .file {{ break-inside:avoid; border-color:#999; background:#fff; }} pre {{ background:#f4f4f4; color:#111; }} .file-head {{ background:#eee; color:#000; }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p><strong>{subtitle}</strong></p>
  <p>Included code files: <strong>{len(FILES)}</strong></p>
  <p>GitHub Repository: <a href="{GITHUB_URL}">{GITHUB_URL}</a></p>
</header>
<main>
  <p style="color:#94a3b8">Presentation deck: <a href="Customer_Churn_Prediction_Project_Documentation.pdf">Customer_Churn_Prediction_Project_Documentation.pdf</a></p>
  {''.join(cards)}
</main>
</body>
</html>"""


(OUT / "Customer_Churn_Prediction_Complete_Code.md").write_text(md_doc(), encoding="utf-8")
(OUT / "Customer_Churn_Prediction_Complete_Code.html").write_text(html_doc(), encoding="utf-8")
print("code documentation written to", OUT)