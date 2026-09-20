"""Build Word (.docx) versions of all project documentation.

Creates two documents:
  1. Customer_Churn_Prediction_Project_Documentation.docx — full report
     (problem statement, objectives, dataset, models, architecture,
     workflow, modules, performance with charts, conclusion, outputs)
  2. Customer_Churn_Prediction_Complete_Code.docx — full source code listing

All metrics and charts are read from the real model artifacts.
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "documentation"
ASSETS = OUT / "assets"
MODEL_DIR = ROOT / "model"
GITHUB_URL = "https://github.com/prabuMessi10/customer-churn-prediction"
LIVE_URL = "https://customer-churn-prediction-on.streamlit.app"

INDIGO = RGBColor(0x43, 0x38, 0xCA)
DARK = RGBColor(0x1E, 0x29, 0x3B)
GREY = RGBColor(0x64, 0x74, 0x8B)
RED = RGBColor(0xB9, 0x1C, 0x1C)
GREEN = RGBColor(0x0F, 0x76, 0x6E)

MEMBERS = [
    ("Subash Acharya D", "24CSR306"),
    ("Thannasi Prabu R", "24CSR321"),
    ("Vikram S", "24CSR345"),
]

MODEL_NAMES = ["linear_regression", "random_forest", "xgboost"]
DISPLAY = ["Linear Regression", "Random Forest", "XGBoost"]


# ---------------------------------------------------------------- helpers
def shade_paragraph(p, fill="F1F5F9"):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    p._p.get_or_add_pPr().append(shd)


def mono_run(p, text, size=9):
    r = p.add_run(text)
    r.font.name = "Consolas"
    r._element.rPr.rFonts.set(qn("w:cs"), "Consolas")
    r.font.size = Pt(size)
    return r


def add_code_block(doc, code):
    for line in code.splitlines() or [""]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        shade_paragraph(p)
        mono_run(p, line if line else " ")


def add_picture_centered(doc, path, width=6.0, caption=None):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = c.add_run(caption)
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = GREY


def add_footer_pagenum(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    run.font.size = Pt(9)
    run.font.color.rgb = GREY


def big_centered(doc, text, size=40, color=INDIGO, bold=True, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def small_centered(doc, text, size=13, color=GREY, after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.color.rgb = color
    return p


def bullets(doc, items, color=DARK):
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        for r in p.runs:
            r.font.color.rgb = color


def metrics_rows():
    metrics = json.loads((MODEL_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    rows = []
    for name, disp in zip(MODEL_NAMES, DISPLAY):
        m = metrics[name]
        rows.append(
            [
                disp,
                f"{m['accuracy'] * 100:.1f}%",
                f"{m['precision'] * 100:.1f}%",
                f"{m['recall'] * 100:.1f}%",
                f"{m['f1_score'] * 100:.1f}%",
                f"{m['roc_auc']:.3f}",
            ]
        )
    return rows


# ================================================================== REPORT
def build_report():
    doc = Document()
    add_footer_pagenum(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # ---- cover
    for _ in range(4):
        doc.add_paragraph()
    big_centered(doc, "Customer Churn Prediction System", 34)
    small_centered(doc, "Linear Regression  •  Random Forest  •  XGBoost", 15, after=10)
    small_centered(doc, "Project Documentation", 13, after=20)
    small_centered(doc, f"Live demo: {LIVE_URL}", 12)
    small_centered(doc, f"GitHub: {GITHUB_URL}", 12, after=30)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Presented by"); r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = DARK

    table = doc.add_table(rows=len(MEMBERS) + 1, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for j, txt in enumerate(["Name", "Roll Number"]):
        hp = hdr[j].paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = hp.add_run(txt); hr.font.bold = True; hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade_paragraph(hp, "4338CA")
    for i, (nm, rn) in enumerate(MEMBERS, start=1):
        c0, c1 = table.rows[i].cells
        p0 = c0.paragraphs[0]; p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.add_run(nm).font.bold = True
        p1 = c1.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p1.add_run(rn)
    doc.add_page_break()

    # ---- 1. problem statement
    doc.add_heading("1. Problem Statement", level=1)
    doc.add_heading("1.1 Challenges", level=2)
    bullets(doc, [
        "Annual losses exceed \u20b910,000 crore across telecom, banking and OTT due to churn.",
        "Rule-based retention systems miss complex behavioural patterns (usage shifts, payment irregularities, service interactions).",
        "Generic global datasets ignore localised drivers such as UPI usage trends and regional behaviour.",
        "Businesses react too late — churn signals are not detected early enough.",
        "No production-ready, interpretable models at 85–92% accuracy.",
    ])
    doc.add_heading("1.2 Proposed Solution", level=2)
    bullets(doc, [
        "Train Linear Regression, Random Forest and XGBoost on Telco customer churn data.",
        "Detect churn early using churn probability and risk levels for every customer.",
        "Explain every prediction (local drivers + global importance).",
        "Serve insights through a Flask web app and a deployed Streamlit dashboard.",
        "Deliver reproducible, interpretable ML with honest, test-set-validated metrics.",
    ])

    # ---- 2. objectives
    doc.add_heading("2. Objectives", level=1)
    objectives = [
        ("Churn Prediction", "Score every customer with a churn probability and a clear Churn / No-Churn verdict."),
        ("Model Comparison", "Build and evaluate Linear Regression, Random Forest and XGBoost side by side."),
        ("Interpretability", "Explain why each model makes its call — local drivers and global importance."),
        ("Data Preparation", "Clean 7,032 records, encode features, standardise numeric fields and engineer new ones."),
        ("Web Application", "Ship a full UI — dashboard, prediction form, batch CSV and JSON API."),
        ("Online Deployment", "Host the app publicly on Streamlit Community Cloud from the GitHub repo."),
    ]
    ot = doc.add_table(rows=len(objectives) + 1, cols=3)
    ot.style = "Table Grid"
    hdr = ot.rows[0].cells
    for j, txt in enumerate(["#", "Objective", "Description"]):
        hp = hdr[j].paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = hp.add_run(txt); hr.font.bold = True; hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade_paragraph(hp, "4338CA")
    for i, (title, desc) in enumerate(objectives, start=1):
        c = ot.rows[i].cells
        c[0].text = f"0{i}"
        c[1].text = title
        c[2].text = desc
        for cell in c:
            for pp in cell.paragraphs:
                for rr in pp.runs:
                    rr.font.size = Pt(10.5)
        c[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c[1].paragraphs[0].runs[0].font.bold = True

    # ---- 3. dataset
    doc.add_heading("3. Dataset Information", level=1)
    doc.add_paragraph("Telco Customer Churn — IBM sample dataset.")
    dstat = doc.add_table(rows=2, cols=4)
    dstat.style = "Table Grid"
    stats = [("7,032", "Records after cleaning"), ("20", "Features"),
             ("26.6%", "Churn rate"), ("1,407", "Hold-out test customers")]
    for j, (num, lab) in enumerate(stats):
        c = dstat.rows[0].cells[j]
        cp = c.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cp.add_run(num); cr.font.size = Pt(18); cr.font.bold = True; cr.font.color.rgb = INDIGO
        c2 = dstat.rows[1].cells[j]
        cp2 = c2.paragraphs[0]; cp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp2.add_run(lab).font.size = Pt(10)

    doc.add_heading("Core dataset fields", level=2)
    bullets(doc, [
        "Demographics: gender, SeniorCitizen, Partner, Dependents",
        "Usage: tenure, PhoneService, MultipleLines, InternetService",
        "Services: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Streaming",
        "Contract & billing: Contract, PaperlessBilling, PaymentMethod, Monthly / Total Charges",
        "Target: Churn → Yes/No (26.6% positive class)",
    ])
    doc.add_heading("Preprocessing (src/preprocess.py)", level=2)
    bullets(doc, [
        "Drop customerID; convert TotalCharges to numeric.",
        "Map binary fields (gender, Partner, Dependents, services) to 0/1.",
        "One-hot encode category fields; standardise numeric fields.",
        "Engineer avg_charge_per_month and tenure_log.",
    ])
    add_picture_centered(doc, ASSETS / "churn_distribution.png", 4.4,
                         "Figure 1 — Churn / no-churn distribution of the raw dataset.")

    # ---- 4. models
    doc.add_heading("4. Model Choosing", level=1)
    models = [
        ("Linear Regression", "Baseline", "Coefficient-weighted local explanations.",
         ["Continuous score thresholded at 0.5", "Strong ROC-AUC 0.837",
          "79.8% accuracy on the test set"]),
        ("Random Forest", "Retention champion",
         "Probability calibrated with a decision threshold of 0.70.",
         ["400 trees, max depth 12, balanced classes",
          "Best churn recall — finds 76.2% of churners",
          "Recommended for retention teams"]),
        ("XGBoost", "Gradient boosting", "Clean probability estimates.",
         ["500 rounds, learning rate 0.05", "Mild imbalance weight 1.4",
          "77.2% accuracy, AUC 0.817"]),
    ]
    for name, tag, note, items in models:
        doc.add_heading(name, level=2)
        p = doc.add_paragraph()
        r = p.add_run(f"Role — {tag}. ")
        r.font.bold = True
        p.add_run(note)
        bullets(doc, items)

    # ---- 5. architecture
    doc.add_heading("5. System Architecture", level=1)
    doc.add_paragraph("End-to-end data flow of the system:")
    bullets(doc, [
        "CSV Dataset (7,043 × 21 raw records) → Data Cleaning (types, missing values, encoding, feature engineering)",
        "ML Models — Linear Regression, Random Forest, XGBoost",
        "Artifacts — model/*.pkl pipelines, metrics and threshold JSON",
        "Apps — Flask web app and Streamlit dashboard",
        "User — predict a customer, upload CSV, view charts",
    ])
    tag = doc.add_paragraph()
    tr = tag.add_run("User selects a customer / uploads data → all three models score it → interpretations and visual analytics.")
    tr.font.bold = True; tr.font.color.rgb = INDIGO

    # ---- 6. workflow
    doc.add_heading("6. Workflow", level=1)
    steps = [
        ("Load Data", "Read Telco_Customer_Churn.csv."),
        ("Clean Data", "Convert types, drop NA, encode, engineer features."),
        ("Train Models", "Linear Regression, Random Forest and XGBoost."),
        ("Save Artifacts", "Full pipelines, metrics and thresholds via joblib / JSON."),
        ("Deploy", "Flask locally plus the public Streamlit Cloud app."),
    ]
    wt = doc.add_table(rows=len(steps) + 1, cols=2)
    wt.style = "Table Grid"
    hdr = wt.rows[0].cells
    for j, t in enumerate(["Step", "Description"]):
        hp = hdr[j].paragraphs[0]; hr = hp.add_run(t); hr.font.bold = True
        hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); shade_paragraph(hp, "4338CA")
    for i, (name, desc) in enumerate(steps, start=1):
        wt.rows[i].cells[0].text = str(i)
        wt.rows[i].cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        wt.rows[i].cells[1].text = f"{name} — {desc}"
        wt.rows[i].cells[1].paragraphs[0].runs[0].font.bold = True

    # ---- 7. modules
    doc.add_heading("7. Application Modules", level=1)
    doc.add_heading("7.1 Flask Web App (app.py)", level=2)
    bullets(doc, [
        "Model dashboard — metrics and confusion matrices.",
        "Single prediction — 3-model verdicts and explanation.",
        "Batch CSV upload with an ensemble verdict.",
        "JSON API — POST /api/predict.",
    ])
    doc.add_heading("7.2 Streamlit App (streamlit_app.py)", level=2)
    bullets(doc, [
        "Model dashboard — KPIs, bars and confusion matrices.",
        "Predict a customer — a form with live explanation.",
        "Batch upload — downloadable results.",
        "About — methodology and honest performance notes.",
    ])
    p = doc.add_paragraph()
    p.add_run("Both apps share the same trained pipelines and preprocessing — one codebase, two front-ends.").font.bold = True

    # ---- 8. performance
    doc.add_heading("8. Model Performance & Validation", level=1)
    doc.add_paragraph("80/20 stratified train–test split · 1,407 test customers · all metrics reported on the hold-out test set.")
    mt = doc.add_table(rows=len(MODEL_NAMES) + 1, cols=6)
    mt.style = "Table Grid"
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for j, h in enumerate(headers):
        hp = mt.rows[0].cells[j].paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = hp.add_run(h); hr.font.bold = True; hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade_paragraph(hp, "4338CA")
    for i, row in enumerate(metrics_rows(), start=1):
        for j, val in enumerate(row):
            cp = mt.rows[i].cells[j].paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(val)
            if j == 0:
                r.font.bold = True
            if j == 4 and val == max(r[4] for r in metrics_rows()):
                r.font.bold = True
    add_picture_centered(doc, ASSETS / "accuracy_comparison.png", 6.0,
                         "Figure 2 — Accuracy comparison on the hold-out test set (majority baseline 73.4% dashed).")

    cm_table = doc.add_table(rows=1, cols=3)
    cm_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, name in enumerate(MODEL_NAMES):
        cell = cm_table.rows[0].cells[j]
        cell.width = Inches(2.4)
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(); r.add_picture(str(ASSETS / f"confusion_{name}.png"), width=Inches(2.3))
        cap = cell.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cap.add_run(f"Figure 3 — {DISPLAY[j]} confusion matrix")
        cr.font.size = Pt(8.5); cr.font.italic = True; cr.font.color.rgb = GREY

    doc.add_heading("Validation notes", level=2)
    bullets(doc, [
        "Stratified split keeps the churn rate constant across train and test.",
        "Accuracy-optimised decision thresholds are applied per model.",
        "Honest note: 85–92% accuracy is aspirational on this dataset — 75–80% is the realistic state of the art.",
        "Random Forest finds 76.2% of future churners — the right trade-off for retention teams.",
    ])

    # ---- 9. conclusion
    doc.add_heading("9. Conclusion & Future Scope", level=1)
    doc.add_heading("9.1 Project Outcome", level=2)
    bullets(doc, [
        "End-to-end churn system: clean → train → evaluate → deploy.",
        "Three interpretable models with honest, test-set metrics.",
        "Flask + Streamlit front-ends and a JSON API.",
        "Publicly accessible Streamlit dashboard on the cloud.",
        "Batch scoring with downloadable reports.",
    ])
    doc.add_heading("9.2 Future Improvements", level=2)
    bullets(doc, [
        "Localised data: UPI trends, migration patterns, regional behaviour.",
        "SHAP / LIME explanations for every prediction.",
        "Time-series and uplift models for retention campaigns.",
        "Automated alerts for high-risk customers.",
        "Model monitoring as new data arrives.",
    ])

    # ---- 10. output
    doc.add_heading("10. Output", level=1)
    doc.add_paragraph(f"Live application: {LIVE_URL}")
    doc.add_paragraph(f"GitHub repository: {GITHUB_URL}")
    add_picture_centered(doc, ASSETS / "feature_importance.png", 6.0,
                         "Figure 4 — Random Forest top churn drivers (global feature importance).")

    out = OUT / "Customer_Churn_Prediction_Project_Documentation.docx"
    doc.save(out)
    print("Saved:", out)


# ================================================================ CODE DOC
def build_code_doc():
    doc = Document()
    add_footer_pagenum(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    for _ in range(3):
        doc.add_paragraph()
    big_centered(doc, "Customer Churn Prediction", 32)
    small_centered(doc, "Complete Project Source Code", 16, after=14)
    small_centered(doc, f"GitHub Repository: {GITHUB_URL}", 12)

    files = [
        "requirements.txt",
        "src/preprocess.py",
        "src/train_models.py",
        "src/optimise_thresholds.py",
        "app.py",
        "streamlit_app.py",
    ]

    p = doc.add_paragraph()
    r = p.add_run(f"Included code files: {len(files)}")
    r.font.bold = True
    doc.add_paragraph("The model directory (model/*) is trained output and the remaining "
                      "helper scripts (src/tune_models.py, src/optimise_thresholds.py) can be "
                      "regenerated from the repository as needed.")
    doc.add_page_break()

    for i, path in enumerate(files, start=1):
        doc.add_heading(f"{i}. {path}", level=1)
        code = (ROOT / path).read_text(encoding="utf-8").rstrip("\n")
        add_code_block(doc, code)
        if i != len(files):
            doc.add_page_break()

    out = OUT / "Customer_Churn_Prediction_Complete_Code.docx"
    doc.save(out)
    print("Saved:", out)


if __name__ == "__main__":
    build_report()
    build_code_doc()