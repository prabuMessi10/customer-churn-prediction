"""Build the Customer Churn Prediction slide-deck documentation (.pptx)."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "documentation" / "assets"
OUT = ROOT / "documentation" / "Customer_Churn_Prediction_Project_Documentation.pptx"

# ---------------------------------------------------------------- palette
BG = RGBColor(0x0F, 0x17, 0x2A)
BG2 = RGBColor(0x1E, 0x29, 0x3B)
PRIMARY = RGBColor(0x63, 0x66, 0xF1)
PRIMARY_D = RGBColor(0x4F, 0x46, 0xE5)
WHITE = RGBColor(0xE2, 0xE8, 0xF0)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
GREEN = RGBColor(0x10, 0xB9, 0x81)
RED = RGBColor(0xEF, 0x44, 0x44)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
SOFT = RGBColor(0x33, 0x41, 0x55)

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sp = s.shapes.add_shape(shape, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    if radius is not None:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    sp.shadow.inherit = False
    return sp


def txt(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=None):
    """lines = list of (text, size, bold, color, space_after_pt)"""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    for i, (t, size, bold, color, after) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        r = p.add_run()
        r.text = t
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = "Segoe UI"
    return tb


def bg(s):
    rect(s, 0, 0, SW, SH, BG)


def header(s, title, sub=None):
    rect(s, 0, 0, SW, Inches(1.25), BG2)
    rect(s, 0, Inches(1.25), SW, Pt(3), PRIMARY)
    txt(s, Inches(0.55), Inches(0.22), Inches(12), Inches(0.9),
        [(title, 30, True, WHITE, 0)], anchor=None)
    if sub:
        txt(s, Inches(0.55), Inches(0.78), Inches(12), Inches(0.5),
            [(sub, 13, False, MUTED, 0)])


def bullets(s, x, y, w, h, items, size=14, color=WHITE, gap=10, marker="•  "):
    lines = [(marker + it, size, False, color, gap) for it in items]
    txt(s, x, y, w, h, lines)


def footer(s, num):
    txt(s, Inches(12.4), Inches(7.08), Inches(0.75), Inches(0.35),
        [(str(num), 12, True, MUTED, 0)], align=PP_ALIGN.RIGHT)


# ================================================================= SLIDE 1
s = slide()
bg(s)
rect(s, 0, 0, SW, Inches(0.18), PRIMARY)
rect(s, Inches(4.16), Inches(0.6), Inches(5), Inches(5), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
txt(s, Inches(4.16), Inches(1.15), Inches(5), Inches(1.1),
    [("Customer Churn Prediction", 30, True, WHITE, 0)], align=PP_ALIGN.CENTER)
txt(s, Inches(4.16), Inches(2.15), Inches(5), Inches(0.9),
    [("System", 30, True, WHITE, 0)], align=PP_ALIGN.CENTER)
txt(s, Inches(4.16), Inches(3.05), Inches(5), Inches(0.6),
    [("Linear Regression  •  Random Forest  •  XGBoost", 13, False, MUTED, 0)], align=PP_ALIGN.CENTER)
txt(s, Inches(4.16), Inches(3.75), Inches(5), Inches(0.5),
    [("Live demo:  customer-churn-prediction-on.streamlit.app", 12, False, AMBER, 0)], align=PP_ALIGN.CENTER)

rect(s, Inches(3.9), Inches(4.55), Inches(5.5), Inches(2.4), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
txt(s, Inches(4.0), Inches(4.75), Inches(5.3), Inches(0.5),
    [("PRESENTED BY", 16, True, PRIMARY, 0)], align=PP_ALIGN.CENTER)
names = [
    ("SUBASH ACHARYA D", "Roll No. 24CSR306"),
    ("THANNASI PRABU R", "Roll No. 24CSR321"),
    ("VIKRAM S", "Roll No. 24CSR345"),
]
y = 5.25
for nm, rn in names:
    txt(s, Inches(4.0), Inches(y), Inches(5.3), Inches(0.5),
        [(nm, 15, True, WHITE, 0), (rn, 11, False, MUTED, 4)], align=PP_ALIGN.CENTER)
    y += 0.55
footer(s, 1)

# ================================================================= SLIDE 2
s = slide()
bg(s)
header(s, "Problem Statement", "Why churn prediction matters")

rect(s, Inches(0.55), Inches(1.7), Inches(5.95), Inches(5.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
txt(s, Inches(0.85), Inches(1.95), Inches(5.3), Inches(0.5), [("Challenges", 20, True, RED, 0)])
bullets(s, Inches(0.85), Inches(2.55), Inches(5.4), Inches(4.0), [
    "Annual losses exceed ₹10,000 crore across telecom, banking and OTT due to churn.",
    "Rule-based retention systems miss complex behavioural patterns (usage shifts, payment irregularities, service interactions).",
    "Generic global datasets ignore localised drivers such as UPI usage trends and regional behaviour.",
    "Businesses react too late — churn signals are not detected early enough.",
    "No production-ready, interpretable models at 85–92% accuracy.",
], size=13.5, gap=12)

rect(s, Inches(6.85), Inches(1.7), Inches(5.95), Inches(5.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
txt(s, Inches(7.15), Inches(1.95), Inches(5.3), Inches(0.5), [("Proposed Solution", 20, True, GREEN, 0)])
bullets(s, Inches(7.15), Inches(2.55), Inches(5.4), Inches(4.0), [
    "Train Linear Regression, Random Forest and XGBoost on Telco customer churn data.",
    "Detect churn early using churn probability + risk levels for every customer.",
    "Explain every prediction (local drivers + global importance).",
    "Serve insights through a Flask web app and a deployed Streamlit dashboard.",
    "Deliver reproducible, interpretable ML with honest, test-set-validated metrics.",
], size=13.5, gap=12)
footer(s, 2)

# ================================================================= SLIDE 3
s = slide()
bg(s)
header(s, "Objectives")
obj = [
    ("01", "Churn Prediction", "Score every customer with a churn probability and a clear Churn / No-Churn verdict."),
    ("02", "Model Comparison", "Build and evaluate Linear Regression, Random Forest and XGBoost side by side."),
    ("03", "Interpretability", "Explain why each model makes its call — local drivers and global importance."),
    ("04", "Data Preparation", "Clean 7,032 records, encode features, standardise numeric fields and engineer new ones."),
    ("05", "Web Application", "Ship a full UI — dashboard, prediction form, batch CSV and JSON API."),
    ("06", "Online Deployment", "Host the app publicly on Streamlit Community Cloud from the GitHub repo."),
]
positions = [(0.55, 1.7), (4.66, 1.7), (8.78, 1.7), (0.55, 4.35), (4.66, 4.35), (8.78, 4.35)]
for (num, title, desc), (x, y) in zip(obj, positions):
    rect(s, x, y, Inches(4.0), Inches(2.45), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.09)
    rect(s, x + Inches(0.25), y + Inches(0.25), Inches(0.95), Inches(0.95), PRIMARY, shape=MSO_SHAPE.OVAL)
    txt(s, x + Inches(0.25), y + Inches(0.42), Inches(0.95), Inches(0.6),
        [(num, 18, True, WHITE, 0)], align=PP_ALIGN.CENTER)
    txt(s, x + Inches(1.35), y + Inches(0.32), Inches(2.5), Inches(0.75),
        [(title, 15, True, WHITE, 0)])
    txt(s, x + Inches(0.3), y + Inches(1.35), Inches(3.45), Inches(1.0),
        [(desc, 12.5, False, MUTED, 0)])
footer(s, 3)

# ================================================================= SLIDE 4
s = slide()
bg(s)
header(s, "Dataset Information", "Telco Customer Churn — IBM sample dataset")

stats = [
    ("7,032", "Records after cleaning"),
    ("20", "Features"),
    ("26.6%", "Churn rate"),
    ("1,407", "Hold-out test customers"),
]
for i, (num, lab) in enumerate(stats):
    x = Inches(0.55 + i * 3.12)
    rect(s, x, Inches(1.55), Inches(2.9), Inches(1.85), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    txt(s, x, Inches(1.8), Inches(2.9), Inches(0.9),
        [(num, 32, True, PRIMARY, 0)], align=PP_ALIGN.CENTER)
    txt(s, x, Inches(2.75), Inches(2.9), Inches(0.5),
        [(lab, 13, False, MUTED, 0)], align=PP_ALIGN.CENTER)

rect(s, Inches(0.55), Inches(3.75), Inches(6.1), Inches(3.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
txt(s, Inches(0.85), Inches(3.95), Inches(5.5), Inches(0.5), [("Core dataset fields", 16, True, WHITE, 0)])
bullets(s, Inches(0.85), Inches(4.55), Inches(5.6), Inches(2.1), [
    "Demographics: gender, SeniorCitizen, Partner, Dependents",
    "Usage: tenure, PhoneService, MultipleLines, InternetService",
    "Services: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Streaming",
    "Contract & billing: Contract, PaperlessBilling, PaymentMethod, Monthly / Total Charges",
], size=12.5, gap=8)

rect(s, Inches(6.95), Inches(3.75), Inches(5.85), Inches(3.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
txt(s, Inches(7.25), Inches(3.95), Inches(5.3), Inches(0.5), [("Preprocessing in src/preprocess.py", 16, True, WHITE, 0)])
bullets(s, Inches(7.25), Inches(4.55), Inches(5.4), Inches(2.1), [
    "Drop customerID; convert TotalCharges to numeric",
    "Map binary fields (gender, Partner, Dependents, services) to 0/1",
    "One-hot encode category fields; standardise numeric fields",
    "Engineer avg_charge_per_month and tenure_log",
    "Target: Churn → Yes/No (26.6% positive class)",
], size=12.5, gap=8)
footer(s, 4)

# ================================================================= SLIDE 5
s = slide()
bg(s)
header(s, "Model Choosing")
models = [
    ("Linear Regression", "Baseline", PRIMARY, [
        "Continuous score thresholded at 0.5",
        "Strong ROC-AUC 0.837",
        "Coefficient-weighted local explanations",
        "79.8% accuracy on the test set",
    ]),
    ("Random Forest", "Retention champion", GREEN, [
        "400 trees, max depth 12, balanced classes",
        "Best churn recall — finds 76.2% of churners",
        "Probability calibrated with threshold 0.70",
        "Recommended for retention teams",
    ]),
    ("XGBoost", "Gradient boosting", AMBER, [
        "500 rounds, learning rate 0.05",
        "Mild imbalance weight 1.4",
        "Clean probability estimates",
        "77.2% accuracy, AUC 0.817",
    ]),
]
for i, (name, tag, color, items) in enumerate(models):
    x = Inches(0.55 + i * 4.17)
    rect(s, x, Inches(1.6), Inches(3.95), Inches(5.1), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
    rect(s, x, Inches(1.6), Inches(3.95), Inches(0.12), color)
    txt(s, x + Inches(0.3), Inches(1.9), Inches(3.4), Inches(0.6),
        [(name, 19, True, WHITE, 0)])
    rect(s, x + Inches(0.32), Inches(2.55), Inches(2.6), Inches(0.42), SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    txt(s, x + Inches(0.42), Inches(2.6), Inches(2.4), Inches(0.32),
        [(tag, 12, True, color, 0)])
    bullets(s, x + Inches(0.3), Inches(3.25), Inches(3.4), Inches(3.2),
            items, size=12.5, gap=14)
footer(s, 5)

# ================================================================= SLIDE 6
s = slide()
bg(s)
header(s, "System Architecture")
nodes = [
    ("CSV Dataset", "7,043 × 21\nraw records"),
    ("Data Cleaning", "types • missing • encoding\nfeature engineering"),
    ("ML Models", "Linear Regression\nRandom Forest • XGBoost"),
    ("Artifacts", "model/*.pkl\nmetrics • thresholds JSON"),
    ("Apps", "Flask web app\nStreamlit dashboard"),
    ("User", "predict a customer\nupload CSV • view charts"),
]
positions = [
    (Inches(0.4), Inches(2.4)), (Inches(2.55), Inches(2.4)), (Inches(4.7), Inches(2.4)),
    (Inches(6.85), Inches(2.4)), (Inches(9.0), Inches(2.4)), (Inches(11.15), Inches(2.4)),
]
for (title, sub), (x, y) in zip(nodes, positions):
    rect(s, x, y, Inches(1.95), Inches(2.6), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    rect(s, x, y, Inches(1.95), Inches(0.14), PRIMARY)
    txt(s, x + Inches(0.12), y + Inches(0.4), Inches(1.7), Inches(0.7),
        [(title, 13.5, True, WHITE, 0)], align=PP_ALIGN.CENTER)
    lines = sub.split("\n")
    txt(s, x + Inches(0.12), y + Inches(1.15), Inches(1.7), Inches(1.3),
        [(ln, 10.5, False, MUTED, 2) for ln in lines], align=PP_ALIGN.CENTER)
for x in positions[:-1]:
    ax = x[0] + Inches(1.95)
    ay = x[1] + Inches(1.3)
    ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, ax + Inches(0.02), ay, Inches(0.56), Inches(0.5))
    ar.fill.solid(); ar.fill.fore_color.rgb = PRIMARY
    ar.line.fill.background(); ar.shadow.inherit = False
txt(s, Inches(0.4), Inches(5.6), Inches(12.5), Inches(0.6),
    [("User selects a customer / uploads data  →  all three models score it  →  interpretations and visual analytics", 14, False, AMBER, 0)],
    align=PP_ALIGN.CENTER)
footer(s, 6)

# ================================================================= SLIDE 7
s = slide()
bg(s)
header(s, "Workflow")
steps = [
    ("1", "Load Data", "Read Telco_Customer_Churn.csv"),
    ("2", "Clean Data", "Convert types • drop NA • encode • engineer features"),
    ("3", "Train Models", "Linear Regression • Random Forest • XGBoost"),
    ("4", "Save Artifacts", "Full pipelines, metrics, thresholds via joblib/JSON"),
    ("5", "Deploy", "Flask locally + public Streamlit Cloud app"),
]
for i, (num, title, desc) in enumerate(steps):
    y = Inches(1.85 + i * 0.98)
    rect(s, Inches(0.55), y, Inches(0.9), Inches(0.9), PRIMARY, shape=MSO_SHAPE.OVAL)
    txt(s, Inches(0.55), y + Inches(0.2), Inches(0.9), Inches(0.5),
        [(num, 20, True, WHITE, 0)], align=PP_ALIGN.CENTER)
    txt(s, Inches(1.75), y + Inches(0.02), Inches(3.2), Inches(0.5),
        [(title, 17, True, WHITE, 0)])
    txt(s, Inches(5.1), y + Inches(0.1), Inches(7.6), Inches(0.7),
        [(desc, 13.5, False, MUTED, 0)])

rect(s, Inches(0.55), Inches(5.55), Inches(5.95), Inches(1.25), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
txt(s, Inches(0.85), Inches(5.7), Inches(5.4), Inches(0.4), [("Inputs", 13, True, PRIMARY, 0)])
txt(s, Inches(0.85), Inches(6.1), Inches(5.4), Inches(0.6),
    [("Customer profile / CSV upload — predictions, probabilities, risk levels", 12, False, MUTED, 0)])
rect(s, Inches(6.85), Inches(5.55), Inches(5.95), Inches(1.25), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
txt(s, Inches(7.15), Inches(5.7), Inches(5.4), Inches(0.4), [("Outputs", 13, True, GREEN, 0)])
txt(s, Inches(7.15), Inches(6.1), Inches(5.4), Inches(0.6),
    [("Verdicts + churn probability, interpretation, CSV downloads, live charts", 12, False, MUTED, 0)])
footer(s, 7)

# ================================================================= SLIDE 8
s = slide()
bg(s)
header(s, "Application Modules")
apps = [
    ("Flask Web App  (app.py)", PRIMARY, [
        "Model dashboard — metrics & confusion matrices",
        "Single prediction — 3-model verdicts + explanation",
        "Batch CSV upload with ensemble verdict",
        "JSON API — POST /api/predict",
    ]),
    ("Streamlit App  (streamlit_app.py)", GREEN, [
        "Model dashboard — KPIs, bars, confusion matrices",
        "Predict a customer — form + live explanation",
        "Batch upload — downloadable results",
        "About — methodology and honest performance notes",
    ]),
]
for i, (title, color, items) in enumerate(apps):
    x = Inches(1.2 + i * 6.1)
    rect(s, x, Inches(1.7), Inches(5.9), Inches(4.6), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
    rect(s, x, Inches(1.7), Inches(5.9), Inches(0.14), color)
    txt(s, x + Inches(0.35), Inches(2.0), Inches(5.2), Inches(0.6),
        [(title, 18, True, WHITE, 0)])
    bullets(s, x + Inches(0.4), Inches(2.85), Inches(5.1), Inches(3.2),
            items, size=14, gap=16)
txt(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.5),
    [("Both apps share the same trained pipelines and preprocessing — one codebase, two front-ends.", 13, False, AMBER, 0)],
    align=PP_ALIGN.CENTER)
footer(s, 8)

# ================================================================= SLIDE 9
s = slide()
bg(s)
header(s, "Model Performance & Validation", "80/20 stratified train-test split • 1,407 test customers • all metrics test-set only")

big = [
    ("79.8%", "Linear Regression accuracy", PRIMARY),
    ("76.2%", "Random Forest churn recall", GREEN),
    ("0.837", "Best ROC-AUC (Linear Regression)", AMBER),
    ("73.4%", "Majority-class baseline", RED),
]
for i, (num, lab, color) in enumerate(big):
    x = Inches(0.55 + i * 3.12)
    rect(s, x, Inches(1.7), Inches(2.9), Inches(1.7), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    txt(s, x, Inches(1.95), Inches(2.9), Inches(0.85),
        [(num, 30, True, color, 0)], align=PP_ALIGN.CENTER)
    txt(s, x + Inches(0.1), Inches(2.9), Inches(2.7), Inches(0.45),
        [(lab, 12, False, MUTED, 0)], align=PP_ALIGN.CENTER)

rect(s, Inches(0.55), Inches(3.7), Inches(7.0), Inches(3.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
s.shapes.add_picture(str(ASSETS / "accuracy_comparison.png"), Inches(0.85), Inches(3.85), height=Inches(2.7))

rect(s, Inches(7.9), Inches(3.7), Inches(4.9), Inches(3.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
txt(s, Inches(8.15), Inches(3.9), Inches(4.4), Inches(0.5), [("Validation notes", 15, True, WHITE, 0)])
bullets(s, Inches(8.15), Inches(4.5), Inches(4.45), Inches(2.1), [
    "Stratified split keeps churn rate constant",
    "Accuracy-optimised decision thresholds per model",
    "Honest note: 85–92% is aspirational on this dataset — 75–80% is the realistic state of the art",
    "Random Forest finds 76% of future churners",
], size=12, gap=10)
footer(s, 9)

# ================================================================= SLIDE 10
s = slide()
bg(s)
header(s, "Conclusion & Future Scope")
rect(s, Inches(0.55), Inches(1.7), Inches(5.95), Inches(5.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
txt(s, Inches(0.85), Inches(1.95), Inches(5.3), Inches(0.5), [("Project Outcome", 18, True, GREEN, 0)])
bullets(s, Inches(0.85), Inches(2.6), Inches(5.4), Inches(3.9), [
    "End-to-end churn system: clean → train → evaluate → deploy",
    "Three interpretable models with honest, test-set metrics",
    "Flask + Streamlit front-ends and a JSON API",
    "Publicly accessible Streamlit dashboard on the cloud",
    "Batch scoring with downloadable reports",
], size=13.5, gap=12)

rect(s, Inches(6.85), Inches(1.7), Inches(5.95), Inches(5.0), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
txt(s, Inches(7.15), Inches(1.95), Inches(5.3), Inches(0.5), [("Future Improvements", 18, True, AMBER, 0)])
bullets(s, Inches(7.15), Inches(2.6), Inches(5.4), Inches(3.9), [
    "Localised data: UPI trends, migration patterns, regional behaviour",
    "SHAP / LIME explanations for every prediction",
    "Time-series and uplift models for retention campaigns",
    "Automated alerts for high-risk customers",
    "Model monitoring as new data arrives",
], size=13.5, gap=12)
footer(s, 10)

# ================================================================= SLIDE 11
s = slide()
bg(s)
header(s, "Output — Dashboard & Predictions", "Live app: https://customer-churn-prediction-on.streamlit.app")
rect(s, Inches(0.55), Inches(1.6), Inches(6.0), Inches(5.1), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
s.shapes.add_picture(str(ASSETS / "churn_distribution.png"), Inches(1.1), Inches(2.2), height=Inches(3.9))
rect(s, Inches(6.85), Inches(1.6), Inches(5.95), Inches(5.1), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
s.shapes.add_picture(str(ASSETS / "feature_importance.png"), Inches(7.2), Inches(1.8), height=Inches(4.7))
txt(s, Inches(0.55), Inches(6.85), Inches(12.2), Inches(0.5),
    [("Every customer gets a churn probability, risk level, consensus verdict and an explanation of the deciding features.",
      13, False, AMBER, 0)], align=PP_ALIGN.CENTER)
footer(s, 11)

# ================================================================= SLIDE 12
s = slide()
bg(s)
header(s, "Output — Confusion Matrices (hold-out test set)")
for i, (name, img) in enumerate([
    ("Linear Regression", "confusion_linear_regression.png"),
    ("Random Forest", "confusion_random_forest.png"),
    ("XGBoost", "confusion_xgboost.png"),
]):
    x = Inches(0.4 + i * 4.25)
    rect(s, x, Inches(1.55), Inches(4.05), Inches(4.9), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    txt(s, x, Inches(1.7), Inches(4.05), Inches(0.5),
        [(name, 16, True, WHITE, 0)], align=PP_ALIGN.CENTER)
    s.shapes.add_picture(str(ASSETS / img), x + Inches(0.35), Inches(2.35), height=Inches(3.9))
txt(s, Inches(0.55), Inches(6.7), Inches(12.2), Inches(0.5),
    [("Random Forest trades a little accuracy for far higher churn recall — the right trade for retention teams.",
      13, False, AMBER, 0)], align=PP_ALIGN.CENTER)
footer(s, 12)

# ================================================================= SLIDE 13
s = slide()
bg(s)
header(s, "Output — Model Comparison Dashboard", "Full interactive analytics on the deployed Streamlit app")
rect(s, Inches(0.55), Inches(1.6), Inches(12.25), Inches(4.6), BG2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
s.shapes.add_picture(str(ASSETS / "accuracy_comparison.png"), Inches(1.1), Inches(1.85), height=Inches(4.1))
txt(s, Inches(0.55), Inches(6.5), Inches(12.2), Inches(0.6),
    [("Accuracy, precision, recall, F1, ROC-AUC and confusion matrices for all three models, side by side.",
      13, False, AMBER, 0)], align=PP_ALIGN.CENTER)
footer(s, 13)

prs.save(OUT)
print("Saved:", OUT)