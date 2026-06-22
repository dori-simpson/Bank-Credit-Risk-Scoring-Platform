"""
Credit Risk Scoring — Full Streamlit App
=========================================
Run:   streamlit run app.py
Deps: pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, os, joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Risk Scoring",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600&display=swap');

/* ── Bank palette ── */
:root {
    --navy:      #1B3A6B;   /* primary brand – trust, authority  */
    --navy-lt:   #2A529A;   /* hover / accent */
    --navy-pale: #EBF0F9;   /* tinted surface  */
    --gold:      #C9A84C;   /* secondary accent – premium        */
    --danger:    #B91C1C;
    --danger-lt: #FEE2E2;
    --warn:      #B45309;
    --warn-lt:   #FEF3C7;
    --success:   #166534;
    --success-lt:#DCFCE7;
    --bg:        #F4F6FA;   /* page background – cool off-white  */
    --surface:   #FFFFFF;
    --border:    #DDE3EE;
    --txt:       #1A202C;
    --txt-muted: #64748B;
    --radius:    10px;
    --font:      'Inter', system-ui, sans-serif;
}

html, body, [class*="css"] { font-family: var(--font); background: var(--bg); color: var(--txt); }
.stApp { background: var(--bg); }

/* ── Sidebar: deep navy ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #CBD5E8 !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: #7A98C8 !important; font-weight: 600; margin-bottom: 4px;
}
[data-testid="stSidebar"] .stRadio label { color: #CBD5E8 !important; font-size: 0.875rem; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong { color: #E8EDF5 !important; }
[data-testid="stSidebar"] .stCaption { color: #7A98C8 !important; }
[data-testid="stSidebar"] hr { border-color: #2A529A !important; }
[data-testid="stSidebar"] .stFileUploader label { color: #CBD5E8 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--navy);
    border-radius: var(--radius);
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(27,58,107,0.06);
}
[data-testid="metric-container"] label {
    color: var(--txt-muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-size: 1.9rem !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
}
[data-baseweb="tab"] {
    border-radius: 7px !important;
    color: var(--txt-muted) !important;
    font-weight: 500;
    font-size: 0.84rem;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--navy) !important;
    color: #ffffff !important;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--navy); color: #ffffff; border: none;
    border-radius: 7px; font-weight: 600; padding: 10px 24px;
    font-size: 0.875rem; letter-spacing: 0.01em; transition: background .15s;
}
.stButton > button:hover { background: var(--navy-lt) !important; border: none !important; }

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--txt) !important;
    font-size: 0.875rem;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}

/* ── Expanders / alerts ── */
.stAlert { border-radius: var(--radius); }
.stExpander { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }

/* ── Section headers ── */
.section-head {
    font-size: 0.95rem; font-weight: 600; color: var(--navy);
    border-left: 3px solid var(--navy); padding-left: 10px;
    margin: 24px 0 12px; letter-spacing: -0.01em;
}

/* ── Risk badges ── */
.risk-high {
    background: var(--danger-lt); border: 1px solid #FECACA;
    color: var(--danger); border-radius: var(--radius);
    padding: 14px 18px; font-weight: 600; font-size: 1rem;
}
.risk-low {
    background: var(--success-lt); border: 1px solid #BBF7D0;
    color: var(--success); border-radius: var(--radius);
    padding: 14px 18px; font-weight: 600; font-size: 1rem;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: var(--navy) !important; }

/* ── Page header band ── */
.page-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 18px 0 14px;
    margin-bottom: 20px;
}
.page-header h1 { color: var(--navy); font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }
.page-header p  { color: var(--txt-muted); font-size: 0.83rem; margin: 0; }

/* ── Mobile ── */
@media (max-width: 768px) {
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    [data-baseweb="tab"] { font-size: 0.72rem !important; padding: 5px 8px !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS / MAPPINGS  (shared across tabs)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'annual_income', 'debt_to_income', 'inquiries_last_12m',
    'total_credit_lines', 'open_credit_lines', 'total_credit_limit',
    'total_credit_utilized', 'num_collections_last_12m',
    'num_historical_failed_to_pay', 'current_accounts_delinq',
    'accounts_opened_24m', 'num_satisfactory_accounts',
    'num_active_debit_accounts', 'total_debit_limit',
    'num_total_cc_accounts', 'num_open_cc_accounts',
    'num_cc_carrying_balance', 'num_mort_accounts',
    'account_never_delinq_percent', 'tax_liens',
    'public_record_bankrupt', 'loan_amount', 'interest_rate',
    'installment', 'balance', 'paid_total',
    'homeownership', 'verified_income', 'loan_purpose',
    'application_type', 'grade',
]

STATUS_MAP = {
    'Fully Paid': 0, 'Current': 0,
    'Charged Off': 1, 'Late (31-120 days)': 1,
    'Late (16-30 days)': 1, 'In Grace Period': 1, 'Default': 1,
}
HOMEOWNERSHIP_MAP = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
VERIFIED_MAP      = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
GRADE_MAP         = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
GRADE_MAP_INV     = {v: k for k, v in GRADE_MAP.items()}
PURPOSE_LIST      = sorted([
    'car', 'credit_card', 'debt_consolidation', 'educational',
    'home_improvement', 'house', 'major_purchase', 'medical',
    'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding',
])
PURPOSE_MAP = {v: i for i, v in enumerate(PURPOSE_LIST)}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FFFFFF',
    font_color='#1A202C',
    font_family='Inter',
    xaxis=dict(gridcolor='#EEF1F7', zerolinecolor='#DDE3EE', linecolor='#DDE3EE'),
    yaxis=dict(gridcolor='#EEF1F7', zerolinecolor='#DDE3EE', linecolor='#DDE3EE'),
)

# Bank colour sequence: navy → gold → slate blue → muted red → steel
BANK_COLORS = ['#1B3A6B', '#C9A84C', '#4472C4', '#B91C1C', '#7A98C8',
               '#2A529A', '#A38830', '#5B8DD9', '#D94F4F', '#5B7FAA']

def apply_theme(fig):
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=50, b=25),
        title_font_size=14,
        title_font_color='#1B3A6B',
        title_font_family='Inter',
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font_size=11),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, title_font=dict(size=11, color='#64748B'))
    fig.update_yaxes(showgrid=True, gridwidth=1, title_font=dict(size=11, color='#64748B'))
    return fig

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['default'] = df['loan_status'].map(STATUS_MAP)

    num_fill = [
        'interest_rate', 'debt_to_income', 'months_since_last_delinq',
        'months_since_90d_late', 'months_since_last_credit_inquiry',
        'num_accounts_120d_past_due', 'debt_to_income_joint', 'annual_income_joint',
    ]
    for col in num_fill:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    df.dropna(subset=['emp_length', 'default'], inplace=True)

    # Encode categoricals
    df['homeownership']   = df['homeownership'].map(HOMEOWNERSHIP_MAP)
    df['verified_income'] = df['verified_income'].map(VERIFIED_MAP)
    df['grade']           = df['grade'].map(GRADE_MAP)

    purpose_vals = df['loan_purpose'].dropna().unique()
    p_map = {v: i for i, v in enumerate(sorted(purpose_vals))}
    df['loan_purpose'] = df['loan_purpose'].map(p_map)

    df['application_type'] = df['application_type'].str.lower().map({'individual': 0, 'joint': 1})
    return df

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model…")
def train_model(path: str):
    df = load_and_preprocess(path)
    subset = df[FEATURE_COLS + ['default']].dropna()

    X = subset[FEATURE_COLS]
    y = subset['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_s, np.ravel(y_train))

    y_pred       = model.predict(X_test_s)
    y_prob       = model.predict_proba(X_test_s)[:, 1]
    auc          = roc_auc_score(y_test, y_prob)
    ap           = average_precision_score(y_test, y_prob)
    report       = classification_report(y_test, y_pred, target_names=['Non-Default', 'Default'], output_dict=True)
    cm           = confusion_matrix(y_test, y_pred)
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    coef_df = pd.DataFrame({
        'Feature':     FEATURE_COLS,
        'Coefficient': model.coef_[0],
    }).sort_values('Coefficient', ascending=False)

    return dict(
        model=model, scaler=scaler,
        X_test=X_test, y_test=y_test, y_pred=y_pred, y_prob=y_prob,
        auc=auc, ap=ap, report=report, cm=cm,
        fpr=fpr, tpr=tpr, prec=prec, rec=rec,
        coef_df=coef_df, X_train=X_train, df_model=subset,
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ CreditIQ")
    st.markdown("**Credit Risk Scoring Platform**")
    st.divider()

    st.markdown("## Dataset")
    data_source = st.radio("Source", ["Upload CSV", "Use demo data"], index=1)

    uploaded_file = None
    data_path = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("loans_full_schema.csv", type="csv")
        if uploaded_file:
            tmp = "/tmp/loans_uploaded.csv"
            with open(tmp, "wb") as f:
                f.write(uploaded_file.read())
            data_path = tmp
    else:
        DEFAULT_PATH = "loans_full_schema.csv"
        if os.path.exists(DEFAULT_PATH):
            data_path = DEFAULT_PATH
        else:
            data_path = "__DEMO__"

    st.divider()
    st.markdown("## About")
    st.caption("Open-source ML credit risk tool built with Streamlit + scikit-learn. All computation runs locally.")

# ─────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Generating demo data…")
def make_demo_data(n=2000, seed=42) -> str:
    rng = np.random.default_rng(seed)
    grades   = rng.choice(['A','B','C','D','E','F','G'], n, p=[.2,.25,.2,.15,.1,.06,.04])
    grade_n  = np.array([GRADE_MAP[g] for g in grades])
    income   = np.clip(rng.lognormal(11, 0.5, n), 20000, 500000)
    dti      = np.clip(rng.normal(18, 8, n), 0, 80)
    amount   = np.clip(rng.lognormal(9.5, 0.6, n), 1000, 100000)
    rate     = 5 + grade_n * 2.5 + rng.normal(0, 1, n)
    rate     = np.clip(rate, 4, 30)
    install  = amount * rate / 100 / 12
    balance  = amount * rng.uniform(0.3, 1.0, n)
    paid     = amount - balance

    fail_hist = rng.integers(0, 5, n)
    p_default = 0.05 + grade_n * 0.06 + fail_hist * 0.04 + (dti / 200)
    p_default = np.clip(p_default, 0.02, 0.7)
    statuses  = np.where(rng.random(n) < p_default, 'Charged Off', 'Fully Paid')

    df = pd.DataFrame({
        'loan_status':                  statuses,
        'annual_income':                income,
        'debt_to_income':               dti,
        'inquiries_last_12m':           rng.integers(0, 10, n),
        'total_credit_lines':           rng.integers(3, 50, n),
        'open_credit_lines':            rng.integers(1, 20, n),
        'total_credit_limit':           rng.uniform(5000, 200000, n),
        'total_credit_utilized':        rng.uniform(500, 50000, n),
        'num_collections_last_12m':     rng.integers(0, 5, n),
        'num_historical_failed_to_pay': fail_hist,
        'current_accounts_delinq':      rng.integers(0, 4, n),
        'accounts_opened_24m':          rng.integers(0, 10, n),
        'num_satisfactory_accounts':    rng.integers(1, 30, n),
        'num_active_debit_accounts':    rng.integers(0, 10, n),
        'total_debit_limit':            rng.uniform(500, 30000, n),
        'num_total_cc_accounts':        rng.integers(0, 20, n),
        'num_open_cc_accounts':         rng.integers(0, 15, n),
        'num_cc_carrying_balance':      rng.integers(0, 10, n),
        'num_mort_accounts':            rng.integers(0, 5, n),
        'account_never_delinq_percent': rng.uniform(60, 100, n),
        'tax_liens':                    rng.integers(0, 3, n),
        'public_record_bankrupt':       rng.integers(0, 2, n),
        'loan_amount':                  amount,
        'interest_rate':                rate,
        'installment':                  install,
        'balance':                      balance,
        'paid_total':                   paid,
        'homeownership':                rng.choice(['RENT','MORTGAGE','OWN'], n, p=[.4,.45,.15]),
        'verified_income':              rng.choice(['Not Verified','Source Verified','Verified'], n),
        'loan_purpose':                 rng.choice(PURPOSE_LIST, n),
        'application_type':             rng.choice(['individual','joint'], n, p=[.85,.15]),
        'grade':                        grades,
        'emp_length':                   rng.integers(0, 20, n),
    })
    path = "/tmp/demo_loans.csv"
    df.to_csv(path, index=False)
    return path

# ─────────────────────────────────────────────
# RESOLVE DATA PATH
# ─────────────────────────────────────────────
if data_path == "__DEMO__":
    data_path = make_demo_data()
    st.sidebar.info("Demo data generated (2 000 synthetic loans).")

if data_path is None:
    st.title("⬡ CreditIQ — Credit Risk Scoring")
    st.info("Upload a CSV in the sidebar to get started, or switch to **Use demo data**.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────
raw_df = load_and_preprocess(data_path)
results = train_model(data_path)

model   = results['model']
scaler  = results['scaler']

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
  <span style='font-size:2rem'>⬡</span>
  <div>
    <h1 style='margin:0;font-size:1.8rem;font-weight:800;letter-spacing:-0.02em'>CreditIQ</h1>
    <p style='margin:0;color:#8b949e;font-size:0.85rem'>Credit Risk Scoring Platform</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Top-level KPIs
col1, col2, col3, col4 = st.columns(4)
n_loans   = len(raw_df)
n_default = int(raw_df['default'].sum())
def_rate  = n_default / n_loans * 100
col1.metric("Total Loans",    f"{n_loans:,}")
col2.metric("Defaults",       f"{n_default:,}")
col3.metric("Default Rate",   f"{def_rate:.1f}%")
col4.metric("AUC Score",      f"{results['auc']:.3f}")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Explore",
    "🤖 Model",
    "🔍 Predict",
    "📈 Portfolio",
    "📋 Report",
])

# ════════════════════════════════════════════
# TAB 1 — EXPLORE
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Filters", expanded=False):
        fc1, fc2 = st.columns(2)
        inc_range = fc1.slider("Annual Income ($)", 0, 500_000, (0, 300_000), step=5000)
        dti_range = fc2.slider("Debt-to-Income", 0.0, 80.0, (0.0, 60.0), step=0.5)

    raw_num = pd.read_csv(data_path)
    mask = (
        (raw_num['annual_income'].between(*inc_range)) &
        (raw_num['debt_to_income'].between(*dti_range))
    )
    df_eda = raw_num[mask]

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = px.histogram(df_eda, x='loan_amount', nbins=50,
                           color_discrete_sequence=[BANK_COLORS[0]],
                           title='Loan Amount Distribution',
                           labels={'loan_amount': 'Loan Amount ($)', 'count': 'Frequency'})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        if 'grade' in df_eda.columns and 'loan_status' in df_eda.columns:
            grade_df = df_eda.copy()
            grade_df['defaulted'] = grade_df['loan_status'].map(STATUS_MAP).fillna(0)
            gd = grade_df.groupby('grade')['defaulted'].mean().reset_index()
            gd.columns = ['Grade','Default Rate']
            gd = gd[gd['Grade'].isin(['A','B','C','D','E','F','G'])]
            gd = gd.sort_values('Grade')
            fig2 = px.bar(gd, x='Grade', y='Default Rate',
                          color='Default Rate', color_continuous_scale='YlOrRd',
                          title='Default Rate by Loan Grade')
            apply_theme(fig2)
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        sample = df_eda.sample(min(1500, len(df_eda)), random_state=1)
        sample['Status'] = sample['loan_status'].map(lambda s: 'Default' if STATUS_MAP.get(s, 0) == 1 else 'Non-Default')
        fig3 = px.scatter(sample, x='annual_income', y='debt_to_income',
                          color='Status',
                          color_discrete_map={'Default': BANK_COLORS[3], 'Non-Default': BANK_COLORS[0]},
                          opacity=0.6, title='Income vs Debt-to-Income',
                          labels={'annual_income': 'Annual Income ($)', 'debt_to_income': 'DTI Ratio'})
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        if 'grade' in df_eda.columns:
            grade_order = ['A','B','C','D','E','F','G']
            dfg = df_eda[df_eda['grade'].isin(grade_order)]
            fig4 = px.box(dfg, x='grade', y='interest_rate',
                          category_orders={'grade': grade_order},
                          color_discrete_sequence=[BANK_COLORS[4]],
                          title='Interest Rate by Grade',
                          labels={'interest_rate': 'Interest Rate (%)', 'grade': 'Grade'})
            apply_theme(fig4)
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    if 'loan_purpose' in df_eda.columns:
        purpose_counts = df_eda['loan_purpose'].value_counts().reset_index()
        purpose_counts.columns = ['Purpose', 'Count']
        fig5 = px.bar(purpose_counts, x='Count', y='Purpose', orientation='h',
                      color_discrete_sequence=[BANK_COLORS[2]],
                      title='Loans by Purpose')
        apply_theme(fig5)
        fig5.update_layout(height=420)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-head">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_eda.head(200), use_container_width=True, height=300)

# ════════════════════════════════════════════
# TAB 2 — MODEL
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    rep = results['report']
    m1.metric("AUC-ROC",   f"{results['auc']:.4f}")
    m2.metric("Avg Precision", f"{results['ap']:.4f}")
    m3.metric("F1 (Default)", f"{rep['Default']['f1-score']:.4f}")
    m4.metric("Recall (Default)", f"{rep['Default']['recall']:.4f}")

    mc1, mc2 = st.columns(2)

    with mc1:
        fpr, tpr = results['fpr'], results['tpr']
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     line=dict(color=BANK_COLORS[0], width=2.5),
                                     name=f'ROC (AUC={results["auc"]:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     line=dict(color='#8b949e', dash='dash'), name='Random'))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate', **PLOTLY_TEMPLATE,
                              margin=dict(l=20, r=20, t=50, b=25))
        st.plotly_chart(fig_roc, use_container_width=True)

    with mc2:
        prec, rec = results['prec'], results['rec']
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines',
                                    line=dict(color=BANK_COLORS[2], width=2.5),
                                    name=f'PR (AP={results["ap"]:.3f})'))
        fig_pr.update_layout(title='Precision-Recall Curve',
                             xaxis_title='Recall', yaxis_title='Precision',
                             **PLOTLY_TEMPLATE, margin=dict(l=20, r=20, t=50, b=25))
        st.plotly_chart(fig_pr, use_container_width=True)

    mc3, mc4 = st.columns(2)

    with mc3:
        cm = results['cm']
        fig_cm = px.imshow(cm, text_auto=True, aspect='auto',
                           x=['Non-Default','Default'], y=['Non-Default','Default'],
                           color_continuous_scale='Blues',
                           title='Confusion Matrix',
                           labels=dict(x='Predicted', y='Actual'))
        apply_theme(fig_cm)
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with mc4:
        coef = results['coef_df'].copy()
        coef['Color'] = coef['Coefficient'].apply(lambda x: BANK_COLORS[0] if x > 0 else BANK_COLORS[1])
        top = pd.concat([coef.head(10), coef.tail(10)])
        fig_coef = px.bar(top, x='Coefficient', y='Feature', orientation='h',
                          color='Color', color_discrete_map='identity',
                          title='Top Feature Coefficients')
        apply_theme(fig_coef)
        fig_coef.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown('<div class="section-head">Classification Report</div>', unsafe_allow_html=True)
    rep_df = pd.DataFrame(results['report']).T.round(4)
    rep_df = rep_df.drop(index=['accuracy'], errors='ignore')
    st.dataframe(rep_df.style.background_gradient(cmap='Blues', subset=['f1-score','precision','recall']),
                 use_container_width=True)

    st.markdown('<div class="section-head">Default Probability Distribution</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        'Probability': results['y_prob'],
        'Label': results['y_test'].map({0: 'Non-Default', 1: 'Default'}),
    })
    fig_dist = px.histogram(prob_df, x='Probability', color='Label', nbins=60,
                            barmode='overlay', opacity=0.7,
                            color_discrete_map={'Default': BANK_COLORS[3], 'Non-Default': BANK_COLORS[0]},
                            title='Model Score Distribution')
    apply_theme(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3 — PREDICT
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-head">Single Applicant Assessment</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**👤 Applicant**")
            annual_income   = st.number_input("Annual Income ($)", 0.0, 2e6, 60000.0, step=1000.0)
            debt_to_income  = st.number_input("Debt-to-Income Ratio", 0.0, 100.0, 15.0)
            homeownership   = st.selectbox("Home Ownership", list(HOMEOWNERSHIP_MAP.keys()))
            verified_income = st.selectbox("Income Verification", list(VERIFIED_MAP.keys()))
            application_type= st.selectbox("Application Type", ["individual", "joint"])
            num_historical_failed_to_pay = st.number_input("Historical Failed Payments", 0, 50, 0)
            public_record_bankrupt = st.number_input("Bankruptcy Records", 0, 10, 0)
            tax_liens = st.number_input("Tax Liens", 0, 10, 0)

        with p2:
            st.markdown("**💳 Credit Profile**")
            inquiries_last_12m          = st.number_input("Credit Inquiries (12m)", 0, 30, 1)
            open_credit_lines           = st.number_input("Open Credit Lines", 0, 50, 5)
            total_credit_limit          = st.number_input("Total Credit Limit ($)", 0.0, 500000.0, 30000.0, step=1000.0)
            total_credit_utilized       = st.number_input("Total Credit Utilized ($)", 0.0, 200000.0, 8000.0, step=500.0)
            total_credit_lines          = st.number_input("Total Credit Lines", 0, 100, 15)
            num_satisfactory_accounts   = st.number_input("Satisfactory Accounts", 0, 50, 10)
            num_active_debit_accounts   = st.number_input("Active Debit Accounts", 0, 30, 3)
            total_debit_limit           = st.number_input("Total Debit Limit ($)", 0.0, 100000.0, 5000.0, step=500.0)
            num_total_cc_accounts       = st.number_input("Total CC Accounts", 0, 50, 5)
            num_open_cc_accounts        = st.number_input("Open CC Accounts", 0, 30, 3)
            num_cc_carrying_balance     = st.number_input("CCs Carrying Balance", 0, 30, 2)
            num_mort_accounts           = st.number_input("Mortgage Accounts", 0, 20, 1)
            account_never_delinq_percent= st.number_input("% Accounts Never Delinquent", 0.0, 100.0, 90.0)
            num_collections_last_12m    = st.number_input("Collections (12m)", 0, 20, 0)
            current_accounts_delinq     = st.number_input("Current Delinquent Accounts", 0, 20, 0)
            accounts_opened_24m         = st.number_input("Accounts Opened (24m)", 0, 30, 3)

        with p3:
            st.markdown("**🏦 Loan Details**")
            loan_amount   = st.number_input("Loan Amount ($)", 0.0, 500000.0, 15000.0, step=500.0)
            interest_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
            installment   = st.number_input("Monthly Installment ($)", 0.0, 10000.0, 350.0)
            balance       = st.number_input("Current Balance ($)", 0.0, 500000.0, 10000.0, step=500.0)
            paid_total    = st.number_input("Total Paid ($)", 0.0, 500000.0, 5000.0, step=500.0)
            grade         = st.selectbox("Loan Grade", list(GRADE_MAP.keys()))
            loan_purpose  = st.selectbox("Loan Purpose", PURPOSE_LIST)

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        row = [
            annual_income, debt_to_income, inquiries_last_12m,
            total_credit_lines, open_credit_lines, total_credit_limit,
            total_credit_utilized, num_collections_last_12m,
            num_historical_failed_to_pay, current_accounts_delinq,
            accounts_opened_24m, num_satisfactory_accounts,
            num_active_debit_accounts, total_debit_limit,
            num_total_cc_accounts, num_open_cc_accounts,
            num_cc_carrying_balance, num_mort_accounts,
            account_never_delinq_percent, tax_liens,
            public_record_bankrupt, loan_amount, interest_rate,
            installment, balance, paid_total,
            HOMEOWNERSHIP_MAP[homeownership],
            VERIFIED_MAP[verified_income],
            PURPOSE_MAP.get(loan_purpose, 0),
            0 if application_type == 'individual' else 1,
            GRADE_MAP[grade],
        ]
        X_input = scaler.transform([row])
        pred  = model.predict(X_input)[0]
        prob  = model.predict_proba(X_input)[0][1]

        st.divider()
        rc1, rc2, rc3 = st.columns([2, 1, 1])

        with rc1:
            if pred == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH RISK — Likely to Default</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ LOW RISK — Unlikely to Default</div>', unsafe_allow_html=True)
            st.progress(float(prob))
            st.caption(f"Default probability: **{prob:.1%}** |  Decision threshold: 50%")

        with rc2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'color': '#1A202C', 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
                    'bar': {'color': BANK_COLORS[3] if prob > 0.5 else BANK_COLORS[0]},
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(27,58,107,0.05)'},
                        {'range': [30, 60], 'color': 'rgba(201,168,76,0.1)'},
                        {'range': [60, 100],'color': 'rgba(185,28,28,0.05)'},
                    ],
                    'threshold': {'line': {'color': '#1B3A6B', 'width': 2}, 'value': 50},
                },
                title={'text': 'Default Probability', 'font': {'color': '#64748B', 'size': 13}},
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1A202C',
                                    height=250, margin=dict(l=15,r=15,t=40,b=15))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with rc3:
            coef_df = results['coef_df'].copy()
            coef_df['Input'] = row
            coef_df['Impact'] = coef_df['Coefficient'] * coef_df['Input']
            top_drivers = coef_df.nlargest(5, 'Impact')[['Feature','Impact']]
            st.markdown("**Top Risk Drivers**")
            for _, r in top_drivers.iterrows():
                col_color = "🔴" if r.Impact > 0 else "🟢"
                st.markdown(f"{col_color} `{r.Feature}` — {r.Impact:+.3f}")

        train_probs = model.predict_proba(scaler.transform(results['X_train']))[:, 1]
        pctile = int(np.mean(train_probs < prob) * 100)
        st.info(f"This applicant's default probability is higher than **{pctile}%** of the training population.")

# ─────────────────────────────────────────────
# TAB 4 — PORTFOLIO
# ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-head">Portfolio-Level Risk Analysis</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="Scoring portfolio…")
    def score_portfolio(path):
        df = load_and_preprocess(path)
        sub = df[FEATURE_COLS].dropna()
        idx = sub.index
        X_s = scaler.transform(sub)
        probs = model.predict_proba(X_s)[:, 1]
        out = df.loc[idx].copy()
        out['default_prob'] = probs
        out['risk_bucket'] = pd.cut(probs, [0, .2, .4, .6, .8, 1.0],
                                    labels=['Very Low','Low','Medium','High','Very High'])
        return out

    portfolio = score_portfolio(data_path)

    po1, po2 = st.columns(2)

    with po1:
        bucket_counts = portfolio['risk_bucket'].value_counts().reset_index()
        bucket_counts.columns = ['Bucket','Count']
        order = ['Very Low','Low','Medium','High','Very High']
        bucket_counts['Bucket'] = pd.Categorical(bucket_counts['Bucket'], order)
        bucket_counts = bucket_counts.sort_values('Bucket')
        fig_b = px.bar(bucket_counts, x='Bucket', y='Count',
                       color_discrete_sequence=[BANK_COLORS[0]],
                       title='Portfolio Risk Distribution')
        apply_theme(fig_b)
        st.plotly_chart(fig_b, use_container_width=True)

    with po2:
        fig_ph = px.histogram(portfolio, x='default_prob', nbins=60,
                              color_discrete_sequence=[BANK_COLORS[2]],
                              title='Default Probability Histogram',
                              labels={'default_prob': 'Default Probability'})
        apply_theme(fig_ph)
        st.plotly_chart(fig_ph, use_container_width=True)

    po3, po4 = st.columns(2)

    with po3:
        portfolio['expected_loss'] = portfolio['default_prob'] * portfolio['loan_amount']
        el_by_bucket = portfolio.groupby('risk_bucket', observed=True)['expected_loss'].sum().reset_index()
        el_by_bucket.columns = ['Bucket','Expected Loss ($)']
        el_by_bucket['Bucket'] = pd.Categorical(el_by_bucket['Bucket'], order, ordered=True)
        el_by_bucket = el_by_bucket.sort_values('Bucket')
        fig_el = px.bar(el_by_bucket, x='Bucket', y='Expected Loss ($)',
                        color_discrete_sequence=[BANK_COLORS[1]],
                        title='Expected Loss by Risk Bucket')
        apply_theme(fig_el)
        st.plotly_chart(fig_el, use_container_width=True)

    with po4:
        raw_port = pd.read_csv(data_path)
        raw_port = raw_port.loc[portfolio.index] if len(raw_port) == len(portfolio) else raw_port
        if 'loan_purpose' in raw_port.columns:
            raw_port['default_prob'] = portfolio['default_prob'].values[:len(raw_port)]
            purpose_risk = raw_port.groupby('loan_purpose')['default_prob'].mean().reset_index()
            purpose_risk.columns = ['Purpose','Avg Default Prob']
            purpose_risk = purpose_risk.sort_values('Avg Default Prob', ascending=True)
            fig_pur = px.bar(purpose_risk, x='Avg Default Prob', y='Purpose', orientation='h',
                             color_discrete_sequence=[BANK_COLORS[4]],
                             title='Avg Default Probability by Purpose')
            apply_theme(fig_pur)
            fig_pur.update_layout(height=400)
            st.plotly_chart(fig_pur, use_container_width=True)

    st.markdown('<div class="section-head">High-Risk Loans (Prob > 60%)</div>', unsafe_allow_html=True)
    high_risk = portfolio[portfolio['default_prob'] > 0.6][
        ['loan_amount', 'interest_rate', 'annual_income', 'debt_to_income', 'default_prob', 'risk_bucket']
    ].sort_values('default_prob', ascending=False).head(100)
    high_risk['default_prob'] = (high_risk['default_prob'] * 100).round(1).astype(str) + '%'
    st.dataframe(high_risk.reset_index(drop=True), use_container_width=True, height=300)

# ════════════════════════════════════════════
# TAB 5 — REPORT
# ════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-head">Model Report & Documentation</div>', unsafe_allow_html=True)

    rep = results['report']
    rp1, rp2, rp3 = st.columns(3)
    rp1.metric("Precision (Default)", f"{rep['Default']['precision']:.3f}")
    rp2.metric("Recall (Default)",    f"{rep['Default']['recall']:.3f}")
    rp3.metric("F1 (Default)",        f"{rep['Default']['f1-score']:.3f}")

    st.markdown("""
---
### Model Summary

| Item | Detail |
|---|---|
| **Algorithm** | Logistic Regression (lbfgs solver) |
| **Features** | 31 numerical + encoded categorical |
| **Train / Test Split** | 70% / 30% (stratified) |
| **Class Weighting** | Balanced (handles imbalance) |
| **Scaling** | StandardScaler |
| **Target** | Binary — 1 = Default, 0 = Non-Default |

### Feature Engineering

- `homeownership` — ordinal encoded (RENT=0, MORTGAGE=1, OWN=2, OTHER=3)
- `verified_income` — ordinal encoded (Not Verified=0, Source Verified=1, Verified=2)
- `grade` — ordinal encoded A→G (0–6)
- `loan_purpose` — label encoded (alphabetical sort)
- `application_type` — binary (individual=0, joint=1)
- Missing numericals filled with column mean
- Rows with missing `emp_length` or target are dropped

### Interpreting Default Probability

| Score | Risk Level | Recommended Action |
|---|---|---|
| 0 – 20% | Very Low | Approve |
| 20 – 40% | Low | Approve with monitoring |
| 40 – 60% | Medium | Manual review |
| 60 – 80% | High | Enhanced due diligence |
| 80 – 100% | Very High | Decline or collateral required |

### Limitations

- Logistic Regression assumes linear decision boundary; non-linear patterns may be missed.
- Model performance depends on data quality and representativeness.
- This is a prototype — production deployment requires regulatory review, fairness auditing, and ongoing monitoring.
""")

    st.markdown('<div class="section-head">Full Feature Coefficient Table</div>', unsafe_allow_html=True)
    coef_display = results['coef_df'].copy()
    coef_display['Direction'] = coef_display['Coefficient'].apply(
        lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Reduces Risk'
    )
    st.dataframe(coef_display.style.background_gradient(cmap='RdYlGn_r', subset=['Coefficient']),
                 use_container_width=True, height=600)

    st.divider()
    rep_df_dl = pd.DataFrame(results['report']).T.round(4)
    csv_bytes = rep_df_dl.to_csv().encode('utf-8')
    st.download_button(
        "Download Classification Report CSV",
        data=csv_bytes,
        file_name='creditiq_report.csv',
        mime='text/csv',
    )
