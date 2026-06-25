# streamlit_app.py
# Run: python -m streamlit run streamlit_app.py

import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Student Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── Modern Premium CSS ────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    header {visibility:hidden;} [data-testid="stSidebar"] {display:none;}
    .stDeployButton {display:none;}

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Sora:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Aurora-style animated gradient background */
    .stApp {
        background: linear-gradient(160deg, #0b1224 0%, #131c33 35%, #0d1526 100%);
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: -20%; left: -10%;
        width: 70%; height: 70%;
        background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
        filter: blur(60px);
        pointer-events: none;
        z-index: 0;
        animation: drift 18s ease-in-out infinite alternate;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -20%; right: -10%;
        width: 70%; height: 70%;
        background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
        filter: blur(60px);
        pointer-events: none;
        z-index: 0;
        animation: drift2 22s ease-in-out infinite alternate;
    }
    @keyframes drift { 0% { transform: translate(0,0); } 100% { transform: translate(40px,30px); } }
    @keyframes drift2 { 0% { transform: translate(0,0); } 100% { transform: translate(-40px,-20px); } }

    .block-container { position: relative; z-index: 1; }

    /* Hero */
    .hero { text-align: center; padding: 36px 0 28px 0; }
    .hero-icon {
        font-size: 4rem; margin-bottom: 10px; display: inline-block;
        filter: drop-shadow(0 0 20px rgba(99,102,241,0.5));
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
        0%,100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(-2deg); }
    }
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #a78bfa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px; letter-spacing: -1px;
    }
    .hero-sub { color: #94a3b8; font-size: 1.05rem; font-weight: 500; }
    .hero-badge {
        display:inline-flex; align-items:center; gap:6px;
        background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
        color:#a5b4fc; padding:6px 16px; border-radius:30px;
        font-size:0.78rem; font-weight:700; letter-spacing:0.5px; margin-bottom:18px;
        text-transform:uppercase;
    }

    /* Glass Section Cards */
    .section-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 26px 26px 8px 26px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .section-head {
        font-family: 'Sora', sans-serif;
        font-size: 1.05rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 20px;
        display: flex; align-items: center; gap: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 14px;
    }

    /* Inputs */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 0.9rem !important;
        padding: 4px 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.18) !important;
    }
    div[data-baseweb="select"] span { color: #f1f5f9 !important; }
    label { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 0.85rem !important; margin-bottom: 6px !important; }
    ul[role="listbox"] { background: #1e293b !important; }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #38bdf8) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-top: 8px !important;
        letter-spacing: 0.3px;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 10px 30px -8px rgba(99,102,241,0.6) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 16px 34px -8px rgba(99,102,241,0.7) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* Result Banners */
    .result-low, .result-medium, .result-high {
        border-radius: 22px; padding: 34px; text-align: center;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-low    { background: rgba(16,185,129,0.08); border-top: 3px solid #34d399; }
    .result-medium { background: rgba(245,158,11,0.08); border-top: 3px solid #fbbf24; }
    .result-high   { background: rgba(239,68,68,0.08); border-top: 3px solid #f87171; }

    /* Probability Pills */
    .prob-row { display: flex; gap: 14px; margin: 18px 0; flex-wrap: wrap; }
    .prob-pill {
        flex: 1; min-width: 130px; padding: 20px 14px;
        border-radius: 16px; text-align: center;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.2s;
    }
    .prob-pill:hover { transform: translateY(-3px); }
    .pill-low    { border-bottom: 3px solid #34d399; }
    .pill-medium { border-bottom: 3px solid #fbbf24; }
    .pill-high   { border-bottom: 3px solid #f87171; }

    .pill-val { font-family:'Sora',sans-serif; font-size: 1.5rem; font-weight: 800; }
    .pill-lbl { font-size: 0.8rem; color: #94a3b8; font-weight: 600; margin-top: 4px; }
    .pill-low    .pill-val { color: #34d399; }
    .pill-medium .pill-val { color: #fbbf24; }
    .pill-high   .pill-val { color: #f87171; }

    /* Insight box */
    .insight-box {
        border-radius: 16px; padding: 20px; margin: 22px 0;
        font-size: 0.95rem; line-height: 1.65;
        backdrop-filter: blur(10px);
    }
    .insight-low    { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.2); color: #6ee7b7; }
    .insight-medium { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2); color: #fcd34d; }
    .insight-high   { background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2); color: #fca5a5; }
    .insight-box b { color: #f1f5f9; }

    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        padding: 10px 16px !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(255,255,255,0.02) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        color: #cbd5e1 !important;
    }
    [data-testid="stExpander"] p { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE,'mental_health_model.pkl'),'rb') as f:
        m = pickle.load(f)
    with open(os.path.join(BASE,'scaler.pkl'),'rb') as f:
        s = pickle.load(f)
    with open(os.path.join(BASE,'feature_cols.pkl'),'rb') as f:
        c = pickle.load(f)
    return m, s, c

try:
    model, scaler, feature_cols = load_model()
    loaded = True
except:
    loaded = False

# ── Recovery Plans Config ─────────────────
LOW_STEPS = [
    ("🌙 Fix sleep schedule", "Go to bed and wake up at the same time daily — ensure 7-8 hours of sleep."),
    ("🏃 Daily 20-min walk or exercise", "An outdoor walk helps naturally boost your mood."),
    ("📖 Start a gratitude journal", "Write down 3 positive things every night — this helps reduce anxiety."),
    ("👥 Make weekly time for friends", "Social connection is one of the strongest protective factors for mental health."),
    ("🎨 30 mins daily for your favorite hobby", "It is crucial to engage in activities that bring you genuine happiness."),
]

LOW_EXTRA = [
    ("🧘 5-min morning breathing exercise", "Start your day calmly to prevent stress from building up."),
    ("📵 Avoid phone usage 1 hour before sleep", "Blue light disrupts and lowers the quality of your sleep."),
]

MEDIUM_STEPS = [
    ("⏰ Establish a fixed daily routine", "Fix your times for sleeping, waking, and eating — uncertainty heightens stress."),
    ("🧘 10-min mindfulness or daily meditation", "A highly proven technique to regulate and manage anxiety."),
    ("✍️ Maintain a stress journal — write your feelings", "Put down whatever is on your mind onto paper — it will make you feel lighter."),
    ("🏫 Visit the university counseling center", "Speaking with a professional is a sign of wisdom, not weakness."),
    ("☕ Cut down caffeine and energy drinks", "These substances trigger anxiety — substitute them with water or herbal tea."),
]

MEDIUM_EXTRA = [
    ("🥗 Eat three balanced meals — do not skip", "Sudden drops in blood sugar levels directly impact your mood and anxiety."),
    ("📅 Plan a weekly study schedule", "Academic pressure can be easily managed through proper planning."),
]

HIGH_STEPS = [
    ("🆘 Talk to a trusted person today", "Reach out to a friend, family member, or teacher — do not isolate yourself. Talking is the first step."),
    ("💊 Consult a professional therapist or psychiatrist", "Just like physical health, mental health requires medical care — there is no shame in seeking help."),
    ("🏥 Register at your university health center", "Free or highly affordable counseling options are available — register today."),
    ("📵 Take a 1-week break from social media", "Constant comparison and negative content severely damage mental health."),
    ("🛌 Make sleep your absolute priority", "Sleep 8 hours a night — sleep deprivation is one of the biggest drivers of High Risk."),
]

HIGH_EXTRA = [
    ("🌿 Get 15 mins of sunlight daily", "Natural sunlight increases serotonin production — a natural way to combat depression."),
    ("🚨 Save a crisis helpline in your phone", "Pakistan: Umang helpline 0317-4288665 — available 24/7."),
]

def get_steps_for_risk(pred, low_pct, med_pct, high_pct):
    if pred == 0:
        steps = LOW_STEPS.copy()
        if med_pct >= 30: steps += LOW_EXTRA
        return steps, "low"
    elif pred == 1:
        steps = MEDIUM_STEPS.copy()
        if high_pct >= 25: steps += MEDIUM_EXTRA
        return steps, "medium"
    else:
        steps = HIGH_STEPS.copy()
        if high_pct >= 70: steps += HIGH_EXTRA
        return steps, "high"

def render_steps(steps, risk_type):
    border_colors = {"low": "#34d399", "medium": "#fbbf24", "high": "#f87171"}
    bg_colors     = {"low": "rgba(16,185,129,0.05)", "medium": "rgba(245,158,11,0.05)", "high": "rgba(239,68,68,0.05)"}
    title_colors  = {"low": "#6ee7b7", "medium": "#fcd34d", "high": "#fca5a5"}
    desc_colors   = {"low": "#cbd5e1", "medium": "#cbd5e1", "high": "#cbd5e1"}

    bc = border_colors[risk_type]
    bg = bg_colors[risk_type]
    tc = title_colors[risk_type]
    dc = desc_colors[risk_type]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {bc}; border-radius:14px; padding:16px 18px; margin:12px 0; display:flex; gap:14px; align-items: flex-start; border-top: 1px solid rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04);">
            <div style="background:{bc}; color:#0b1224; border-radius:50%; min-width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:800; margin-top:2px;">
                {i}
            </div>
            <div style="flex: 1;">
                <div style="font-weight:700; color:{tc}; font-size:0.98rem; margin-bottom:4px; letter-spacing:-0.2px;">{title}</div>
                <div style="color:{dc}; font-size:0.87rem; line-height:1.55; font-weight:500;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Header ───────────────────────────────
# st.markdown("""
# <div class="hero">
#     <div class="hero-badge">✨ AI Mental Health Insight</div>
#     <div class="hero-icon">🧠</div>
#     <div class="hero-title">Student Mental Health Classifier</div>
#     <div class="hero-sub">AI-powered risk assessment with personalized recovery plan</div>
# </div>
# """, unsafe_allow_html=True)

if not loaded:
    st.error("❌ Model files not found! Place .pkl files in the same folder.")
    st.stop()

# ── Section 1: Personal Info ──────────────
st.markdown('<div class="section-card"><div class="section-head">👤 Personal Information</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    gender  = st.selectbox("Gender", ["Female", "Male"])
    age     = st.selectbox("Age", list(range(15, 41)))
    year    = st.selectbox("Year of Study", ["1st Year","2nd Year","3rd Year","4th Year"])
with c2:
    cgpa    = st.selectbox("CGPA Range", ["3.50 - 4.00","3.00 - 3.49","2.50 - 2.99","2.00 - 2.49","Below 2.00"])
    married = st.selectbox("Marital Status", ["Single", "Married"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Lifestyle ──────────────────
st.markdown('<div class="section-card"><div class="section-head">📊 Lifestyle Factors</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    sleep    = st.selectbox("Sleep Hours Per Night", ["Less than 5 hours","5 - 6 hours","7 - 8 hours","More than 8 hours"])
    social   = st.selectbox("Social Support Level",  ["1 - No support","2 - Very little","3 - Some support","4 - Good support","5 - Great support"])
    academic = st.selectbox("Academic Pressure",     ["1 - Very Low","2 - Low","3 - Medium","4 - High","5 - Very High"])
with c4:
    study    = st.selectbox("Daily Study Hours",     ["1 - 2 hours","3 - 4 hours","5 - 6 hours","7 - 8 hours","More than 8 hours"])
    financial= st.selectbox("Financial Stress",      ["1 - No stress","2 - Little","3 - Moderate","4 - High stress","5 - Severe stress"])
    physical = st.selectbox("Exercise Days/Week",    ["0 days","1 - 2 days","3 - 4 days","5 - 6 days","Every day"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Symptoms ───────────────────
st.markdown('<div class="section-card"><div class="section-head">🧬 Mental Health Symptoms</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    depression = st.selectbox("Do you feel depressed? (sadness, hopelessness)", ["No","Yes"])
    anxiety    = st.selectbox("Do you experience anxiety? (worry, fear)",       ["No","Yes"])
with c6:
    panic     = st.selectbox("Do you have panic attacks?",              ["No","Yes"])
    treatment = st.selectbox("Have you sought specialist treatment?",   ["No","Yes"])
st.markdown('</div>', unsafe_allow_html=True)

cgpa_map     = {"3.50 - 4.00":3.75,"3.00 - 3.49":3.25,"2.50 - 2.99":2.75,"2.00 - 2.49":2.25,"Below 2.00":1.75}
sleep_map    = {"Less than 5 hours":4.0,"5 - 6 hours":5.5,"7 - 8 hours":7.5,"More than 8 hours":9.0}
study_map    = {"1 - 2 hours":1,"3 - 4 hours":3,"5 - 6 hours":5,"7 - 8 hours":7,"More than 8 hours":9}
physical_map = {"0 days":0,"1 - 2 days":1,"3 - 4 days":3,"5 - 6 days":5,"Every day":7}

# ── Predict Button ────────────────────────
predict = st.button("🔍 Check Your Mental Health Risk")

if predict:
    inp = {
        'Gender'           : 1 if gender=="Male" else 0,
        'Age'              : int(age),
        'Year'             : int(year[0]),
        'CGPA'             : cgpa_map[cgpa],
        'Married'          : 1 if married=="Married" else 0,
        'SleepHours'       : sleep_map[sleep],
        'SocialSupport'    : int(social[0]),
        'AcademicPressure' : int(academic[0]),
        'StudyHours'       : study_map[study],
        'FinancialStress'  : int(financial[0]),
        'PhysicalActivity' : physical_map[physical],
        'Depression'       : 1 if depression=="Yes" else 0,
        'Anxiety'          : 1 if anxiety=="Yes" else 0,
        'PanicAttack'      : 1 if panic=="Yes" else 0,
        'Treatment'        : 1 if treatment=="Yes" else 0,
    }

    arr   = np.array([[inp[c] for c in feature_cols]])
    pred  = model.predict(scaler.transform(arr))[0]
    proba = model.predict_proba(scaler.transform(arr))[0]

    low_pct = round(proba[0]*100, 1)
    med_pct = round(proba[1]*100, 1)
    high_pct= round(proba[2]*100, 1)
    conf    = round(proba[pred]*100, 1)

    labels  = {0:"🟢 Low Risk",    1:"🟡 Medium Risk", 2:"🔴 High Risk"}
    classes = {0:"result-low",     1:"result-medium",  2:"result-high"}
    icons   = {0:"✅",             1:"⚠️",             2:"🚨"}

    insights = {
        0: "Your mental health appears <b>good</b>! There are no major signs of concern currently. However, prevention is key — strengthen your routine with the suggestions below to protect your long-term well-being.",
        1: "You are experiencing <b>moderate stress</b>. While it may not be critical right now, it should not be ignored. Making small updates to your daily schedule can significantly improve this level — follow the active care steps listed below.",
        2: "Your results indicate <b>serious mental health challenges</b>. Just like consulting a doctor when dealing with a physical illness or fever, seeking expert guidance here is vital. <b>You do not have to carry this alone. 💙</b>"
    }
    insight_cls = {0:"insight-low", 1:"insight-medium", 2:"insight-high"}

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result Banner ─────────────────────
    st.markdown(f"""
    <div class="{classes[pred]}">
        <div style='font-size:2.8rem; margin-bottom:12px;'>{icons[pred]}</div>
        <h2 style='margin:0 0 8px 0; color:#f1f5f9; font-size:1.8rem; font-weight:800; letter-spacing: -0.5px; font-family: "Sora", sans-serif;'>
            {labels[pred]}
        </h2>
        <p style='color:#94a3b8; font-size:1rem; margin:0; font-weight: 600;'>
            Confidence: <span style="color:#f1f5f9; font-weight: 800;">{conf}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Insight ───────────────────────────
    st.markdown(f"""
    <div class="insight-box {insight_cls[pred]}">
        <div style="font-weight:800; font-size:1.05rem; margin-bottom:10px; display:flex; align-items:center; gap:6px;">📌 What does this mean?</div>
        <div style="font-weight: 500;">{insights[pred]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Pills ─────────────────
    st.markdown(f"""
    <p style='font-weight:800; color:#e2e8f0; font-size:1rem; margin-top:28px; margin-bottom:14px; letter-spacing:-0.3px;'>📊 Probability Breakdown:</p>
    <div class="prob-row">
        <div class="prob-pill pill-low">
            <div class="pill-val">{low_pct}%</div>
            <div class="pill-lbl">🟢 Low Risk</div>
        </div>
        <div class="prob-pill pill-medium">
            <div class="pill-val">{med_pct}%</div>
            <div class="pill-lbl">🟡 Medium Risk</div>
        </div>
        <div class="prob-pill pill-high">
            <div class="pill-val">{high_pct}%</div>
            <div class="pill-lbl">🔴 High Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recovery Plan ─────────────────────
    steps, risk_type = get_steps_for_risk(pred, low_pct, med_pct, high_pct)

    badge_text = {
        "low"   : "Maintenance & Prevention Plan",
        "medium": "Active Improvement Plan",
        "high"  : "Urgent Recovery Plan"
    }

    plan_colors = {"low":"#6ee7b7","medium":"#fcd34d","high":"#fca5a5"}
    plan_bg     = {"low":"rgba(16,185,129,0.1)","medium":"rgba(245,158,11,0.1)","high":"rgba(239,68,68,0.1)"}
    pc = plan_colors[risk_type]
    pb = plan_bg[risk_type]

    total_steps = len(steps)
    note = ""
    if pred == 1 and high_pct >= 25:
        note = f"<div style='color:#fcd34d; font-size:0.86rem; font-weight:700; margin-top:10px;'>⚠️ High Risk is at {high_pct}% — extra items have been added to your care plan.</div>"
    elif pred == 0 and med_pct >= 30:
        note = f"<div style='color:#6ee7b7; font-size:0.86rem; font-weight:700; margin-top:10px;'>ℹ️ Medium Risk is at {med_pct}% — precautionary elements are included.</div>"
    elif pred == 2 and high_pct >= 70:
        note = f"<div style='color:#fca5a5; font-size:0.86rem; font-weight:700; margin-top:10px;'>🚨 High confidence level ({high_pct}%) — crisis support options are attached.</div>"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:22px; padding:26px; margin-top:32px; backdrop-filter: blur(16px);">
        <div style="font-size:1.25rem; font-weight:800; color:#f1f5f9; margin-bottom:12px; letter-spacing:-0.5px; font-family:'Sora',sans-serif;">🗺️ Personalized Recovery Plan</div>
        <div style="display:inline-block; padding:6px 16px; border-radius:20px; background:{pb}; color:{pc}; font-size:0.82rem; font-weight:800;">
            {badge_text[risk_type]}
        </div>
        {note}
        <div style="color:#94a3b8; font-size:0.88rem; margin-top:14px; font-weight:600;">
            {total_steps} personalized steps — custom tailored to your risk levels and probabilities
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render interactive step cards
    render_steps(steps, risk_type)

    # ── Crisis Warning ────────────────────
    if pred == 2:
        st.markdown("""
        <div style='background: rgba(239,68,68,0.05); border: 1px dashed rgba(248,113,113,0.4); border-radius:18px; padding:24px; text-align:center; margin-top:24px;'>
            <div style='font-size:2rem; margin-bottom:8px;'>🆘</div>
            <p style='color:#fca5a5; font-weight:800; font-size:1.1rem; margin:0 0 6px 0; letter-spacing:-0.3px;'>
                If you are facing a crisis or feel unsafe:
            </p>
            <p style='color:#fda4af; font-size:0.95rem; margin:0; line-height:1.6; font-weight:600;'>
                Pakistan Umang Helpline: <span style="font-size:1.05rem; font-weight:800; color:#f87171;">0317-4288665</span> (24/7)<br>
                Or visit your university counseling center immediately today.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── View Answers Expandable Section ───────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View Your Submitted Answers"):
    a1, a2 = st.columns(2)
    with a1:
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Year:** {year}")
        st.write(f"**CGPA:** {cgpa}")
        st.write(f"**Sleep:** {sleep}")
        st.write(f"**Social Support:** {social}")
        st.write(f"**Academic Pressure:** {academic}")
    with a2:
        st.write(f"**Study Hours:** {study}")
        st.write(f"**Financial Stress:** {financial}")
        st.write(f"**Exercise:** {physical}")
        st.write(f"**Depression:** {depression}")
        st.write(f"**Anxiety:** {anxiety}")
        st.write(f"**Panic Attacks:** {panic}")
        st.write(f"**Treatment:** {treatment}")

# Footer
st.markdown("""
<p style='text-align:center; color:#475569; font-size:0.85rem; font-weight:600; margin-top:36px; padding-bottom:20px;'>
    🧠 Powered by Machine Learning + A* Search Algorithm<br>
    <span style="font-weight:500; color:#334155;">This tool is for educational purposes only — not a medical diagnosis.</span>
</p>
""", unsafe_allow_html=True)