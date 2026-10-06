import numpy as np, pandas as pd, altair as alt, joblib, streamlit as st
from pathlib import Path

st.set_page_config(page_title="HealthWise Premium Engine", page_icon="🏥", layout="wide")
HERE = Path(__file__).parent

# ───────────────────────── Style ─────────────────────────
ACCENT = "#6366f1"
TIER_COL = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
TIER_SCALE = alt.Scale(domain=list(TIER_COL), range=list(TIER_COL.values()))
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3 {font-family:'Inter',sans-serif;}
.block-container {padding-top:2rem; max-width:1250px;}
.hero {background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
       padding:1.6rem 2rem; border-radius:18px; margin-bottom:1.2rem;}
.hero h1 {margin:0; font-size:2rem; font-weight:800; color:#fff;}
.hero p {margin:.35rem 0 0; color:#fff; opacity:.92;}
.card {border:1px solid rgba(128,128,128,.28); background:rgba(128,128,128,.08);
       border-radius:16px; padding:1rem 1.25rem; height:100%;}
.card .lbl {font-size:.75rem; text-transform:uppercase; letter-spacing:.07em; opacity:.7;}
.card .val {font-size:2rem; font-weight:800; line-height:1.25;}
.card .sub {font-size:.85rem; opacity:.7;}
.badge {display:inline-block; padding:.2rem .9rem; border-radius:999px; color:#fff;
        font-weight:800; font-size:1.4rem; margin-top:.15rem;}
.insight {border-left:4px solid #6366f1; padding:.6rem 1rem; margin:.5rem 0;
          background:rgba(99,102,241,.10); border-radius:0 10px 10px 0;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
</style>""", unsafe_allow_html=True)

def card(label, value, sub="", color=None):
    v = f'<span class="badge" style="background:{color}">{value}</span>' if color else f'<div class="val">{value}</div>'
    st.markdown(f'<div class="card"><div class="lbl">{label}</div>{v}<div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)

# ───────────────────────── Model & data ─────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(HERE / "healthwise_model.joblib")

@st.cache_data
def load_data():
    p = HERE / "healthwise.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).drop_duplicates("customer_id")
    d["age_band"] = pd.cut(d.age, [17, 29, 39, 49, 64], labels=["18-29", "30-39", "40-49", "50-64"])
    return d

M = load_model()
clf, regs, FEATURES, NUM, med = M["clf"], M["regs"], M["features"], M["num"], M["med"]
DATA = load_data()

# Linear link learned from the dataset:  BMI ≈ 42.17 + 0.0136·age − 3.705·exercise_days
B0, B_AGE, B_EX = 42.17, 0.0136, -3.705
link = lambda age, ex: B0 + B_AGE * age + B_EX * ex

def predict_df(df):
    X = df.copy()
    for k in NUM:
        X[k] = pd.to_numeric(X[k], errors="coerce").fillna(med[k]) if k in X else med[k]
    X = X[FEATURES]
    tiers = clf.predict(X)
    prem = np.zeros(len(X))
    for t in np.unique(tiers):
        m = tiers == t
        prem[m] = regs[t].predict(X[m])
    return tiers, prem

def score(c):
    t, p = predict_df(pd.DataFrame([c]))
    return t[0], float(p[0])

# ───────────────────────── Sidebar inputs ─────────────────────────
for k, v in dict(age=40, exercise=3, bmi=27.0).items():
    st.session_state.setdefault(k, v)

def sync_bmi():
    if st.session_state.mode == "Simulated":
        b = link(st.session_state.age, st.session_state.exercise)
        st.session_state.bmi = float(np.clip(round(b, 1), 16.0, 50.0))

with st.sidebar:
    st.markdown("### 🎛️ Customer profile")
    st.radio("Slider mode", ["Simulated", "General"], key="mode", horizontal=True, on_change=sync_bmi,
             help="Simulated: BMI moves automatically when Age or Exercise changes (based on patterns in your data). "
                  "General: every slider is independent.")
    if st.session_state.mode == "Simulated":
        st.caption("🔗 BMI auto-updates with Age and Exercise. You can still fine-tune BMI by hand.")
    else:
        st.caption("🎚️ All sliders move independently.")
    age = st.slider("Age", 18, 64, key="age", on_change=sync_bmi)
    exercise = st.slider("Exercise days / week", 0, 7, key="exercise", on_change=sync_bmi)
    bmi = st.slider("BMI", 16.0, 50.0, step=0.1, key="bmi")
    children = st.slider("Children", 0, 4, 1)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])

base = dict(age=age, bmi=bmi, children=children, exercise_freq=exercise, sex=sex, smoker=smoker, region=region)
sim = st.session_state.mode == "Simulated"

def adj_bmi(a, e):
    """BMI after changing age/exercise: follows the link in Simulated mode, unchanged in General mode."""
    if not sim:
        return bmi
    return float(np.clip(bmi + link(a, e) - link(age, exercise), 16, 50))

tier, premium = score(base)
probs = dict(zip(clf.classes_, clf.predict_proba(pd.DataFrame([base])[FEATURES])[0]))
bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

# ───────────────────────── Header ─────────────────────────
st.markdown("""<div class="hero"><h1>🏥 HealthWise — Smart Premium Engine</h1>
<p>Stage 1 classifies the risk tier; Stage 2 prices the customer with that tier's own model.
Adjust the profile in the sidebar and everything updates instantly.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "🔮 Predictive analysis", "💡 Data insights", "📂 Batch scoring"])

# ───────────────────────── Tab 1: Prediction ─────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Predicted risk tier", tier, "Stage 1 classifier", TIER_COL[tier])
    with c2: card("Estimated annual premium", f"${premium:,.0f}", "Stage 2 tier-specific model")
    with c3: card("Monthly equivalent", f"${premium/12:,.0f}", "per month")
    with c4: card("BMI category", bmi_cat, f"BMI {bmi:.1f}  ·  {exercise} exercise days/wk")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Tier confidence")
        pdf = pd.DataFrame({"Tier": list(probs), "Probability": list(probs.values())})
        st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=6).encode(
            y=alt.Y("Tier:N", sort=["Low", "Medium", "High"], title=None),
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Tier:N", scale=TIER_SCALE, legend=None),
            tooltip=["Tier", alt.Tooltip("Probability:Q", format=".0%")]).properties(height=170))
    with right:
        st.markdown("##### Where this premium sits")
        if DATA is not None:
            same = DATA[DATA.risk_tier == tier].annual_charge
            pct = (same < premium).mean() * 100
            st.markdown(f"""<div class="insight">Higher than <b>{pct:.0f}%</b> of existing <b>{tier}</b>-tier customers
            (tier average <b>${same.mean():,.0f}</b>).</div>
            <div class="insight">Overall portfolio average premium is <b>${DATA.annual_charge.mean():,.0f}</b>;
            this customer is <b>{premium/DATA.annual_charge.mean():.1f}×</b> that.</div>""", unsafe_allow_html=True)
        else:
            st.info("Add healthwise.csv next to app.py to see portfolio comparisons.")
        if sim:
            st.markdown(f'<div class="insight">🔗 Simulated link: BMI ≈ {link(age, exercise):.1f} expected for age {age} '
                        f'and {exercise} exercise days.</div>', unsafe_allow_html=True)

# ───────────────────────── Tab 2: Predictive analysis ─────────────────────────
with tab2:
    st.markdown("#### What-if scenarios")
    st.caption("Each scenario changes one thing and re-runs the full two-stage model.")
    scen = {
        ("Quit smoking" if smoker == "yes" else "Starts smoking"): {**base, "smoker": "no" if smoker == "yes" else "yes"},
        "+2 exercise days / week": {**base, "exercise_freq": min(7, exercise + 2),
                                    "bmi": adj_bmi(age, min(7, exercise + 2))},
        "BMI down by 3 points": {**base, "bmi": max(16.0, bmi - 3)},
        "5 years older": {**base, "age": min(64, age + 5), "bmi": adj_bmi(min(64, age + 5), exercise)},
    }
    cols = st.columns(4)
    for col, (name, cfg) in zip(cols, scen.items()):
        t2, p2 = score(cfg)
        with col:
            st.metric(name, f"${p2:,.0f}", f"{p2 - premium:+,.0f} vs now", delta_color="inverse")
            st.caption(f"Tier: {t2}" + (" (changed)" if t2 != tier else ""))

    st.markdown("#### Premium trajectory & sensitivity")
    def sweep(col, values, bmi_fn=None):
        df = pd.DataFrame([base] * len(values)); df[col] = values
        if bmi_fn: df["bmi"] = [bmi_fn(v) for v in values]
        t, p = predict_df(df)
        return pd.DataFrame({col: values, "Premium": p, "Tier": t})

    def curve(df, x, title, now):
        line = alt.Chart(df).mark_line(interpolate="step-after", color=ACCENT, strokeWidth=3).encode(
            x=alt.X(f"{x}:Q", title=title), y=alt.Y("Premium:Q", title="Annual premium ($)", axis=alt.Axis(format="$,.0f")))
        pts = alt.Chart(df).mark_circle(size=60).encode(
            x=f"{x}:Q", y="Premium:Q", color=alt.Color("Tier:N", scale=TIER_SCALE, legend=alt.Legend(orient="bottom")),
            tooltip=[x, alt.Tooltip("Premium:Q", format="$,.0f"), "Tier"])
        mark = alt.Chart(pd.DataFrame({x: [now]})).mark_rule(strokeDash=[5, 4], color="gray").encode(x=f"{x}:Q")
        return (line + pts + mark).properties(height=250)

    a, b, c = st.columns(3)
    with a:
        st.markdown("**By age**" + (" (BMI follows)" if sim else ""))
        st.altair_chart(curve(sweep("age", list(range(18, 65)), lambda v: adj_bmi(v, exercise)), "age", "Age", age))
    with b:
        st.markdown("**By BMI**")
        st.altair_chart(curve(sweep("bmi", [round(x, 1) for x in np.arange(16, 50.1, 0.5)]), "bmi", "BMI", bmi))
    with c:
        st.markdown("**By exercise days**" + (" (BMI follows)" if sim else ""))
        st.altair_chart(curve(sweep("exercise_freq", list(range(0, 8)), lambda v: adj_bmi(age, v)),
                              "exercise_freq", "Exercise days / week", exercise))
    st.caption("The model is a decision tree, so curves are step-shaped: premiums jump when a risk threshold is "
               "crossed instead of rising smoothly. The dashed line marks the current profile.")

    st.markdown("#### What drives the risk tier?")
    pre, tree = clf.named_steps["pre"], clf.named_steps["model"]
    imp = pd.Series(tree.feature_importances_, index=pre.get_feature_names_out()).rename(
        lambda s: s.split("__")[-1].replace("_", " ")).sort_values(ascending=False)
    imp = imp[imp > 0].reset_index(); imp.columns = ["Feature", "Importance"]
    st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=6, color=ACCENT).encode(
        y=alt.Y("Feature:N", sort="-x", title=None), x=alt.X("Importance:Q", axis=alt.Axis(format="%")),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")]).properties(height=max(120, 34 * len(imp))))

# ───────────────────────── Tab 3: Insights ─────────────────────────
with tab3:
    if DATA is None:
        st.warning("Add **healthwise.csv** to the same folder as app.py (and your GitHub repo) to enable this tab.")
    else:
        d = DATA
        k1, k2, k3, k4 = st.columns(4)
        with k1: card("Customers", f"{len(d):,}", "in the portfolio")
        with k2: card("Avg annual premium", f"${d.annual_charge.mean():,.0f}", f"median ${d.annual_charge.median():,.0f}")
        with k3: card("Smokers", f"{(d.smoker == 'yes').mean():.0%}", "of customers")
        with k4: card("High-risk share", f"{(d.risk_tier == 'High').mean():.0%}", "of customers")

        sm = d.groupby("smoker").annual_charge.mean()
        r_ab, r_eb = d.age.corr(d.bmi), d.exercise_freq.corr(d.bmi)
        hi_sm = (d[d.smoker == "yes"].risk_tier == "High").mean()
        drivers = ", ".join(imp.Feature.head(3).str.lower())
        st.markdown("#### Key insights")
        st.markdown(f"""
<div class="insight">🚬 <b>Smoking is the biggest cost lever:</b> smokers pay <b>${sm['yes']:,.0f}</b> on average vs
<b>${sm['no']:,.0f}</b> for non-smokers (<b>{sm['yes']/sm['no']:.1f}×</b>), and {hi_sm:.0%} of smokers land in the High tier.</div>
<div class="insight">🏃 <b>Exercise and BMI move together</b> (correlation {r_eb:.2f}). Each extra weekly exercise day goes with
roughly <b>{abs(B_EX):.1f}</b> lower BMI points, so improving fitness lowers premiums indirectly.</div>
<div class="insight">📅 <b>Age and BMI are only weakly linked</b> (correlation {r_ab:.2f}). Average BMI barely rises across age bands,
which is why the Simulated mode ties BMI mostly to exercise, with age as a small effect.</div>
<div class="insight">🌳 <b>Top model drivers:</b> {drivers}.</div>""", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### Average premium by age band & smoking")
            g = d.groupby(["age_band", "smoker"], observed=True).annual_charge.mean().reset_index()
            st.altair_chart(alt.Chart(g).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("age_band:N", title="Age band"), xOffset="smoker:N",
                y=alt.Y("annual_charge:Q", title="Avg premium ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("smoker:N", scale=alt.Scale(domain=["no", "yes"], range=["#10b981", "#ef4444"]),
                                title="Smoker"), tooltip=["age_band", "smoker", alt.Tooltip("annual_charge:Q", format="$,.0f")]
            ).properties(height=280))
        with r2:
            st.markdown("##### Risk-tier mix: smokers vs non-smokers")
            m = d.groupby(["smoker", "risk_tier"]).size().reset_index(name="n")
            st.altair_chart(alt.Chart(m).mark_bar().encode(
                y=alt.Y("smoker:N", title="Smoker"), x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share"),
                color=alt.Color("risk_tier:N", scale=TIER_SCALE, title="Tier"),
                tooltip=["smoker", "risk_tier", "n"]).properties(height=280))

        r3, r4 = st.columns(2)
        s = d.dropna(subset=["bmi", "exercise_freq"])
        with r3:
            st.markdown(f"##### Age vs BMI  (r = {r_ab:.2f})")
            sc = alt.Chart(s.sample(min(600, len(s)), random_state=1)).mark_circle(opacity=.45, size=40, color=ACCENT).encode(
                x=alt.X("age:Q", scale=alt.Scale(zero=False)), y=alt.Y("bmi:Q", scale=alt.Scale(zero=False)))
            st.altair_chart((sc + sc.transform_regression("age", "bmi").mark_line(color="#ef4444", strokeWidth=3)).properties(height=280))
        with r4:
            st.markdown(f"##### Exercise vs BMI  (r = {r_eb:.2f})")
            sc = alt.Chart(s.sample(min(600, len(s)), random_state=1)).mark_circle(opacity=.45, size=40, color="#10b981").encode(
                x=alt.X("exercise_freq:Q", title="Exercise days / week"), y=alt.Y("bmi:Q", scale=alt.Scale(zero=False)))
            st.altair_chart((sc + sc.transform_regression("exercise_freq", "bmi").mark_line(color="#ef4444", strokeWidth=3)).properties(height=280))

        st.markdown("##### Correlation map")
        cm = d[["age", "bmi", "children", "exercise_freq", "annual_charge"]].corr().round(2)
        cm = cm.reset_index().melt("index"); cm.columns = ["a", "b", "r"]
        hm = alt.Chart(cm).mark_rect(cornerRadius=4).encode(
            x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None),
            color=alt.Color("r:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]), title="r"))
        st.altair_chart((hm + hm.mark_text(fontSize=13).encode(text=alt.Text("r:Q", format=".2f"),
                        color=alt.value("black"))).properties(height=300))

# ───────────────────────── Tab 4: Batch scoring ─────────────────────────
with tab4:
    st.markdown("#### Score unseen customers (CSV)")
    need = ["age", "bmi", "children", "exercise_freq", "sex", "smoker", "region"]
    up = st.file_uploader(f"Upload a CSV with columns: {', '.join(need)}", type="csv")
    if up:
        new = pd.read_csv(up)
        missing = [c for c in need if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            t, p = predict_df(new)
            new["predicted_tier"], new["predicted_premium"] = t, p.round()
            n1, n2, n3 = st.columns(3)
            n1.metric("Customers scored", f"{len(new):,}")
            n2.metric("Average premium", f"${new.predicted_premium.mean():,.0f}")
            n3.metric("High-risk share", f"{(new.predicted_tier == 'High').mean():.0%}")
            st.dataframe(new)
            st.download_button("Download predictions", new.to_csv(index=False), "predictions.csv")
