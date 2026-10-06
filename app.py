import streamlit as st, pandas as pd, joblib

M = joblib.load('healthwise_model.joblib')
clf, regs, FEATURES, NUM, med = M['clf'], M['regs'], M['features'], M['num'], M['med']

def predict(c):
    row = pd.DataFrame([c])
    for k in NUM:
        if k not in row or pd.isna(row.at[0, k]): row[k] = med[k]
    t = clf.predict(row[FEATURES])[0]
    return t, float(regs[t].predict(row[FEATURES])[0])

st.set_page_config(page_title="HealthWise Premium Engine", page_icon="🏥")
st.title("🏥 HealthWise — Smart Premium Engine")
st.caption("Stage 1 classifies the risk tier; Stage 2 prices the customer with that tier's own model. Change any input and the result updates instantly.")

col1, col2 = st.columns(2)
with col1:
    age      = st.slider("Age", 18, 64, 40)
    bmi      = st.slider("BMI", 16.0, 50.0, 27.0, 0.1)
    children = st.slider("Children", 0, 4, 1)
    exercise = st.slider("Exercise / week", 0, 7, 3)
with col2:
    sex    = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])

tier, premium = predict(dict(age=age, bmi=bmi, children=children, exercise_freq=exercise,
                             sex=sex, smoker=smoker, region=region))
m1, m2 = st.columns(2)
m1.metric("Predicted Risk Tier", tier)
m2.metric("Estimated Annual Premium", f"${premium:,.0f}")

# BONUS: batch scoring of unseen customers
st.divider()
st.subheader("Score unseen customers (CSV)")
up = st.file_uploader("Upload a CSV with columns: age, bmi, children, exercise_freq, sex, smoker, region", type="csv")
if up:
    new = pd.read_csv(up)
    out = [predict(dict(r)) for _, r in new.iterrows()]
    new["predicted_tier"] = [t for t, _ in out]
    new["predicted_premium"] = [round(p) for _, p in out]
    st.dataframe(new)
    st.download_button("Download predictions", new.to_csv(index=False), "predictions.csv")
