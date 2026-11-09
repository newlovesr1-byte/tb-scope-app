import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('tb_xgboost_model.pkl')

st.title("TB-SCOPE AI: ระบบคัดกรองความเสี่ยงวัณโรคปอด")
st.write("กรอกผลเลือด CBC และอายุเพื่อประเมินความเสี่ยงด้วยปัญญาประดิษฐ์")

st.sidebar.header("กรอกข้อมูลผลเลือด")
age = st.sidebar.number_input("อายุ (ปี)", min_value=1, max_value=120, value=50)
wbc = st.sidebar.number_input("WBC", value=7.5)
hb = st.sidebar.number_input("Hb", value=13.0)
hct = st.sidebar.number_input("Hct", value=39.0)
mcv = st.sidebar.number_input("MCV", value=88.0)
rdw = st.sidebar.number_input("RDW", value=13.5)
plt = st.sidebar.number_input("Plt", value=250)
neu = st.sidebar.number_input("Neu", value=60.0)
lym = st.sidebar.number_input("Lym", value=30.0)
mono = st.sidebar.number_input("Mono", value=6.0)

nlr = neu / lym if lym > 0 else 0.0

input_data = pd.DataFrame([{
    'WBC': wbc, 'Hb': hb, 'Hct': hct, 'MCV': mcv, 'RDW': rdw,
    'Plt': plt, 'Neu': neu, 'Lym': lym, 'Mono': mono, 'NLR': nlr, 'อายุ_num': age
}])

if st.button("ประเมินความเสี่ยง"):
    prob = model.predict_proba(input_data)[0][1]
    threshold = 0.3
    
    st.markdown("---")
    if prob >= threshold:
        st.error(f"พบความเสี่ยงสูง (Detected) - ค่าความเสี่ยง: {prob*100:.1f}%")
        st.markdown("**คำแนะนำ:** ควรส่งตรวจยืนยันเพิ่มเติมด้วยการตรวจเสมหะหรือเอกซเรย์ปอด")
    else:
        st.success(f"ปกติ / ความเสี่ยงต่ำ (Not Detected) - ค่าความเสี่ยง: {prob*100:.1f}%")
        st.markdown("**คำแนะนำ:** ไม่พบสัญญาณความเสี่ยงเด่นชัด ติดตามอาการตามนัด")
