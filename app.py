import pandas as pd
import joblib
import streamlit as st

model = joblib.load('model.pkl')
pipeline = joblib.load('pipeline.pkl')

st.title("!! Insurance Charges !!")

st.markdown("provide the follewing details")

age = st.slider("Age",18,60,40)
sex = st.selectbox("SEX",['M','F'])
region = st.selectbox("In Which Region You Live",['southwest','northwest','southeast','northwest'])
smoker = st.selectbox("You Do Smoke",['Yes','No'])
bmi_category = st.selectbox("BMI",['Normal','Obese','Overweight','Underweight'])
children = st.number_input("Children",0,10,0)

if st.button("Predict"):
    row_input = {
        'age':age,
        'sex':sex,
        'region':region,
        'smoker':smoker,
        'bmi_category':bmi_category,
        'children':children
    }

    input_df = pd.DataFrame([row_input])

    pipeline_input = pipeline.transform(input_df)
    prediction = model.predict(pipeline_input)[0]
    
    if prediction:
        st.success(f"💰 Estimated Insurance Charges: ₹ {prediction:,.2f} Per Year")
    else:
        st.error("Something got wrong")    