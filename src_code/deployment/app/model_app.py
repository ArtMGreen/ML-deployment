import streamlit as st
import requests as rq



st.title('Grade prediction')

icp = st.slider("Your in-class participation points", min_value=0, max_value=5)
midterm = st.slider("Your Midterm points", min_value=0, max_value=30)
a1 = st.slider("Your percentage in Assignment 1", min_value=0, max_value=100)

predicted_grade = rq.get("http://ml-api-app:8000/predict",
                         params={"icp": icp,
                                 "midterm": midterm,
                                 "a1": a1})

st.write('Your expected grade is', predicted_grade.text)
