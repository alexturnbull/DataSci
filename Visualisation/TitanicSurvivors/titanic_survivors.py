import streamlit as st
import altair as alt
import pandas as pd


survivors_by_class_and_sex = pd.read_parquet('/home/alex/Projects/DataSci/SampleData/Datalake/Titanic_Survivors_Lake/survivors_by_class_and_sex.parquet')
predicted_survivors_by_class_and_sex = pd.read_parquet('/home/alex/Projects/DataSci/SampleData/Datalake/Titanic_Survivors_Lake/predicted_survivors_by_class_and_sex.parquet')


st.title('Titanic Survivors')

st.write('This shows the real survival rate of passangers on the titanic by class and sex')

# Display the data as a barchart with the class and sex on the x-axis and the number of survivors on the y-axis
chart = (
    alt.Chart(survivors_by_class_and_sex.melt("Pclass", var_name="measure", value_name="value"))
    .mark_bar()
    .encode(x="measure", y="value", color="measure", column="Pclass")
)
st.altair_chart(chart)

st.write('This shows the predicted survival rate of passangers on the titanic by class and sex')

chart = (
    alt.Chart(predicted_survivors_by_class_and_sex.melt("Pclass", var_name="measure", value_name="value"))
    .mark_bar()
    .encode(x="measure", y="value", color="measure", column="Pclass")
)
st.altair_chart(chart)
