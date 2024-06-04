import io

import altair as alt
import pandas as pd
import streamlit as st

text = """Date|sys|dias
2022/09/30 08:49:00|149|109
2022/09/30 17:19:00|139|97
2022/09/30 21:40:00|127|91
2022/10/01 09:00:00|144|106
2022/10/01 15:41:00|131|96
2022/10/02 09:10:00|140|104
"""

df = pd.read_csv(io.StringIO(text), sep="|")
df.Date = pd.to_datetime(df.Date)

chart = (
    alt.Chart(df.melt("Date", var_name="measure", value_name="value"))
    .mark_bar()
    .encode(x="measure", y="value", color="measure", column="Date")
)
st.altair_chart(chart)