import streamlit as st
import pandas as pd

st.title('Daftar Bahan Tidak Kritis')
data = pd.read_csv("./Daftar_Bahan_Tidak_Kritis.csv", sep=';',  error_bad_lines=False)  # read a CSV file inside the 'data" folder next to 'app.py'

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.table(data)