import streamlit as st
from utils import project1_desc as p1d
import pandas as pd

def app():
    st.write(''' 
    ## streamlit으로 dataframe을 가져와보자
    ''')

    p1d.desc()
    
    df=p1d.get_data()
    st.dataframe(df.iloc[:3,:3])

    p1d.asce()

    df_now = p1d.takes_the_code()
    st.dataframe(df_now) 
    