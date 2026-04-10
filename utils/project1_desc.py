import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO 

def desc():
    st.write('''
    indent가 중요하단다
    ''')


def get_data():
    fn = './data/stock_px_2.csv'
    df = pd.read_csv(fn, index_col=[0])
    return df


def takes_the_code():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download'
    try:
        tables = pd.read_html(url, header=0, encoding='euc-kr')
    
        if  len(tables) >0 : 
            return tables[0]
        else:
            return pd.DataFrame()    
    except Exception as e:   
        print('에러발생함 :' ,e)
        return pd.DataFrame()

''' teacher instructor 김태현 
def get_code():
code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0, encoding='euc-kr')
df = code_all[0]
return df '''
