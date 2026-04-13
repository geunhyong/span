import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path



def desc1():
    st.write('''
    indent가 중요하단다
    ''')


def asce():
    st.write('''
    krx에서 상장회사의 종목 정보를 가져옵니다.''')


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


def get_code_name():
    code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0, encoding='cp949')
    columns = code_all[0].columns
    code_name_stock_df = code_all[0][[columns[0], columns[2]]]
    code_name_stock_df.columns = ['names', 'code']
    code_name_stock_df['change_price'] = np.nan 
    return code_name_stock_df




def data_from_csv():
    file_name ='./data/'+'stock_name_code_ChangePrice.csv'
    df = pd.read_csv(file_name)
    df = df.iloc[ :,1:]
    df.columns = ['names','code', 'change_price']
    return df



def data_from__csv():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, 'data', 'stock_name_code_ChangePrice.csv') 
    df = pd.read_csv(file_path, encoding='cp949')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='cp949')
    return df






def data_from_csv___():
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / 'data' /  'stock_name_code_ChangePrice.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, encoding='cp949')
        return df
    else:
        raise FileNotFoundError(f'파일 없음 : {file_path}')    