import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path
from io import StringIO
import requests
import yfinance as yf
import pandas_datareader.data as web
from datetime import timedelta as td, datetime, date
import calendar
import koreanize_matplotlib

def desc1():
    st.write('''
    indent가 중요하단다
    ''')


def asce():
    st.write('''
    krx에서 상장회사의 종목 정보를 가져옵니다.''')

def asce1():
    st.write('''
    csv파일을 불러 옵니다''')





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




def data_from_csv(fn):
    #file_name ='./data/'+'stock_name_code_ChangePrice.csv'
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


def data_from_csv():
    file_name = './data/'+'stock_name_code_ChangePrice.csv'
    df = pd.read_csv(file_name)
    df = df.iloc[:, 1:]
    df.columns = ['names', 'code', 'change_price']
    return df


def get_eps(stock_name_code_ChangePrice_df):
    df = stock_name_code_ChangePrice_df
    df_10 = df.sort_values(by='change_price', ascending=False)[:10]
    df_dict = {}
    for i, j in df_10.iterrows():
        url = f"https://finance.naver.com/item/main.naver?code=000660"
        fis = pd.read_html(StringIO(requests.get(url).text))
        df = fis[4].iloc[9 , : ]
        if len(df) <4:
            continue    
        df.name =j.names
        df_dict[j.code] = df
    return (df_dict, df_10)


def date_transform(df):
     #df['level_1'] = pd.to_datetime(df['level_1'])
    df2 = pd.DataFrame()
    name = df.columns[-1] # 
    df2[name] = df[name]  # 방식인 이유 , 인덱스그대로 쓰려고
    for i in df.index:
        str_date = df.loc[i].level_1
        year = int(str_date[:4])
        month = int(str_date[5:7])
        fd, last_day = calendar.monthrange(year, month)
        dates = date(year, month, last_day)
        df2.loc[i,'level_1']=dates # 열추가 이름: level_1 , 시리즈: dates
        df2 = df2 [['level_1', name]]
    return df2




def get_date_timestamp(dfs):
    df_dict = dfs.copy()
    for i in df_dict:
        df = df_dict[i]
        df = df.dropna()
        df = df.reset_index()
        name = df.columns[3]
        df = df[df.level_0 == '최근 분기 실적'][['level_1', name]]
  
        df = date_transform(df)
        start=date(df.iloc[0, 0].year, df.iloc[0,0].month,1)
        end=df.iloc[-1, 0]
        select_figure_TF = select_figure(df) 
        df_dict[i] = [start, end, select_figure_TF, df]
        #return df_dict
    return df_dict



def select_figure(df):
    name = df.columns[-1]  #name :: 고려아연
    df_value = pd.to_numeric(df[name])
    if len(df_value) == len(df_value[df_value>0]):
        return True
    return False


def get_stock_data(df_trans):  # df_trans은 자료형은? 딕셔너리
    for i in df_trans:
        start = df_trans[i][0]
        end = date.today()
        codes = i+'.KS'
        df = yf.download(codes , start=start, end=end)
        df = df['Close']
        #df = pd.to_numeric(df)
        df_trans[i].append(df)
    return df_trans



def plot_data(i, stock, eps):
    company = eps.columns[-1]
    fig, ax = plt.subplots(figsize=(8,6))
    eps_ax = ax.twinx()
    ax.plot(stock.index, stock.values, label='주가')
    eps_ax.plot(eps.index, eps[company], 'o-r', label='주당순이익')
    plt.title(company)
    ax.legend(loc='upper left')
    eps_ax.legend(loc='lower right')
    #plt.show()
    #file_name = './data/'+str(i)+'.png'
    file_name ='..\\data\\'+ str(i)+'.png'
    #plt.savefig(file_name)
    return fig


def plots(df):
    figs=[]
    for i in df:
        stock = df[i][4]
        eps_data = df[i][3]
        eps_data.index = pd.to_datetime(df[i][3]['level_1'], format = '%Y-%m-%d') #일자서식 통일
        fig= plot_data(i, stock, eps_data)
        figs.append(fig)
    return figs
# pyplot(x,y,값) <--- streamlit 에 있는 함수 .pyplot()



#streamlit 에서는 app.py 파일이 있는 디렉토리가"/" 가 된다. 








#    st.function_name(data=None, *, options...)
#    st.write(*args, **kwargs)