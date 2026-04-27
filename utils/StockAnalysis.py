import streamlit as st
from datetime import timedelta as td, datetime, date
import pandas as pd
import requests
from io import StringIO
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import calendar
import koreanize_matplotlib
#from dateutil.relativedelta import relativedelta as rtd

def get_code_name():
	code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0, encoding='euc-kr')
	columns = code_all[0].columns
	df = code_all[0][[columns[0], columns[2]]]
	df.columns=['names', 'code']
	df['change_price'] = np.nan
	return df

def start_end():
	end = datetime.now()-td(days=1)
	start = datetime.now()-td(days=100)
	return start, end

def get_change_price(df):
	start, end = start_end()
	for i, j in enumerate(df.code):
		target_code = j
		stock = web.DataReader(target_code, 'naver', start=start, end=end)
		change_price = int(stock.iloc[-1, 3]) - int(stock.iloc[0, 3])
		df.iloc[i, 2] = change_price
	return df

def data_from_csv(fn):
	#file_name = './data/'+'stock_name_code_ChangePrice.csv'
	df = pd.read_csv(fn, encoding='cp949')
	df = df.iloc[:, 1:]
	df.columns=['names', 'code', 'change_price']
	return df

def get_eps(df):
	df_10 = df.sort_values(by='change_price', ascending=False)[:10]
	df_dict = {}                                                                                    
	for i, j in df_10.iterrows():
		url = f"https://finance.naver.com/item/main.nhn?code={j.code}" 
		financial_stmt = pd.read_html(StringIO(requests.get(url).text))                             
		if type(financial_stmt[3].columns) != type(financial_stmt[0].columns):                      
			df = financial_stmt[3].iloc[9, :]                                                       
		else:                                                                                       
			df = financial_stmt[4].iloc[9, :]                                                       
		if len(df)<4:
			continue
		df.name = j.names                                                         
		df_dict[j.code] = df
	return (df_dict, df_10)

def date_transform(df):
    for i in df.index:
        str_date = df.loc[i].level_1
        year = int(str_date[:4])
        month = int(str_date[5:7])
        fd, last_day = calendar.monthrange(year, month)
        dates = date(year, month, last_day)
        df.loc[i,'level_1']=dates
    return df

def select_figure(df):
    name = df.columns[-1]
    df_value = pd.to_numeric(df[name])
    if len(df_value) == len(df_value[df_value>0]): 
        return True
    return False

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
    return df_dict

def get_stock_data(df_trans):
    for i in df_trans:
        start = df_trans[i][0]
        #end = df_trans[i][1]
        end = date.today()
        df = web.DataReader(i, 'naver', start=start, end=end)
        df = df['Close']
        df = pd.to_numeric(df)
        df_trans[i].append(df)
    return df_trans

def adj_datetime_data(df):
    df_dict={}
    for i in df:
        df2 = df[i][3]
        df2[df2.columns[0]] = pd.to_datetime(df2[df2.columns[0]])
        df2[df2.columns[1]] = pd.to_numeric(df2[df2.columns[1]])
        df2.set_index('level_1', inplace=True)
        df_dict[i] = [df[i][2], df[i][4], df2]
    return df_dict

def plot_data(i, stock, eps):
    company = eps.columns[-1]
    fig, ax = plt.subplots(figsize=(8,6))
    eps_ax = ax.twinx()
    ax.plot(stock.index, stock.values, label='주가')
    eps_ax.plot(eps.index, eps[company], 'o-r', label='주당순이익')
    plt.title(company)
    ax.legend(loc='upper left')
    eps_ax.legend(loc='lower right')
    file_name = './data/'+str(i)+'.png'
    plt.savefig(file_name)

def plots(df):
    for i in df:
        stock = df[i][1]
        eps_data = df[i][2]
        plot_data(i, stock, eps_data)
