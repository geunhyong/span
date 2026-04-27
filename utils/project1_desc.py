import streamlit as st
from datetime import timedelta as td, datetime, date
import pandas as pd
import requests
from bs4 import BeautifulSoup as BS
from io import StringIO
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import calendar
import koreanize_matplotlib
#from dateutil.relativedelta import relativedelta as rtd
from scipy.stats import norm
from scipy import stats


def get_code_name():
	code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0, encoding='euc-kr')
	columns = code_all[0].columns
	df = code_all[0][[columns[0], columns[2]]]
	df.columns=['names', 'code']
	df['change_price'] = np.nan
	return df




def data_from_csv(fn):
	#file_name = './data/'+'stock_name_code_ChangePrice.csv'
	df = pd.read_csv(fn, encoding='cp949')
	df = df.iloc[:, 1:]
	df.columns=['names', 'code', 'change_price']
	return df




def orderly(df) :
    df['change_price'] = pd.to_numeric(df['change_price'], errors='coerce') # 문자열이라 비교해서 정렬 못할 수 있으니까
    df_10=df.sort_values(by='change_price',ascending=False)[:10]
    df_10z=df_10['code'].copy()
    newCodeList=[]
    for i in df_10z:
        code=i.zfill(6)
        newCodeList.append(code)
    df_10['code']=newCodeList
    newCode_df_10 =df_10
    return newCodeList , newCode_df_10

#codelist, df_10 = orderly(df)   
#print(df_10)
#print(codelist)



def get_url_list(codelist):
    urlList=[]
    for i in codelist:
        url=("https://finance.naver.com/item/main.nhn?code=" + str(i))
        urlList.append(url)
    return urlList
    
#urlList = get_url_list(codelist)

def get_soup_list(urlList):
    resp_10List=[]
    for i in urlList:
        resp=requests.get(i)
        soup= BS(resp.text,'html.parser')
        resp_10List.append(soup)
    return resp_10List
        
#soups = get_soup_list(urlList) # 10 soup in soups 길이 10개 , 한개의 soup 길이는 5개




def extract_data(soups):
    all_table = []
    for num, soup in enumerate(soups):
        section = soup.select_one('.section.cop_analysis') 
        #**select()**는 결과가 하나든 여러 개든 무조건 **리스트(봉투)**에 담아서 줍니다.
        #그러니 리스트는 꾸러미일 뿐이라 그자체로는 다시 select_one을 사용할 능력이 없으니 
        #리스트가아닌 태그(알맹이)를 건내주는 select_one으로 주소를 찾아주자.  
        if section:
            table_tag =section.select_one("table")
        if table_tag:
            df_temp = pd.read_html(StringIO(str(table_tag)))
            df_0415client_page = df_temp[0]
            print(f"{num} 번째 종목 처리 중")
            all_table.append(df_0415client_page)
                
    return all_table

#all_table = extract_data(soups)
#print(f"현재 {len(all_table)}개의 표가 리스트에 안전하게 보관되어 있습니다.")




def df_gets_names(df_10):
    for i in df_10:
        Sri_10_name =df_10.loc[:,'names'].copy()
        Sri_10_change_price =df_10.loc[:,'change_price'].copy()
        df_10names = pd.DataFrame()
        df_10change_price = pd.DataFrame()
        df_10names['10names'] = Sri_10_name
        df_10change_price['change_price'] = Sri_10_change_price
        df_10names = df_10names.reset_index(drop=True)
        df_10change_price = df_10change_price.reset_index(drop=True)
        print('고유번호(행 인덱스)를 지우고 기본행인덱스가 생성되었습니다')
    return df_10names , df_10change_price
#df_10names , df_10change_price =df_gets_names(df_10)
#%% 확인 출력


def change_soups_Dframes(all_table,para_df_10):
    df_10names , df_10change_price = df_gets_names(para_df_10)
    names_list = df_10names['10names'].tolist() # 반복문을 위해 리스트로 변환
    change_pirce_list = df_10change_price['change_price'].tolist() # 반복문을 위해 리스트로 변환
    dict_DF10 = {}
    for arg_table, name , change_price in zip(all_table, names_list,change_pirce_list):
        temp_df = arg_table.copy()
        temp_df = temp_df.reset_index(drop=True)
        # print('숫자 열인덱스를 지웠습니다')
        multi_colName = temp_df.columns[0]
        temp_df=temp_df.set_index(multi_colName)
        
        temp_df = temp_df.T.copy()
        temp_df.index = temp_df.index.get_level_values(1)
        
        
        temp_df.index.name = None 
        # .loc['새로운인덱스명'] = 값  형태를 사용하면 맨 아래에 한 줄이 추가됩니다.
        temp_df.loc['2026.04.15(chagne_price)'] = change_price
        stick_colname = list(temp_df.columns)
        temp_df.columns = stick_colname
       
        dict_DF10[name] = temp_df
     
    return dict_DF10
    
#dict_DF10 =change_soups_Dframes(all_table, df_10)    





import re
def date_transform(dict_name_vs_df):
    final_dict={}
    for name, df_10 in dict_name_vs_df.items(): 
        ratify_indexlist =list(df_10.index)
        new_index=[]
        row_date = ratify_indexlist[:-1]
        fix_val_date = ratify_indexlist[-1] #2026.04.15(change_price)
        for i in row_date:
            str_date=str(i)
            apart_ofdate = str_date.split('.')
            year = int(re.sub(r'[^0-9]', '',apart_ofdate[0]))
            month = int(re.sub(r'[^0-9]','' ,apart_ofdate[1]))
            _, last_day = calendar.monthrange(year, month)
            new_index.append(date(year, month, last_day))
        if len(new_index) == len(row_date):
            new_index.append(fix_val_date)
        # print(new_index)
        df_10.index = new_index
        final_dict[name] = df_10
    return final_dict
#final_dict = date_transform(dict_DF10)
#print(final_dict)






def close_and_before_val_extract_fromNaver(soups):
    catch_close={}
    catch_before={}
    val=[]
    bf=[]
    
    for i, soup in enumerate(soups):
        tag = soup.select_one('#rate_info_krx')
        if tag:
            close_val =tag.select_one('span')
        if close_val:
            close=(close_val.get_text().replace(',', ''))
            val.append(close)
        catch_close['Close'] = val   
     
    for j, soup in enumerate(soups):        
        tg = soup.select_one('#rate_info_krx.rate_info')
        if tg :
            tagg = tg.select_one('.first')
        if tagg:
            fv=tagg.select_one('.blind')
            bval=(fv.get_text().replace(',', ''))
            bf.append(bval)
        catch_before['the_day_before'] = bf
    
    return catch_before, catch_close
            
#catch_before, catch_close  = close_and_before_val_extract_fromNaver(soups)
#print(catch_before)
#print(catch_close)







def get_framework(final_dict, catch_before, catch_close):
    #catch_before, catch_close  = close_and_before_val_extract_fromNaver(soups)
    fine_dict={}
    fin_dictVal = []
    for num, (name, i) in enumerate(final_dict.items()):
        work_df = i[['매출액', '영업이익', '당기순이익','영업이익률', '순이익률', 'EPS(원)','PBR(배)']].copy()
        work_df.columns =  ['매출액(억 원)', '영업이익(억 원)', '당기순이익(억 원)','영업이익률(%)', '순이익률(%)', 'EPS(원)','PBR(배)']
        work_df.loc['the_day_before'] = catch_before['the_day_before'][num]
        work_df.loc['Close'] = catch_close['Close'][num]
        # .loc['새로운인덱스명'] = 값  형태를 사용하면 맨 아래에 한 줄이 추가됩니다.
        fine_dict[name] = work_df
   
    return fine_dict


#fine_dict =get_framework(final_dict , catch_before, catch_close)
#print(type(fine_dict))
#print(fine_dict.keys())
#print(fine_dict)



def plot_data(fine_dict):
    for name, df_content in fine_dict.items():
        company = name
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







##% ------------------------------------------------------------------------------------------------------





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


#%%
import pandas as pd
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates




# 1. 현재 파일(project1_desc.py)의 절대 경로를 가져옵니다.
current_file_path = os.path.abspath(__file__)

# 2. utils 폴더에서 한 단계 위인 프로젝트 루트(span 또는 stock_ML_DL)로 올라갑니다.
# os.path.dirname을 두 번 쓰면 한 단계 위로 올라갑니다.
base_dir = os.path.dirname(os.path.dirname(current_file_path))

# 3. 루트 폴더 아래에 있는 'data' 폴더 경로를 생성합니다.
data_dir = os.path.join(base_dir, "data")

# 4. 해당 폴더에서 OBS_ASOS_MI_로 시작하는 모든 csv 파일을 찾습니다.
file_paths = glob.glob(os.path.join(data_dir, "OBS_ASOS_MI_*.csv"))

# [확인용 출력] 서버 로그(Manage app)에서 이 숫자가 0이 아닌지 꼭 확인하세요!
print(f"--- 프로젝트 루트 경로: {base_dir} ---")
print(f"--- 데이터 폴더 경로: {data_dir} ---")
print(f"--- 찾은 파일 개수: {len(file_paths)} ---")

def files_list(file_paths):
    if not file_paths:
        return []
    
    file_lists = []
    for i in file_paths:
        # 파일 읽기 (인코딩 주의: cp949)
        dfs = pd.read_csv(i, parse_dates=[0], index_col=[0], encoding='cp949')
        file_lists.append(dfs)
    return file_lists

# 데이터 리스트 생성
file_lists = files_list(file_paths)
dfs = file_lists.copy()



def dfs_go_dict(dfs):
    dicts={}
    for i ,contents in enumerate(dfs):
        contents = pd.DataFrame(contents)
        dicts[i] = contents
    return dicts

dicts=dfs_go_dict(dfs)


def concatenates_dicts(dicts):
    tot_dfs = pd.concat(list(dicts.values()), axis=0)
    return tot_dfs

tot_dfs = concatenates_dicts(dicts)



def check_forward(tot_dfs):
    df = tot_dfs.copy()
    df.isna().sum().sum() 
    df[df['기온(°C)'].isnull()]
    df.reset_index(drop=True)
    df.set_index('일시', inplace = True)
    df.index = pd.to_datetime(df.index)
    df_work = df.copy()
    return df_work

df_work=check_forward(tot_dfs)


#print( 'df_work' ,df_work)




def diff_abs_work(df_work):

    df_work['기온변화량(°C)'] = df_work['기온(°C)'].diff()
    # 1. '현재 기온 - 1분 전 기온' 계산 (차이값 구하기)
    df_work['기온변화량(°C)'] = df_work['기온(°C)'].diff()
    df_work.loc[df_work['기온변화량(°C)'] >3 , '기온(°C)' ] = np.nan
    # 2. 절댓값을 씌워 변화의 크기만 확인
    df_work['변화절댓값'] = df_work['기온변화량(°C)'].abs()
    df_work.index=sorted(df_work.index)
    print('날짜인덱스확인',df_work.index)
    # print(df_work)
    print(len(df_work))
    df_works = df_work.copy()
    return df_works

df_works=diff_abs_work(df_work)
#print(df_works)



def check_forward2(df_works):

    df_cal=df_works['변화절댓값']
    df_cal.unique()
    df_cal=df_cal.reset_index()
    df_cal['step_sum_60minutes'] = df_cal['변화절댓값'].rolling(window=60).sum()
    df_cal['min_out'] = df_cal['step_sum_60minutes'] < 0.1

    out_periods = df_cal[df_cal['min_out'] == True]
 
    df_cal=df_cal.set_index('index')
    df_works = df_works.drop(['변화절댓값'], axis=1)
    #df_work ['지점명', '기온(°C)', '기온변화량(°C)', '변화절댓값']
    #df_cal ['변화절댓값', 'step_sum_60minutes', 'min_out'] 
    
    df_work = pd.concat([df_works, df_cal] ,axis=1).copy()
    df_work = df_work.drop('min_out', axis=1)
    df_station = df_work['지점명']
    df_work = df_work.drop('지점명', axis=1)
    
    return df_work ,out_periods.index,df_station

    
df_work ,out_periods_num_idx,df_station =check_forward2(df_works)







def carry_out_nan(df_work,out_periods_num_idx,df_station):
    con_num = out_periods_num_idx
    con_num[0] 
    con_num[-1] 
    df_work=df_work.reset_index()
    df_work.iloc[con_num[0] : con_num[-1] , :  ] = np.nan # ' min_out' == True 처리
    nan_rows = df_work[df_work['기온변화량(°C)'].isna()].copy()
    print(nan_rows.index)
    df_work=df_work.set_index('index')
    d_data = df_work['기온(°C)'].dropna().resample('h').agg(['mean', 'size' ]) 
    # Date 하루 중 쌓인 데이터의 size 와 mean 을 테이블로 만들 겠따
    d_data.columns = ['평균기온(°C)' , '데이터개수']
    # d_data['데이터개수'].unique()#Out[347]: array([58, 60, 43,  1])
    # q = d_data['데이터개수'] == 43
    # len(q)#97개 
    d_data.loc[d_data['데이터개수'] <2 ,  : '평균기온(°C)'    ] =np.nan 
    d_data.loc[d_data['데이터개수'] <48 ,  : '평균기온(°C)'    ] =np.nan 
    nan_check = d_data[d_data['평균기온(°C)'].isna()]
    # nan_check.index
    print(f"결측 처리된 시간대 수: {len(nan_check)}개")
    d_data['지점명'] = df_station
    return d_data

d_data  =carry_out_nan(df_work,out_periods_num_idx,df_station)


#%% graph


plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
plt.rcParams['axes.unicode_minus'] = False


def data_mean_plot(d_data):
   
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(d_data.index , d_data['평균기온(°C)' ], label='평균기온(°C)')
    ax.legend()
    return fig


def apply_interpolate_step(d_data):
    methods = ['linear' , 'quadratic' , 'cubic']
    target = d_data['평균기온(°C)' ]

    days_df = pd.DataFrame({ m: target.interpolate(method=m) for m in methods  } , index = target.index )
    return days_df

days_df = apply_interpolate_step(d_data)





def method_plots(days_df):
    fig, ax = plt.subplots(figsize=(20, 10)) 
   
    ax.plot(days_df.index, days_df['linear'], label='linear',color='red', linestyle = '-', linewidth=2)
    ax.plot(days_df.index, days_df['quadratic'], label='quadratic',color='black',linewidth=2, linestyle='--' )
    ax.plot(days_df.index, days_df['cubic'], label='cubic', color='yellow', linewidth=4 , linestyle = '-.', alpha=0.4)
    
    
    # 6시간 간격으로 눈금 위치 잡기
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    
    # 날짜 표시 형식 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # 그리드 
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_xlabel('2024', fontsize=12)
    
    # rotation
    plt.xticks(rotation=45, ha='right')
    
    
    ax.legend(fontsize=12)
    ax.set_title(' comparison of interpolation method', fontsize=15)
    
    
    plt.tight_layout()
    return fig











#%%d_data Source
targets = [ 'interpolates_linear', 'interpolates_quadratic','interpolates_cubic']

def d_data_source(d_data, m):
    d_data['interpolates_linear'] = d_data['평균기온(°C)'].interpolate(method='linear')
    d_data['interpolates_quadratic'] = d_data['평균기온(°C)'].interpolate(method='quadratic')
    d_data['interpolates_cubic'] = d_data['평균기온(°C)'].interpolate(method='cubic')
    #d_data.columns

    nan_check = d_data[d_data['평균기온(°C)'].isna()].index
    
    
  
        
    fig, ax = plt.subplots(figsize=(20, 10)) 
    
    ax.plot(d_data.index, d_data[m], label=m, color='blue', alpha=0.8 )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))     
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    ax.scatter(nan_check, d_data.loc[nan_check, m], 
                   color='red', s=100, zorder=5, label='Interpolated Points')
    for x in nan_check:
        y = d_data.loc[x, m]
        ax.text(x, y + 0.2, f'{y:.2f}', color='red', fontweight='bold', ha='center')
            
    ax.set_title(f'Interpolation Method: {m}',fontsize=18, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    #fig_list.append(fig)
    plt.tight_layout()
    return fig

# 메인에서 직접 루프를 돌며 "그려와!"라고 시킴 # 받은 즉시 화면에 출력





