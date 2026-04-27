import streamlit as st
from utils import project1_desc as p1d
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy import stats

def app():
    st.write('''
    ## streamlit으로 dataframe을 가져와보자     
    ''')

   
    
    url='http://kind.krx.co.kr/corpgeneral/corpList.do?method=download'
    df1=p1d.get_code_name(url)
    st.dataframe(df1)

    fn = 'C:\\Users\\user\\Documents\\stock_ML_DL\\stock_ML_DL\\data\\stock_name_code_ChangePrice.csv'
    
    df=p1d.data_from_csv(fn)
    st.dataframe(df)
    
    codelist, df_10  = p1d.orderly(df)   
    
    st.dataframe(df_10)
    #st.markdown(f"리스트 값: {codelist}")
    #st.text(codelist)

    urlList = p1d.get_url_list(codelist)

    soups =p1d.get_soup_list(urlList)

    catch_before, catch_close  = p1d.close_and_before_val_extract_fromNaver(soups)    
    all_table=p1d.extract_data(soups)
    
    
    

    df_10names , df_10change_price =p1d.df_gets_names(df_10)
    
    dict_DF10 =p1d.change_soups_Dframes(all_table, df_10)  
  
    final_dict = p1d.date_transform(dict_DF10)




    fine_dict =p1d.get_framework(final_dict, catch_before, catch_close)
    
        
    for i ,j in fine_dict.items():
        st.write(i)
        st.dataframe(j)

    
    base_dir =r"C:\Users\user\OneDrive\Desktop\퀀트 1\drive-download-20260427T041452Z-3-001"
    file_paths = glob.glob(os.path.join(base_dir,"*.csv"))


    file_lists = p1d.files_list(file_paths)
    dfs = file_lists.copy()

    
    dicts=p1d.dfs_go_dict(dfs)

    tot_dfs = p1d.concatenates_dicts(dicts)

    df_work=p1d.check_forward(tot_dfs)

    df_works=p1d.diff_abs_work(df_work)
    # df_work ,out_periods =p1d.check_forward2(df_works)
    df_work ,out_periods_num_idx,df_station =p1d.check_forward2(df_works)


    d_data  =p1d.carry_out_nan(df_work,out_periods_num_idx,df_station)


    #plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
    #plt.rcParams['axes.unicode_minus'] = False

    fig=p1d.data_mean_plot(d_data)
    st.pyplot(fig)    

    days_df = p1d.apply_interpolate_step(d_data)

    fig2=p1d.method_plots(days_df)
    st.pyplot(fig2)
    #모듈이 준 결과물을 st.pyplot(), st.write(), st.table()

    targets = [ 'interpolates_linear', 'interpolates_quadratic','interpolates_cubic']
    for m in targets:
        fig = p1d.d_data_source(d_data,m)
        st.pyplot(fig) 



