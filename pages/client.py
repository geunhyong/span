import streamlit as st
from utils import StockAnalysis as MS


def app():
    st.write('''왜 문제냐 (Streamlit 관점) Streamlit은 보통 이렇게 동작합니다. def app() 함수정의 내의 scope 에서 처리 가능, 
    모든 UI + 데이터 처리 여기 안에 있어야 합니다. 운영체제 윈도우는 한글깨짐 방지 순서 cp949, euc-kr ,utf-8 인데 보통 마지막인 
    utf-8d을 기본으로 생략하여 읽기에 한번씩 UnicodeDecodeError가 나면 encoding을 바꾸어 봅시다. 아니면 engine 아니면 import''')


    fn = 'C:\\Users\\user\\Documents\\stock_ML_DL\\stock_ML_DL\\data\\stock_name_code_ChangePrice.csv'
    df = MS.data_from_csv(fn)

    df_dict_eps, df_10 = MS.get_eps(df)

    df_trans = MS.get_date_timestamp(df_dict_eps)

    #df_trans_stock = MS.get_stock_data(df_trans)

    #MS.plots(df_trans_stock)
