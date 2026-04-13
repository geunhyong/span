import streamlit as st
from pages import project1 as p1
from pages import project2 as p2
from pages import intro

st.title('글쓰기공간')

item_list = ['item0','item1', 'item2']

item_labels = {'item0':'소개하는', 'item1':'연습하는', 'item2':'메모하는'}

FIL = lambda x : item_labels[x]
item = st.sidebar.selectbox('항목을 골라요.',  item_list, format_func = FIL )

if item == 'item1':
    p1.app()
elif item == 'item2':
    p2.app()
elif item == 'item0':
    intro.app()
