import streamlit as st
from utils import project2_desc as p2d


def app():
	st.write('''
		낙관적인 말 : 내면의 힘이 생기고 성숙하면 스스로를 해치지 않게 된다. 
		자신에게도 긍정적인 사람이 될 수 있기를 감사일기에 써보자.
		'''
		)
	p2d.desc()
