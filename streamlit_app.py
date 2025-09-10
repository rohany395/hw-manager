import streamlit as st

lab1=st.Page("./homework/HW1.py",title="Home Work 1")
lab2=st.Page("./homework/HW2.py",title="Home Work 2")
pg=st.navigation([lab2,lab1])
st.set_page_config(page_title="Multi page app",)
pg.run()