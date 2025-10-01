import streamlit as st

hw1=st.Page("./homework/HW1.py",title="Home Work 1")
hw2=st.Page("./homework/HW2.py",title="Home Work 2")
hw3=st.Page("./homework/HW3.py", title="Home Work 3")
hw4=st.Page("./homework/HW4.py", title="Home Work 4")
hw5=st.Page("./homework/HW5.py", title="Home Work 5")
pg=st.navigation([hw5,hw4,hw1,hw2,hw3])
st.set_page_config(page_title="Multi page app",)
pg.run()