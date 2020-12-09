import streamlit as st
import pandas as pd
import json

import ast
import pybase64

from annotation.database import load_data

##### FUNCTIONS FOR Downloading The Data #######
def get_table_download_link(df):
            csv = df.to_csv(index=False)
            b64 = pybase64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'

def get_database(start,end, news, engine):
    database = []
    p_bar = st.progress(0)
    for y in range(start,end+1):
        d = load_data(y, news, engine)
        database.append(d)
        p_bar.progress(y/(end+1-start))
    database = pd.DataFrame(database)
    return database

#### Download Data Page ####
def download_data_page(news, engine):
    start = st.number_input(label='Start index',value=1 ,min_value=1,max_value=10000 ,step=1)
    end = st.number_input(label='Start index',value=10 ,min_value=2,max_value=10000 ,step=1)
    if st.button(label='Produce Data',key='producedata'):
        database = get_database(start,end, news, engine)
        st.markdown(get_table_download_link(database), unsafe_allow_html=True)

    if st.button('test'):
        database = []
        for y in range(start,end+1):
            d = load_data(y, news, engine)
            database.append(d)
        #st.write(database)
        with open ('C:/Users/alisobhani/Desktop/crime paper/city_crime/annot_data/data3.json','w') as f:
            json.dump(database,f)
        st.write ('Done!')