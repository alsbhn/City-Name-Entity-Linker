import streamlit as st
import pandas as pd
import numpy as np


@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('train_data/data.csv')
    return data

data = load_data()
dataset = st.selectbox('Select dataset',['all','manual_google','google_auto','sample'])
if dataset == 'manual_google':
    data_show = data [data['dataset'] == 'manual_google']
elif dataset == 'auto_google':
    data_show = data [data['dataset'] == 'auto_google']
elif dataset == 'sample':
    data_show = data [data['dataset'] == 'sample']
else:
    data_show = data
x = st.number_input('index',step=1,value=1,min_value=1,max_value=len(data_show))

data_show.loc [x,['text']][0]