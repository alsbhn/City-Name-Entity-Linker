import streamlit as st
import pandas as pd
import numpy as np
import ast

from annotation.database import initiate_database, import_data, update_data , tags_read, tags_write, city_read, update_data_city
from annotation.style import clean_t, text_box

engine, db, meta, news = initiate_database()

@st.cache(allow_output_mutation=True)
def load_data(x):
    s = news.select(news.c.id == x)
    conn = engine.connect()
    result = conn.execute(s)
    for row in result:
        data = row
    (id,city,url,text,title,dataset,labels,annot_city)=data
    data = {'id':id,'city':city,'url':url,'text':text,'title':title,'dataset':dataset,'labels':labels,'annot_city':annot_city}
    conn.close()
    return data

# import data
#if st.button('Import Data'):
#    import_data()

x = st.sidebar.number_input(label='News ID',value=1 ,min_value=1,max_value=8707 ,step=1)
data = load_data(x)
tag_list = tags_read()
city_list = city_read()

st.subheader(data['title'])
st.markdown(data['city'])
st.write(data['url'])
st.write(data['dataset'])



# label
labels = st.sidebar.multiselect('labels', tag_list,key=1,default= ast.literal_eval(data['labels']))
if st.sidebar.button ('Update labels'):
    update_data(x, str(labels), news, engine)

# city
annot_city = st.sidebar.multiselect('city', city_list,key=2, default= ast.literal_eval(data['annot_city']))
if st.sidebar.button ('Update city'):
    update_data_city(x, str(annot_city), news, engine)

st.markdown(f"{text_box(clean_t(data['labels']))}", unsafe_allow_html=True)
st.markdown(f"{text_box(clean_t(data['annot_city']))}", unsafe_allow_html=True)

st.write(data['text'])
    
new_tag = st.sidebar.text_input("New tag")
if st.sidebar.button('Update tags'):
    tag_list.append(new_tag)
    tags_write(tag_list)