import streamlit as st
import pandas as pd
import numpy as np
import ast

from annotation.database import initiate_database, import_data, update_data , tags_read, tags_write, city_read, update_data_city
from annotation.style import clean_t, text_box
from annotation.annotation_page import annotation_page
from annotation.download_data import download_data_page
from annotation.import_news_page import import_news_page
from annotation.filter_page import filter_page
from extractive_summarization.summarizer_page import summarizer_page

engine, db, meta, news = initiate_database()

@st.cache(allow_output_mutation=True)
def __init__():
    pass

page = st.sidebar.selectbox('Page',['Annotation','Import News','Filter','Data', 'Summarizer'],key='page')

if page == 'Annotation':
    annotation_page(news, engine)

if page == 'Import News':
    import_news_page(news, engine)

if page == 'Filter':
    filter_page(news, engine)

if page == 'Data':
    download_data_page(news, engine)

if page == 'Summarizer':
    summarizer_page (news, engine)