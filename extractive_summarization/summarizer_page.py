import streamlit as st
from extractive_summarization.summarizer import ExtractiveSummarizer
from annotation.database import load_data

def summarizer_page(news, engine):
    ext_summarizer = ExtractiveSummarizer()

    x = st.sidebar.number_input(label='News ID',value=1 ,min_value=1,max_value=8707 ,step=1)
    data = load_data(x, news, engine)
    text = data['text']
    title = data['title']

    if st.button('Summarize'):
        doc = ext_summarizer.summarize(text, title)
        st.subheader('Summary')
        st.write(doc.summarize_title())
        #st.write(doc.summary_centroid)
        st.subheader(title)
        st.write(text)