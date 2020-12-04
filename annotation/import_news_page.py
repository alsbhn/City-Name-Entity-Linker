import streamlit as st
from newspaper import Article
from annotation.database import get_urls, tags_read, city_read, insert_data

def import_news_page(news, engine):
    url = st.text_input('url')
    tag_list = tags_read()
    city_list = city_read()
    # label
    labels = st.sidebar.multiselect('labels', tag_list,key=1)
    # city
    annot_city = st.sidebar.multiselect('city', city_list,key=2)
    # dataset type
    dataset_type = st.sidebar.multiselect('dataset', ['manual_google'],key=3)
    
    if st.checkbox ('Scrape'):
        
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title
        message = st.empty()
        st.subheader(title)
        st.write(text)
        

    if st.sidebar.button ('Import'):
        urls = get_urls(news,engine)
        if url not in urls:
            insert_data(url, text, title, str(dataset_type), str(labels), str(annot_city), news, engine)
            message.info('Data Added')
        else:
            message.warning('Repetative Data')