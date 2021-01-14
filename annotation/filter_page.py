import streamlit as st
import ast

from annotation.database import initiate_database, import_data, load_data, update_data , tags_read, tags_write, city_read, update_data_city, filtered_table, filtered_table_city
from annotation.style import clean_t, text_box

def filter_page(news, engine):
    tag_list = tags_read()
    city_list = city_read()
    
    filter_by = st.radio ('Filter by:',('label','city'))
    
    if filter_by == 'label':
        filters = st.multiselect('Filters', tag_list)
    elif filter_by == 'city':
        filters = st.multiselect('Filters', city_list)
    
    if st.checkbox('filter'):
        if filter_by == 'label':
            filtered_data = filtered_table(f"'{filters[0]}'" ,news, engine)
        elif filter_by == 'city':
            filtered_data = filtered_table_city(f"'{filters[0]}'" ,news, engine)
        
        y = st.sidebar.number_input(label='Number',value=0 ,min_value=0,max_value=len(filtered_data) ,step=1)
        st.write('Total number: ', len(filtered_data))
        x = filtered_data[y]['id']
        data = load_data(x, news, engine)
        
        st.subheader(data['title'])
        st.markdown(data['city'])
        st.write('id: ', x)
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