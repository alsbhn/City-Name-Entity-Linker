import pandas as pd
import ast
import re

# import libraries for working with database
from sqlalchemy import create_engine
from sqlalchemy.sql import select, update
from sqlalchemy import MetaData, Table,Column, Integer, String
from sqlalchemy.orm import scoped_session, sessionmaker

# create or definethe database
def initiate_database():
    engine = create_engine("sqlite:///data.db", echo = True)
    db = scoped_session(sessionmaker(bind=engine))
    # create or define the database table
    meta = MetaData()
    news = Table('news', meta, Column('id', Integer, primary_key=True) , Column('city', String),Column('url', String),
    Column('text', String),Column('title', String),Column('dataset', String), Column('labels',String), Column('annot_city', String))
    #news_filtered = Table('news', meta, Column('id', Integer, primary_key=True) , Column('city', String),Column('url', String),
    #Column('text', String),Column('title', String),Column('dataset', String), Column('labels',String), Column('annot_city', String))
    meta.create_all(engine)
    return engine, db, meta, news

def import_data():
    df = pd.read_csv('train_data/data.csv')
    df = df.to_dict(orient='record')
    for row in df:
        db.execute('INSERT INTO news (city, url, text, title, dataset) VALUES (:city, :url, :text, :title, :dataset)',
        {"city": row['city'], "url": str(row['url']), "text": str(row['text']),"title": str(row['title']),"dataset": str(row['dataset'])})
    db.commit()

def load_data(x, news, engine):
    s = news.select(news.c.id == x)
    conn = engine.connect()
    result = conn.execute(s)
    for row in result:
        data = row
    (id,city,url,text,title,dataset,labels,annot_city)=data
    data = {'id':id,'city':city,'url':url,'text':text,'title':title,'dataset':dataset,'labels':labels,'annot_city':annot_city}
    conn.close()
    return data

def filtered_table(filters , news, engine):
    s = news.select(news.c.labels.contains (filters))
    conn = engine.connect()
    result = conn.execute(s)
    data_out=[]
    for row in result:
        data = row
        (id,city,url,text,title,dataset,labels,annot_city)=data
        data_out.append({'id':id,'city':city,'url':url,'text':text,'title':title,'dataset':dataset,'labels':labels,'annot_city':annot_city})
    conn.close()
    return data_out

def update_data(x,upd, news, engine):
    stmt = news.update().where(news.c.id == x).values(labels= upd)
    conn = engine.connect()
    conn.execute(stmt)
    conn.close()

def update_data_city(x,upd, news, engine):
    stmt = news.update().where(news.c.id == x).values(annot_city= upd)
    conn = engine.connect()
    conn.execute(stmt)
    conn.close()

def insert_data(url, text, title, dataset_type, labels, annot_city, news, engine):
    stmt = news.insert().values(url = url, text= text, title = title, dataset=dataset_type, labels=labels,annot_city=annot_city )
    conn = engine.connect()
    conn.execute(stmt)
    conn.close()

def get_urls (news, engine):
    stmt = select([news.c.url])
    conn = engine.connect()
    result = conn.execute(stmt)
    urls=[]
    for url in result:
        urls.append(url[0])
    conn.close()
    return urls

def tags_read():
    df = pd.read_csv('annotation/tags.csv')
    tags = df.tags.to_list()
    return tags

def tags_write(tags):
    df = pd.DataFrame(tags,columns=['tags'])
    df.to_csv('annotation/tags.csv',index=False)
    
def city_read():
    df = pd.read_csv('annotation/city.csv')
    city_list = df.city.to_list()
    return city_list