##### FUNCTIONS FOR STYLE ######
def text_box(txt):
    txt = f"<span style='padding:5px;border-radius:3px;background-color:#f63366;color:white;font-size:13px'>{txt}</span>"
    return txt
def text_boarder(txt):
    txt = f"<span style='border:1px #f63366 solid;padding:2px;border-radius:3px;'>{txt}</span>"
    return txt  
def clean_t(r):
    r = str(r)
    r = r.replace("'","")
    r = r.replace("[","")
    r = r.replace("]","")
    return r
def ner_tag(tag):
    tags = ['ct','cz','org','st','ev','p']
    tag_out = ['CITY','CITIZENS','ORGANIZATION','STATE','EVENT','PERSON']
    t=''
    try:
        t = tag_out[tags.index(tag)]
    except:
        pass
    return t
def text_ner(txt):
    txt = txt.split("_")
    txt_out=''
    for t in txt:
        if ":" in t:
            t = t.split(":")
            t = "<span style='border:1px #f63366 solid;padding:3px;border-radius:3px;font-weight: 500;'>"+t[0]+"</span>"+"<span style='padding:2px;border-radius:3px;background-color:#fffd80;color:grey;font-size:13px;margin-left:5px;font-weight: 400;'>"+ner_tag(t[1])+"</span>"
        txt_out = txt_out + t
    return txt_out