import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

import torch
from transformers import BertTokenizer, BertModel

from scipy.spatial.distance import cosine

class ExtractiveSummarizer:
    def __init__ (self):
        self.get_tokenizer()
        self.get_model()

    def get_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_model(self):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()

    def summarize(self,text, title, summarize_title=True, summarize_centroid=False):
        doc = Document(text, title, self.tokenizer, self.model)
        if summarize_title == True:
            doc.summarize_title()
        elif summarize_centroid ==True:
            doc.summarize_centroid ()
        return doc

class Document:
    def __init__(self, text, title, tokenizer, model):
        self.text = text
        self.title = title
        self.tokenizer = tokenizer
        self.model = model
        self.sent_list = sent_tokenize(self.text)
        sents = []
        for i , sent in enumerate(self.sent_list):
            s = Sentence(sent, self.tokenizer, self.model)
            s.sentence_vectorizer()
            s.index = i
            sents.append(s)
        self.sent_list = sents
        title = Sentence(self.title, self.tokenizer, self.model)
        title.sentence_vectorizer()
        self.title = title
    
    def centroid (self):
        centroid=[]
        for i in range(len(self.sent_list[0].embedding)):
            c = 0
            for sent in self.sent_list:
                c += sent.embedding[i]
            c=c/len(self.sent_list[0].embedding)
            centroid.append(c)
        return np.array(centroid)  

    def summarize_title(self):
        for sent in self.sent_list:
            similarity = 1 - cosine(self.title.embedding, sent.embedding)
            sent.similarity = similarity
        ranked_list = sorted(self.sent_list,key=lambda x: x.similarity, reverse=True)
        summary = sorted(ranked_list[0:6],key=lambda x: x.index)
        self.summary_title = summary
        self.summary_index_title = [sum.index for sum in ranked_list[0:6]]
        return ' '.join([sum.sent for sum in summary])

    def summarize_centroid(self):
        centroid = self.centroid()
        for sent in self.sent_list:
            similarity = 1 - cosine(centroid, sent.embedding)
            sent.similarity = similarity
        ranked_list = sorted(self.sent_list,key=lambda x: x.similarity, reverse=True)
        summary = sorted(ranked_list[0:6],key=lambda x: x.index)
        self.summary_centroid = summary
        self.summary_index_centroid = [sum.index for sum in ranked_list[0:6]]
        return ' '.join([sum.sent for sum in summary])

    def plot(self):
        vectors = [np.array(sent.embedding) for sent in self.sent_list]
        vectors.append(np.array(self.title.embedding))
        vectors.append(self.centroid())
        vectors = np.array(vectors)
        twodim = PCA(n_components=2).fit_transform(vectors)

        plt.figure(figsize=(10,10))
        plt.scatter(twodim[:,0], twodim[:,1],edgecolors='k', c='r')
        index_list = [sent.index for sent in self.sent_list]
        index_list += ['title' , 'center']
        for index, (x,y) in zip(index_list, twodim):
            plt.text(x+0.05, y+0.05, index)
  
class Sentence:
    def __init__(self,sent, tokenizer, model):
        self.sent = sent
        self.tokenizer = tokenizer
        self.model = model
        
    def sentence_vectorizer(self):
        marked_text = "[CLS] " + self.sent + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        self.embedding = sentence_embedding


