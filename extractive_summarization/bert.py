### BERT ###
import torch
from transformers import BertTokenizer, BertModel

def get_bert_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    return tokenizer, model