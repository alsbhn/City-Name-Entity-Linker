import torch
from transformers import BertTokenizer 

def split_train_test(df):
  train_set = df[:int(len(df)*0.8)] 
  test_set = df[int(len(df)*0.8):]
  return train_set, test_set

def tokenize_dataset(sentences_1 , sentences_2, labels,tokenizer, max_length=512):
  input_ids = []
  attention_masks = []
  for sent1, sent2 in zip(sentences_1, sentences_2):
      encoded_dict = tokenizer.encode_plus(
                          sent1,sent2,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = max_length,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )  
      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)
  return input_ids, attention_masks, labels