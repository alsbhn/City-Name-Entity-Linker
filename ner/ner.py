import pandas as pd
import json
from sklearn.utils import shuffle
from transformers import BertTokenizer

from train import SentencePairBertClassifier

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open ('..//annot_data/train_data.json') as f:
  df = json.load (f)
  df = pd.DataFrame(df)
  df = df[df['description']!='']
  df = shuffle(df)

train_set, test_set = split_train_test(df)

sentences_1 , sentences_2 = train_set.description.values , train_set.summary_title.values
labels = train_set.ner.values



# Prepare for trainig
epochs = 3

spc = SentencePairBertClassifier()
spc.train(train_dataset, val_dataset, epochs)






