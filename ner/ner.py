import pandas as pd
import json
from sklearn.utils import shuffle
from transformers import BertTokenizer
from tokenizer import split_train_test, tokenize_dataset
from torch.utils.data import TensorDataset, random_split
from train import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open ('/content/city_crime/annot_data/train_data.json') as f:
  df = json.load (f)
  df = pd.DataFrame(df)
  df = df[df['description']!='']
  df = shuffle(df)

train_set, test_set = split_train_test(df)

sentences_1 , sentences_2 = train_set.description.values , train_set.summary_title.values
labels = train_set.ner.values

input_ids, attention_masks, labels = tokenize_dataset(sentences_1 , sentences_2, labels, 512, tokenizer)

# Divide up our training set to use 90% for training and 10% for validation.
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Prepare for trainig
epochs = 10

train_dataloader, validation_dataloader = build_data_loader(train_dataset, val_dataset, batch_size = 8)
model = build_model(num_labels = 2)
optimizer = build_optimizer(model, lr = 2e-5,eps = 1e-8)
scheduler = build_scheduler (train_dataloader, optimizer , epochs)

model, training_stats = train(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs)






