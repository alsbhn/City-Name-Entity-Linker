from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import numpy as np
import time
import datetime

def build_data_loader(batch_size = 8):
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )
  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )

def build_model(num_labels = 2):
  model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels = num_labels, # The number of output labels--2 for binary classification.   
    output_attentions = False, output_hidden_states = False, )
  model.cuda()

def build_optimizer(lr = 2e-5,eps = 1e-8):
  optimizer = AdamW(model.parameters(), lr = lr, eps = eps)
