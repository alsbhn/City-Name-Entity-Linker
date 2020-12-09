
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import numpy as np
import time
import datetime

def build_data_loader(train_dataset, val_dataset, batch_size = 8):
  train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
  validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)
  return train_dataloader, validation_dataloader

def build_model(num_labels = 2):
  model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels = num_labels, # The number of output labels--2 for binary classification.   
    output_attentions = False, output_hidden_states = False, )
  model.cuda()
  return model

def build_optimizer(model, lr = 2e-5,eps = 1e-8):
  optimizer = AdamW(model.parameters(), lr = lr, eps = eps)
  return optimizer

def build_scheduler (train_dataloader, optimizer , epochs = 15):
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
  return scheduler

# Define a helper function for calculating accuracy.
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Helper function for formatting elapsed times as hh:mm:ss
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))