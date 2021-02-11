import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from tokenizer import *

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer

import numpy as np
import time
import datetime
import random
import os
import json

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def build_data_loader(train_dataset, val_dataset, batch_size = 8):
  train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
  validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)
  return train_dataloader, validation_dataloader

def build_model(device, num_labels = 2):
  model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels = num_labels, # The number of output labels--2 for binary classification.   
    output_attentions = False, output_hidden_states = False, )
  
  if device.type == 'cuda':
    model.cuda()
  return model

def build_optimizer(model, lr = 2e-5,eps = 1e-8):
  optimizer = AdamW(model.parameters(), lr = lr, eps = eps)
  return optimizer

def build_scheduler (train_dataloader, optimizer , epochs):
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

class SentencePairBertClassifier:
  
  def __init__(self):
    
    if torch.cuda.is_available():       
        self.device = torch.device("cuda")
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        self.device = torch.device("cpu")
    
    self.model = build_model(self.device, num_labels = 2)

  @classmethod
  def load_from_pretrained(cls, pretrained_model_path):
    classifier = cls()
    classifier.model = BertForSequenceClassification.from_pretrained(pretrained_model_path)
    classifier.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    
    if classifier.device.type == 'cuda': # Copy the model to the GPU.
      classifier.model.cuda()
    return classifier

  def train(self, sentences_1 , sentences_2, labels,start_epoch, epochs,save_in_loop, save_folder, save_model_name, preprocess = False):
    
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    if preprocess:
      input_ids, attention_masks, labels = tokenize_dataset(sentences_1 , sentences_2, labels, self.tokenizer, max_length=512)
      dataset = TensorDataset(input_ids, attention_masks, labels)
      train_size = int(0.9 * len(dataset))
      val_size = len(dataset) - train_size
      train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
      print('{:>5,} training samples'.format(train_size))
      print('{:>5,} validation samples'.format(val_size))

      #with open (os.path.join(save_folder, 'train_dataset.json'), 'w') as f:
      #  json.dump({'data': train_dataset},f)
      #with open (os.path.join(save_folder, 'val_dataset.json'), 'w') as f:
      #  json.dump({'data' :val_dataset},f)

    else:
      with open (os.path.join(save_folder, 'train_dataset.json')) as f:
        train_dataset = json.load(f)
        train_dataset = train_dataset ['data']
      with open (os.path.join(save_folder, 'val_dataset.json')) as f:
        val_dataset = json.load(f)
        val_dataset = val_dataset ['data']

    # Divide up our training set to use 90% for training and 10% for validation.
    

    self.train_dataloader, self.validation_dataloader = build_data_loader(train_dataset, val_dataset, batch_size = 8)
    
    self.optimizer = build_optimizer(self.model, lr = 2e-5,eps = 1e-8)
    self.scheduler = build_scheduler (self.train_dataloader, self.optimizer , epochs)

    self.training_stats = []
    total_t0 = time.time()
    for epoch_i in range(start_epoch-1, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.model.zero_grad()        
            output = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss , logits = output.loss, output.logits 
            
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)            
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print("")
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():        
                output = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                (loss, logits) = output.loss, output.logits
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        validation_time = format_time(time.time() - t0)   
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        self.training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        
        # save model
        if save_in_loop:
          print ('saving the model ...')
          self.save_model(save_folder,f"{save_model_name}_{epoch_i + 1}")

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

  def save_model (self, model_path, model_name):

    output_dir = os.path.join(model_path, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    self.tokenizer.save_pretrained(output_dir)

  def test (self, sentences_1 , sentences_2, labels):

    input_ids, attention_masks, labels = tokenize_dataset(sentences_1 , sentences_2, labels, self.tokenizer, max_length=512)
    
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size = 16)

    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    self.model.eval() # Put model in evaluation mode

    predictions , true_labels = [], []
    for batch in prediction_dataloader:
      batch = tuple(t.to(self.device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    print('Test DONE.')
    return flat_true_labels, flat_predictions

  def predict (self, sentences_1 , sentences_2):
    flat_true_labels, flat_predictions = self.test([sentences_1] , [sentences_2], [0])
    return flat_predictions[0]


    