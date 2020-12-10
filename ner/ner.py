import pandas as pd
import json
from sklearn.utils import shuffle
from transformers import BertTokenizer
from tokenizer import split_train_test, tokenize_dataset
from torch.utils.data import TensorDataset, random_split
from train import *

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

class SentencePairBertClassifier:
  
  def __init__(self, ):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if torch.cuda.is_available():       
        self.device = torch.device("cuda")
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        self.device = torch.device("cpu")

    self.train_dataloader, self.validation_dataloader = build_data_loader(train_dataset, val_dataset, batch_size = 8)
    self.model = build_model(num_labels = 2)
    self.optimizer = build_optimizer(model, lr = 2e-5,eps = 1e-8)
    self.scheduler = build_scheduler (self.train_dataloader, optimizer , epochs)

  @classmethod
  def train(epochs):

  def train(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs):
 
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
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
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            self.model.zero_grad()        
            output = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss , logits = output.loss, output.logits 
            
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
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
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
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
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)   
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    #return model, training_stats

model, training_stats = train(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs)






