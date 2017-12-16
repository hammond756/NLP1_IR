
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import os
import numpy as np
import json
import h5py
import heapq
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from pathlib import Path
import time

import sys
import pprint
from collections import Counter,defaultdict
from itertools import chain

class DialogDataset(Dataset):
    
    def __init__(self, json_data, image_features, img2feat, transform=None):
        
        with open(img2feat, 'r') as f:
            self.img2feat = json.load(f)['IR_imgid2id']
            
        self.img_features = np.asarray(h5py.File(image_features, 'r')['img_features'])
        self.json_data = pd.read_json(json_data, orient='index')
        self.corpus = self.get_words()
        self.vocab = list(set(self.corpus))
        self.w2i = {word : i for i, word in enumerate(self.vocab)}
        
    # collect all the words from dialogs and 
    # captions and use them to create embedding map
    def get_words(self):
        words = []
        for idx in range(len(self)):
            item = self.json_data.iloc[idx]

            # Flatten dialog and add caption into 1d array
            dialog = [word for line in item.dialog for word in line[0].split()]
            dialog.extend(item.caption.split(' '))

            words.append(dialog)
            
        return list(chain.from_iterable(words))
    
    def make_context_vector(self, context):
        idxs = [self.w2i[w] for w in context]
        tensor = torch.LongTensor(idxs)
        return tensor

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0 : #Handle negative indices
                key += len( self )
            if key < 0 or key >= len(self) :
                raise IndexError("The index ({}) is out of range.".format(key))
            
            item = self.json_data.iloc[key]

            # Flatten dialog and add caption into 1d array
            dialog = [word for line in item.dialog for word in line[0].split()]
            dialog.extend(item.caption.split(' '))
            dialog = self.make_context_vector(dialog)

            img_ids = np.array(item.img_list)
            img_features = [self.img_features[idx] for idx in map(lambda x: self.img2feat[str(x)], img_ids)]
            img_features = np.array(img_features)
            img_features = torch.FloatTensor(img_features)

            target = item.target
            target = torch.LongTensor(np.array([target]))

            if torch.cuda.is_available():
                dialog, img_features, target = dialog.cuda(), img_features.cuda(), target.cuda()
                
            return dialog, img_features, target


# In[2]:


SAMPLE_EASY = ['Data', 'sample_easy.json']
TRAIN_EASY = ['Data', 'Easy', 'IR_train_easy.json']
EASY_1000 = ['Data', 'Easy', 'IR_train_easy_1000.json']
VALID_EASY = ['Data', 'Easy', 'IR_val_easy.json']
IMG_FEATURES = ['Data', 'Features', 'IR_image_features.h5']
INDEX_MAP = ['Data', 'Features', 'IR_img_features2id.json']

IMG_SIZE = 2048
EMBEDDING_DIM = 50

torch.manual_seed(1)
# dialog_data = DialogDataset(os.path.join(*SAMPLE_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
dialog_data = DialogDataset(os.path.join(*TRAIN_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
valid_data = DialogDataset(os.path.join(*VALID_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))

vocab_size = len(dialog_data.vocab)

print(len(dialog_data[0:3])) # can now slice this bitch up


# In[3]:


import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, output_dim)
        self.activation_function2 = nn.ReLU()
        

    def forward(self, inputs):
        # i believe .view() is useless here because the sum already produces a 1xEMB_DIM vector
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out


# In[4]:


class MaxEnt(torch.nn.Module):
    
    def __init__(self, text_module, text_dim, img_size):
        super(MaxEnt, self).__init__()

        self.text_module = text_module
        self.linear = nn.Linear(text_dim + img_size, 128)
        self.linear2 = nn.Linear(128, 1)
        self.softmax = nn.LogSoftmax()
        
    def prepare (self, dialog, imgFeatures):
        text_features = self.text_module(Variable(dialog))
        text_features = text_features.expand(imgFeatures.size(0), text_features.size(1))
        concat = torch.cat((imgFeatures, text_features.data), 1)
        return concat
    
    def prepareBatch (self, batch):
        inputs = []
        targets = []
        for dialog, imgFeatures, target in batch:
            inputs.append(self.prepare(dialog, imgFeatures))
            targets.append(target)
        inputs = torch.cat(inputs)
        targets = torch.cat(targets)
        return Variable(inputs), Variable(targets)
        
    def forward(self, inp, batch_size=1):
        scores = self.linear(inp)
        scores = F.relu(scores)
        scores = self.linear2(scores).view(batch_size, -1)
        scores = self.softmax(scores)
        return scores


# In[5]:


TEXT_DIM = 512

cbow_model = CBOW(vocab_size, EMBEDDING_DIM, TEXT_DIM)
model = MaxEnt(cbow_model, TEXT_DIM, IMG_SIZE)

if torch.cuda.is_available():
    print("ya ya")
    cbow_model = cbow_model.cuda()
    model = model.cuda()
    print("cuda ready bitches")
else:
    print("no no")
    
training_errors = []
validation_errors = []
epochsTrained = 0


# In[15]:


def validate(model, data, loss_func):
    total_loss = 0
    
    for i, (dialog, imgFeatures, target) in enumerate(data):
        inputs = model.prepare(dialog, imgFeatures)
        
        inputs, target = Variable(inputs), Variable(target)
        
        pred = model(inputs)
        
        loss = loss_func(pred, target)
        total_loss += loss.data[0]
    
    return total_loss / len(data)

def predict(model, data):
    correct_top1 = 0
    correct_top5 = 0
    
    for i, (dialog, img_features, target) in enumerate(data):
        
        inputs = model.prepare(dialog, img_features)
        inputs, target = Variable(inputs), Variable(target)
        
        # For top 1:
        pred = model(inputs)
        img, idx = torch.max(pred, 1)

        if idx.data[0] == target.data[0]:
            correct_top1 += 1
        
        # For top 5:
        pred = pred.data.numpy().flatten()
        top_5 = heapq.nlargest(5, range(len(pred)), pred.__getitem__)
        if target.data[0] in top_5:
            correct_top5 += 1
            
    return correct_top1 / len(data), correct_top5 / len(data)

validate(model, valid_data[:100], nn.NLLLoss())


# In[19]:


def log_to_console(i, n_epochs, batch_size, batch_per_epoch, error, start_time):
    avgProcessingSpeed = (i*batch_size) / (time.time() - start_time)
    percentOfEpc = (i / batch_per_epoch) * 100
    print("{:.0f}s:\t epoch: {}\t batch:{} ({:.1f}%) \t training error: {:.6f}\t speed: {:.1f} dialogs/s"
          .format(time.time() - start_time, 
                  n_epochs, 
                  i, 
                  percentOfEpc, 
                  error, 
                  avgProcessingSpeed))
    
def init_stats_log(label, training_portion, validation_portion, embeddings_dim, epochs, batch_count, learning_rate):
    timestr = time.strftime("%m-%d-%H-%M")
    filename = "{}-t_size_{}-v_size_{}-emb_{}-eps_{}-dt_{}-batch_{}-lr_{}.txt".format(label,
                                                                       training_portion,
                                                                       validation_portion,
                                                                       EMBEDDING_DIM,
                                                                       epochs,
                                                                       timestr,
                                                                       batch_count,
                                                                       learning_rate)

    target_path = ['Training_recordings', filename]
    stats_log = open(os.path.join(*target_path), 'w')
    stats_log.write("EPOCH|AVG_LOSS|TOT_LOSS|VAL_ERROR|CORRECT_TOP1|CORRECT_TOP5\n")
    print("Logging enabled in:", filename)
    
    return stats_log, filename
    


# In[ ]:


batchSize = 10
numEpochs = 25
learningRate = 1e-4
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

startTime = time.time()
lastPrintTime = startTime

continueFromEpc = 0
continueFromI = 0
sampleCount = len(dialog_data)
batchCountPerEpc = int(sampleCount/batchSize)-1
remainderCount = sampleCount - batchCountPerEpc * batchSize
print("we have: {} dialogs, batch size of {} with {} as remainder to offset batches each epoch".format(sampleCount, batchSize, remainderCount))
offset = 0

logging = True

training_portion = len(dialog_data)
validation_portion = len(valid_data)

if logging == True:
    stats_log, filename = init_stats_log("naive_cbow", 
                               training_portion,
                               validation_portion,
                               EMBEDDING_DIM,
                               numEpochs,
                               batchSize,
                               learningRate)

else:
    print("Logging disabled!")
    filename = ""

for t in range(numEpochs):
    lastPrintTime = time.time()
    epochStartTime = time.time()
    
    total_loss = 0
    updates = 0
    
    if t == 0 and continueFromI > 0:
        # continue where I crashed
        print("continuing")
        model.load_state_dict(torch.load('maxent_{}epc_{}iter.pt'.format(continueFromEpc, continueFromI+1)))
    
    for i in range(continueFromI, batchCountPerEpc):
        
        # In case of RNN, clear hidden state
        #model.hidden = steerNet.init_hidden(batchSize)
        
        batchBegin = offset + i * batchSize
        batchEnd = batchBegin + batchSize
        
        batch = dialog_data[batchBegin:batchEnd]
        inputs, targets = model.prepareBatch(batch)
        
        predictions = model(inputs, batchSize)
        
        loss = criterion(predictions, targets)
        training_errors.append(loss.data[0])
        total_loss += loss.data[0]
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if time.time()  - lastPrintTime > 10:
            log_to_console(i, t, batchSize, batchCountPerEpc, total_loss / i, startTime)
            lastPrintTime = time.time()
            
    print("{:.1f}s:\t Finished epoch. Calculating test error..".format(time.time() - startTime))
    
    avg_loss = total_loss / batchCountPerEpc
    top_1_score, top_5_score = predict(model, valid_data)
    validation_error = validate(model, valid_data, criterion)
    
    if logging == True:
        stats_log.write("{}|{}|{}|{}|{}|{}\n".format(t, avg_loss, total_loss, validation_error, top_1_score, top_5_score))
            
    epochsTrained += 1
    offset = (offset + 1) % remainderCount
    print()
    print("<--------------->")
    print("{:.1f}s:\t top-1: \t {:.2f} \t top-5: \t {:.2f} \t test error: {:.6f}".format(time.time() - startTime, top_1_score, top_5_score, validation_error))
    print("<--------------->")
    print()
    continueFromI = 0

if logging == True:
    stats_log.close()


# In[59]:


import matplotlib.pyplot as plt

def draw_graph(filename):
    
    
    # Read file and data
    with open("Training_recordings/" + filename, 'r') as f:
        data = [x.strip() for x in f.readlines()] 
    
    data = np.array([line.split("|") for line in data[1:]]).T
    
    epochs, avg_loss, total_loss, val_error, correct_top_1, correct_top_5 = data
    
    epochs = np.array(epochs, dtype=np.int8)
    
    plt.subplot(4, 1, 1)
    plt.plot(epochs, np.array(avg_loss, dtype=np.float32), '.-')
    plt.title('average loss, validation error and correct predictions')
    plt.ylabel('Average\nLoss')
    plt.xlabel('Epochs')

    plt.subplot(4, 1, 2)
    plt.plot(epochs, np.array(val_error, dtype=np.float32), '-')
    plt.ylabel('Validation\nLoss')
    plt.xlabel('Epochs')

    plt.subplot(4, 1, 3)
    plt.plot(epochs, np.array(correct_top_1, dtype=np.int8), '-')
    plt.ylabel('Correct\ntop 1')
    plt.xlabel('Epochs')

    plt.subplot(4, 1, 4)
    plt.plot(epochs, np.array(correct_top_5, dtype=np.int8), '-')
    plt.ylabel('Correct\ntop 5')
    plt.xlabel('Epochs')

    
    
    plt.show()

draw_graph(filename)

