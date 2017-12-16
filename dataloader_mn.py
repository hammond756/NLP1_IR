
# coding: utf-8

# In[678]:


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
from timeit import default_timer as timer

import sys
import pprint
from collections import Counter,defaultdict
from itertools import chain

PAD = "<pad>"

class DialogDataset(Dataset):
    
    def __init__(self, json_data, image_features, img2feat, transform=None):
        
        with open(img2feat, 'r') as f:
            self.img2feat = json.load(f)['IR_imgid2id']
            
        self.img_features = np.asarray(h5py.File(image_features, 'r')['img_features'])
        self.json_data = pd.read_json(json_data, orient='index')
        self.corpus = self.get_words()
        self.vocab = list(set(self.corpus))
        self.vocab.append(PAD)
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
    
    def convert_to_idx(self, sequence):
        return [self.w2i[w] for w in sequence]
    
    def pad(self, dialog):
        # length of longest question/answer pair
        n = max(map(len, dialog))
        return [sentence + [PAD] * (n - len(sentence)) for sentence in dialog]
        

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
        
        diag = item.dialog
        capt = [item.caption]

        # No appending required if it is already done before:
        if diag[-1] != capt:
            diag.append(capt)
        diag = [QA[0].split() for QA in diag]
        diag = self.pad(diag)
        
        try:
            diag = [self.convert_to_idx(QA) for QA in diag]
        except:
            print(diag)

        diag = torch.LongTensor(diag)

        img_ids = np.array(item.img_list)
        img_features = [self.img_features[idx] for idx in map(lambda x: self.img2feat[str(x)], img_ids)]
        img_features = np.array(img_features)
        img_features = torch.FloatTensor(img_features)

        target = item.target
        target = torch.LongTensor(np.array([target]))

        if torch.cuda.is_available():
            diag, img_features, target = diag.cuda(), img_features.cuda(), target.cuda()

        return diag, img_features, target


# In[679]:


SAMPLE_EASY = ['Data', 'sample_easy.json']
TRAIN_EASY = ['Data', 'Easy', 'IR_train_easy.json']
EASY_1000 = ['Data', 'Easy', 'IR_train_easy_1000.json']
VAL_200 = ['Data', 'Easy', 'IR_val_easy_200.json']
VALID_EASY = ['Data', 'Easy', 'IR_val_easy.json']
IMG_FEATURES = ['Data', 'Features', 'IR_image_features.h5']
INDEX_MAP = ['Data', 'Features', 'IR_img_features2id.json']

IMG_SIZE = 2048
EMBEDDING_DIM = 5

torch.manual_seed(1)
# dialog_data = DialogDataset(os.path.join(*SAMPLE_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
dialog_data = DialogDataset(os.path.join(*TRAIN_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
valid_data = DialogDataset(os.path.join(*VALID_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))

vocab_size = len(dialog_data.vocab)


# In[680]:


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
        
    def forward(self, inputs, batch_size = 1):
        # i believe .view() is useless here because the sum already produces a 1xEMB_DIM vector
        embeds = self.embeddings(inputs)
        sum_dim = 1 if batch_size > 1 else 0
        embeds = torch.sum(embeds, sum_dim)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out


# In[681]:


class MaxEnt(torch.nn.Module):
    
    def __init__(self, text_module, vocab_size, img_size):
        super(MaxEnt, self).__init__()

        self.text_module = text_module
        self.linear = nn.Linear(vocab_size + img_size, 1)
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
        
    def forward(self, inp, batch_size = 1):
        scores = self.linear(inp).view(batch_size, -1)
        scores = self.softmax(scores)
        return scores


# In[682]:


class MemNet(nn.Module):
    """NB: output size of text_module should match memory_dim"""
    def __init__(self, text_module, memory_dim, output_dim):
        super(MemNet, self).__init__()
        
        self.output_dim = output_dim
        self.memory_dim = memory_dim
        
        self.text_module = text_module
        self.linear = nn.Linear(memory_dim, output_dim)
    
    def forward(self, dialog, img_features):
        
#         scores = torch.FloatTensor(len(img_features)).zero_()
        
#         for i, img_feature in enumerate(img_features):
        history = self.text_module(dialog, batch_size = len(dialog))
        history = history.expand(img_features.size(0), history.size(0), history.size(1))
        
        # inner product of history and current image (normalized to prevent extreme values in softmax)
        memory = torch.bmm(history, img_features.unsqueeze(1).transpose(1, 2))
        norms = torch.norm(memory, dim=1, keepdim=True)
        memory = torch.div(memory, norms)

        # Matrix has to be squeezed for the softmax to perform well
        weights = F.softmax(memory.squeeze(2))
        
        # Take weighted sum of history (unsqueeze and transpose to fit matrix multiplication)
        history_vector = torch.bmm(weights.unsqueeze(2).transpose(1,2), history).squeeze()

        # Add weighted history vec to images and pass through linear layer
        combined_features = history_vector + img_features
        scores = self.linear(combined_features)
        scores = F.log_softmax(scores.squeeze())
                
        return scores
        


# In[683]:


# Instantiate CBOW here for sentence embeddings:
# torch.manual_seed(1)

EMBEDDING_DIM = 100
IMG_SIZE = 2048
OUTPUT_DIM = IMG_SIZE # For simplicity...

cbow_model = CBOW(vocab_size, EMBEDDING_DIM, OUTPUT_DIM)
mem_net = MemNet(cbow_model, OUTPUT_DIM, 1)
max_ent = MaxEnt(cbow_model, vocab_size, IMG_SIZE)

if torch.cuda.is_available():
        cbow_model = cbow_model.cuda()
        mem_net = mem_net.cuda()

# In[686]:


def validate(model, data, loss_func):
    total_loss = 0
    
    for i, (dialog, images, target) in enumerate(data):
        
        pred = model(Variable(dialog), Variable(images)).unsqueeze(0)
        target = Variable(target)
        
        loss = loss_func(pred, target)
        total_loss += loss.data[0]

    return total_loss / len(data)

def predict(model, data):
    correct_top1 = 0
    correct_top5 = 0
    
    for i, (dialog, images, target) in enumerate(data):
        
        # For top 1:
        pred = model(Variable(dialog), Variable(images)).unsqueeze(0)
        target = Variable(target)
        
        img, idx = torch.max(pred, 1)

        if idx.data[0] == target.data[0]:
            correct_top1 += 1
        
        # For top 5:
        pred = pred.data.cpu().numpy().flatten()
        top_5 = heapq.nlargest(5, range(len(pred)), pred.__getitem__)
        if target.data[0] in top_5:
            correct_top5 += 1
    
    return correct_top1 / len(data), correct_top5 / len(data)

validate(mem_net, valid_data, nn.NLLLoss())
predict(mem_net, valid_data)


# In[688]:


def log_to_console(i, n_epochs, batch_size, batch_per_epoch, error, start_time, processing_speed):
    percentOfEpc = (i / batch_per_epoch) * 100
    print("{:.0f}s:\t epoch: {}\t batch:{} ({:.1f}%) \t training error: {:.6f}\t speed: {:.1f} dialogs/s"
          .format(timer() - start_time, 
                  n_epochs, 
                  i, 
                  percentOfEpc, 
                  error, 
                  processing_speed))
    
def init_stats_log(label, training_portion, validation_portion, embeddings_dim, epochs, batch_count, learningRate):
    timestr = time.strftime("%m-%d-%H-%M")
    filename = "{}-t_size_{}-v_size_{}-emb_{}-eps_{}-dt_{}-batch_{}-lr_{}.txt".format(label,
                                                                       training_portion,
                                                                       validation_portion,
                                                                       EMBEDDING_DIM,
                                                                       epochs,
                                                                       timestr,
                                                                       batch_count,
                                                                       learningRate)

    target_path = ['Training_recordings', filename]
    stats_log = open(os.path.join(*target_path), 'w')
    stats_log.write("EPOCH|AVG_LOSS|TOT_LOSS|VAL_ERROR|CORRECT_TOP1|CORRECT_TOP5\n")
    print("Logging enabled in:", filename)
    
    return stats_log, filename
    


# In[704]:


model = mem_net

batchSize = 1
numEpochs = 5
learningRate = 1e-4
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

startTime = timer()
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
    stats_log, filename = init_stats_log("memory_net", 
                               training_portion,
                               validation_portion,
                               EMBEDDING_DIM,
                               numEpochs,
                               batchSize, learningRate)

else:
    print("Logging disabled!")
    filename = ""

for t in range(numEpochs):
    lastPrintTime = timer()
    epoch_start_time = timer()
    
    total_loss = 0
    
    for i in range(continueFromI, batchCountPerEpc):
        
        # In case of RNN, clear hidden state
        #model.hidden = steerNet.init_hidden(batchSize)
        
        batchBegin = offset + i * batchSize
        batchEnd = batchBegin + batchSize
        
        dialog, images, targets = dialog_data[batchBegin:batchEnd][0]
        
        predictions = model(Variable(dialog), Variable(images))
        
        loss = criterion(predictions.view(batchSize, -1), Variable(targets))
        total_loss += loss.data[0]
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if time.time()  - lastPrintTime > 2 and i is not 0:
            processing_speed = (i*batchSize) / (timer() - epoch_start_time)
            log_to_console(i, t, batchSize, batchCountPerEpc, total_loss / i, startTime, processing_speed)
            lastPrintTime = time.time()
    
    avg_loss = total_loss / training_portion
    top_1_score, top_5_score = predict(model, valid_data)
    validation_error = validate(model, valid_data, criterion)
    
    if logging == True:
        stats_log.write("{}|{}|{}|{}|{}|{}\n".format(epoch, avg_loss, total_loss, validation_error, top_1_score, top_5_score))
            
    offset = (offset + 1) % remainderCount
    print("{:.1f}s:\t Finished epoch. Calculating test error..".format(timer() - startTime))
    print("{:.1f}s:\t top_1:\t{:.2f}\t top_5: \t {:.2f} \t test error: {:.6f}".format(timer() - startTime, top_1_score, top_5_score, validation_error))
    continueFromI = 0

if logging == True:
    stats_log.close()


# In[ ]:


def draw_graph(filename=trained_stats_file):
    
    
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

draw_graph()

