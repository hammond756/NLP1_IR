
# coding: utf-8

# In[1]:


# source: https://www.daniweb.com/programming/software-development/code/216839/number-to-word-converter-python
def int2word(n):
    """
    convert an integer number n into a string of english words
    """
    
    # Return any string that is not all digits
    if not all([char.isdigit() for char in n]):
        return n
    
    # break the number into groups of 3 digits using slicing
    # each group representing hundred, thousand, million, billion, ...
    n3 = []
    r1 = ""
    # create numeric string
    ns = str(n)
    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        # break if end of ns has been reached
        if q < -2:
            break
        else:
            if  q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r
    
    #print n3  # test
    
    # break each group of 3 digits into
    # ones, tens/twenties, hundreds
    # and form a string
    nw = ""
    for i, x in enumerate(n3):
        b1 = x % 10
        b2 = (x % 100)//10
        b3 = (x % 1000)//100
        #print b1, b2, b3  # test
        if x == 0:
            continue  # skip
        else:
            t = thousands[i]
        if b2 == 0:
            nw = ones[b1] + t + nw
        elif b2 == 1:
            nw = tens[b1] + t + nw
        elif b2 > 1:
            nw = twenties[b2] + ones[b1] + t + nw
        if b3 > 0:
            nw = ones[b3] + "hundred " + nw
    return nw.strip().split()

############# globals ################
ones = ["", "one ","two ","three ","four ", "five ",
    "six ","seven ","eight ","nine "]
tens = ["ten ","eleven ","twelve ","thirteen ", "fourteen ",
    "fifteen ","sixteen ","seventeen ","eighteen ","nineteen "]
twenties = ["","","twenty ","thirty ","forty ",
    "fifty ","sixty ","seventy ","eighty ","ninety "]
thousands = ["","thousand ","million ", "billion ", "trillion ",
    "quadrillion ", "quintillion ", "sextillion ", "septillion ","octillion ",
    "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ",
    "quattuordecillion ", "quindecillion", "sexdecillion ", "septendecillion ", 
    "octodecillion ", "novemdecillion ", "vigintillion "]

def digits_to_text(document):
    digits_to_text = []
    for token in document:
        temp = int2word(token)
        if type(temp) is list:
            digits_to_text.extend(temp)
        else:
            digits_to_text.append(temp)

    return digits_to_text


# In[2]:


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

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# TODO: find scientific reference that also claims Snowball is better
# alternatively: http://www.nltk.org/howto/stem.html claims this already.
from nltk.stem import SnowballStemmer, PorterStemmer

# check if stopword corpus is available on your system
try:
    _ = stopwords.words('english')
except:
    nltk.download('stopwords')
    
try:
    _ = WordNetLemmatizer().lemmatize('test')
except:
    nltk.download('wordnet')
    
# Embeddings don't work well for words that occur < 5 times
THRESHOLD = 5
UNK = "<unk>"

def hide_infrequent_words(document, threshold):
    counter = Counter(document)
    new_document = []
    
    for word in document:
        if counter[word] > threshold:
            new_document.append(word)
    
    return new_document

def filter_document(document):
    """Filter list of words based on some conventional methods, like removing stopwords and
    lemmatization"""

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stop_words.update(punctuation)
    document = list(filter(lambda x: x not in stop_words, document))

    # [I, am, 34] -> [I, am, thirty, four]
    document = digits_to_text(document)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    document = list(map(lemmatizer.lemmatize, document))

    return document

class DialogDataset(Dataset):
    
    def __init__(self, json_data, image_features, img2feat, transform=None):
        
        with open(img2feat, 'r') as f:
            self.img2feat = json.load(f)['IR_imgid2id']
            
        self.img_features = np.asarray(h5py.File(image_features, 'r')['img_features'])
        self.json_data = pd.read_json(json_data, orient='index')
        self.corpus = self.get_words()
        self.corpus = filter_document(self.corpus)
        self.corpus = hide_infrequent_words(self.corpus, THRESHOLD)
        self.vocab = list(set(self.corpus))
        
        self.vocab.append(UNK)
        
        self.w2i = {word : i for i, word in enumerate(self.vocab)}
        self.w2i = defaultdict(lambda: self.w2i[UNK], self.w2i)
        
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
            dialog = filter_document(dialog)
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


# In[3]:


SAMPLE_EASY = ['Data', 'sample_easy.json']
TRAIN_EASY = ['Data', 'Easy', 'IR_train_easy.json']
EASY_1000 = ['Data', 'Easy', 'IR_train_easy_1000.json']
VAL_200 = ['Data', 'Easy', 'IR_val_easy_200.json']
VALID_EASY = ['Data', 'Easy', 'IR_val_easy.json']
IMG_FEATURES = ['Data', 'Features', 'IR_image_features.h5']
INDEX_MAP = ['Data', 'Features', 'IR_img_features2id.json']

IMG_SIZE = 2048
EMBEDDING_DIM = 200 

torch.manual_seed(1)
# dialog_data = DialogDataset(os.path.join(*SAMPLE_EASY), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
dialog_data = DialogDataset(os.path.join(*EASY_1000), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))
valid_data = DialogDataset(os.path.join(*VAL_200), os.path.join(*IMG_FEATURES), os.path.join(*INDEX_MAP))

vocab_size = len(dialog_data.vocab)
print(len(dialog_data[0:3])) # can now slice this bitch up


# In[4]:


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


# In[5]:


class MaxEnt(torch.nn.Module):
    
    def __init__(self, text_module, text_dim, img_size, hidden_units):
        super(MaxEnt, self).__init__()

        self.text_module = text_module
        self.linear = nn.Linear(text_dim + img_size, hidden_units)
        self.linear2 = nn.Linear(hidden_units, 1)
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


# In[6]:


TEXT_DIM = 512
HIDDEN_UNITS_MAXENT = 1024

cbow_model = CBOW(vocab_size, EMBEDDING_DIM, TEXT_DIM)
model = MaxEnt(cbow_model, TEXT_DIM, IMG_SIZE, HIDDEN_UNITS_MAXENT)

if torch.cuda.is_available():
    print("ya ya")
    cbow_model = cbow_model.cuda()
    model = model.cuda()
    print("cuda ready bitches")
else:
    print("no no")
    
training_errors = []
validation_errors = []
epochs_trained = 0

dialog, images, target = dialog_data[0]
get_ipython().run_line_magic('time', 'inputs = model.prepare(dialog, images)')
get_ipython().run_line_magic('time', 'model(Variable(inputs))')


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
        pred = pred.data.cpu().numpy().flatten()
        top_5 = heapq.nlargest(5, range(len(pred)), pred.__getitem__)
        if target.data[0] in top_5:
            correct_top5 += 1
    
    return correct_top1 / len(data), correct_top5 / len(data)

validate(model, valid_data, nn.NLLLoss())


# In[10]:


def log_to_console(i, n_epochs, batch_size, batch_per_epoch, error, start_time, processing_speed):
    percentOfEpc = (i / batch_per_epoch) * 100
    print("{:.0f}s:\t epoch: {}\t batch:{} ({:.1f}%) \t training error: {:.6f}\t speed: {:.1f} dialogs/s"
          .format(timer() - start_time, 
                  n_epochs, 
                  i, 
                  percentOfEpc, 
                  error, 
                  processing_speed))
    
def init_stats_log(label, training_portion, validation_portion, embeddings_dim, epochs, batch_count, learning_rate, weight_dec, hidden):
    timestr = time.strftime("%m-%d-%H-%M")
    filename = "{}-t_size_{}-v_size_{}-emb_{}-eps_{}-dt_{}-batch_{}-lr_{}-wd_{}-hidden_{}.txt".format(label,
                                                                       training_portion,
                                                                       validation_portion,
                                                                       EMBEDDING_DIM,
                                                                       epochs,
                                                                       timestr,
                                                                       batch_count,
                                                                       learning_rate, weight_dec, hidden)

    target_path = ['Training_recordings', filename]
    stats_log = open(os.path.join(*target_path), 'w')
    stats_log.write("EPOCH|AVG_LOSS|TOT_LOSS|VAL_ERROR|CORRECT_TOP1|CORRECT_TOP5\n")
    print("Logging enabled in:", filename)
    
    return stats_log, filename
    


# In[15]:


batch_size = 30 
numEpochs = 25
learningRate = 1e-4
weight_decay = 1e-6 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weight_decay)

start_time = timer()
lastPrintTime = start_time

continueFromEpc = 0
continueFromI = 0
sampleCount = len(dialog_data)
batchCountPerEpc = int(sampleCount/batch_size)-1
remainderCount = sampleCount - batchCountPerEpc * batch_size
print("we have: {} dialogs, batch size of {} with {} as remainder to offset batches each epoch".format(sampleCount, batch_size, remainderCount))
offset = 0

logging = True

training_portion = len(dialog_data)
validation_portion = len(valid_data)

if logging == True:
    stats_log, filename = init_stats_log("l2reg_cbow_pre_easy", 
                               training_portion,
                               validation_portion,
                               EMBEDDING_DIM,
                               numEpochs,
                               batch_size,
                               learningRate, weight_decay, HIDDEN_UNITS_MAXENT)

else:
    print("Logging disabled!")
    filename = ""

for t in range(numEpochs):
    lastPrintTime = timer()
    epoch_start_time = timer()
    
    total_loss = 0
    updates = 0
    
    if t == 0 and continueFromI > 0:
        # continue where I crashed
        print("continuing")
        model.load_state_dict(torch.load('maxent_{}epc_{}iter.pt'.format(continueFromEpc, continueFromI+1)))
    
    for i in range(continueFromI, batchCountPerEpc):
        
        # In case of RNN, clear hidden state
        #model.hidden = steerNet.init_hidden(batch_size)
        
        batchBegin = offset + i * batch_size
        batchEnd = batchBegin + batch_size
        
        batch = dialog_data[batchBegin:batchEnd]
        inputs, targets = model.prepareBatch(batch)
        
        predictions = model(inputs, batch_size)
        
        loss = criterion(predictions, targets)
        training_errors.append(loss.data[0])
        total_loss += loss.data[0]
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if timer()  - lastPrintTime > 3:
            processing_speed = (i*batch_size) / (timer() - epoch_start_time)
            log_to_console(i, epochs_trained, batch_size, batchCountPerEpc, total_loss / i, start_time, processing_speed)
            lastPrintTime = timer()
    print("{:.1f}s:\t Finished epoch. Calculating test error..".format(timer() - start_time))
    
    avg_loss = total_loss / batchCountPerEpc
    top_1_score, top_5_score = predict(model, valid_data)
    validation_error = validate(model, valid_data, criterion)
    
    if logging == True:
        stats_log.write("{}|{}|{}|{}|{}|{}\n".format(epochs_trained, avg_loss, total_loss, validation_error, top_1_score, top_5_score))
            
    epochs_trained += 1
    offset = (offset + 1) % remainderCount
    print()
    print("<--------------->")
    print("{:.1f}s:\t top-1: \t {:.2f} \t top-5: \t {:.2f} \t test error: {:.6f}".format(timer() - start_time, top_1_score, top_5_score, validation_error))
    print("<--------------->")
    print()
    continueFromI = 0

for i in range(batch_size):
    dialog, image, target = dialog_data[i]
    inputs = model.prepare(dialog, image)
    pred = model(Variable(inputs))
    print(pred)

if logging == True:
    stats_log.close()
    path = os.path.join(*['saved_models', filename + '.h5'])
    torch.save(model.state_dict(), path)
