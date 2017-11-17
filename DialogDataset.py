import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

import pprint
from collections import Counter,defaultdict

SAMPLE_EASY = ['Data', 'sample_easy.json']
TRAIN_EASY = ['Data', 'Easy', 'IR_train_easy.json']

class DialogDataset(Dataset):
    def __init__(self, json_data, transform=None):
        self.json_data = pd.read_json(json_data, orient='index')

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data.iloc[idx]
        print(item.dialog[0])

        # Flatten dialog and add caption into 1d array
        dialog = [word for line in item.dialog for word in line[0].split()]
        dialog.extend(item.caption.split(' '))
        #words = np.array(dialog)

        img_ids = np.array(item.img_list)
        target = np.array([item.target, item.target_img_id])

        return {'dialog':dialog, 'img_ids':item.img_list, 'target':item.target_img_id}

def show_batch(sample_batched):
    print(sample_batched)


def createEmbeddings (words, threshold):
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    wordCounts = Counter()

    # count all the words in lower case
    for word in words:
        wordCounts[word.lower()] += 1

    # index all words that occured at least n times
    for word, count in wordCounts.most_common():
        if count >= threshold:
            i2w[w2i[word]] = word
        else:
            break

    return w2i, i2w


# todo: collect all the words from dialogs and 
# captions and use them to create embedding map

dd = DialogDataset(os.path.join(*SAMPLE_EASY))
print(dd[0])
loader = DataLoader(dd, batch_size=4, shuffle=True, num_workers=4)

for batch_num, sample in enumerate(loader):
    show_batch(sample)
    if batch_num == 3:
        break
