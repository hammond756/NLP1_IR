import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

import pprint

SAMPLE_EASY = ['Data', 'sample_easy.json']
TRAIN_EASY = ['Data', 'Easy', 'IR_train_easy.json']

class DialogDataset(Dataset):
    def __init__(self, json_data, transform=None):
        self.json_data = pd.read_json(json_data, orient='index')

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data.iloc[idx]

        # Flatten dialog and add caption into 1d array
        dialog = [word for line in item.dialog for word in line[0].split()]
        dialog.extend(item.caption.split(' '))
        words = np.array(dialog)

        img_ids = np.array(item.img_list)
        target = np.array([item.target, item.target_img_id])

        return np.array([words, img_ids, target])

def show_batch(sample_batched):
    print(sample_batched)

dd = DialogDataset(os.path.join(*SAMPLE_EASY))
print(dd[0])
loader = DataLoader(dd, batch_size=4, shuffle=True, num_workers=4)

for batch_num, sample in enumerate(loader):
    show_batch(sample)
    if batch_num == 3: break
