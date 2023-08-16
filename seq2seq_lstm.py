### lib ###
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext import data, datasets
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

### load data ###
import en_core_web_sm
import de_core_news_sm
spacy_ger = de_core_news_sm.load()
spacy_eng = en_core_web_sm.load()

### help functions ###
sos_tok = '<sos>'
eos_tok = '<eos>'
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

text = 'I like apples'
doc = spacy_eng(text)
for token in doc:
    print(token.text, token.pos_, token.dep_)

train_data, valid_data, test_data = Multi30k(root='.data',split=('train', 'valid', 'test'), language_pair=('de', 'en'))


class TextDatasets(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.datasets = list(raw_data)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        src, trg = self.datasets[idx]
        src = [sos_tok] + tokenize_ger(src) + [eos_tok]
        trg = [sos_tok] + tokenize_eng(trg) + [eos_tok]
        return src, trg


train_datasets = TextDatasets(train_data)
valid_datasets = TextDatasets(valid_data)
test_datasets = TextDatasets(test_data)

print("Train size:", len(train_datasets))
print("Valid size:", len(valid_datasets))
print("Test size:", len(test_datasets))

### network ###
class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass
















