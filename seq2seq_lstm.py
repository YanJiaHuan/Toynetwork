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
import zh_core_web_sm
spacy_ger = zh_core_web_sm.load()
spacy_eng = en_core_web_sm.load()

### help functions ###
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

text = 'I like apples'
doc = spacy_eng(text)
for token in doc:
    print(token.text, token.pos_, token.dep_)
