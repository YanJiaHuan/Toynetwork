### lib ###
import io
import os
import unicodedata
import string
import glob
import torch
import torch.nn as nn
import matplotlib as plt
import random

### utils ###
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD',s)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
            )
def load_data(path):
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    def read_lines(filename):
        lines = io.open(filename, encoding = 'utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    for filename in find_files(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)
def letter_to_tensor(letter):
    tensor = torch.zeros(1,N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line),1,N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


### model ###

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # input to output
        self.softmax = nn.LogSoftmax(dim = 1)
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

category_lines, all_categories = load_data('data/names/*.txt') # 读取数据

